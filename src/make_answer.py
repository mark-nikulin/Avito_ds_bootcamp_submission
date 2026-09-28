"""Финальный пайплайн: строит answer.csv для benchmark_queries.

1. Ретривер обучается на всём train.parquet, поиск — по benchmark_items.
2. Для каждого запроса строятся ~600 кандидатов с признаками.
3. Ранкер (HistGradientBoosting) обучается на кандидатах валидационных сплитов
   (cache/val_cand_*.parquet, их строит build_val.py) — там известна разметка.
4. Берём top-50 по скору ранкера, проверяем формат и сохраняем answer.csv.
"""
import gc, glob, resource, sys, time
import numpy as np
import pandas as pd
from common import CACHE, ROOT, load_data
from retriever import Retriever, build_candidates
from ranker import add_query_relative, train_ranker, predict_top

t = time.time()
pairs, q, bench_items, train_items = load_data()
R = Retriever().fit(pairs, bench_items, train_items,
                    np.load(CACHE / 'emb_bench_items.npy'), np.load(CACHE / 'emb_train_items.npy'))
corpus_ids = set(bench_items.item_id)
del pairs, bench_items, train_items; gc.collect()
C = add_query_relative(build_candidates(R, q))
del R; gc.collect()
print('candidates built', round(time.time() - t), 's; peak RSS GB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, flush=True)

# обучающие данные ранкера — все валидационные сплиты
files = sorted(glob.glob(str(CACHE / 'val_cand_*.parquet')))
V = add_query_relative(pd.concat([pd.read_parquet(f).assign(query_id=lambda d, i=i: f's{i}_' + d.query_id) for i, f in enumerate(files)], ignore_index=True))
model = train_ranker(V)
del V; gc.collect()
pred = predict_top(model, C, k=50)

# --- сборка и проверки формата
rows = []
for qid in q.query_id:
    ids = list(dict.fromkeys(pred.get(qid, [])))[:50]   # без повторов, не более 50
    assert all(len(x) == 16 and x in corpus_ids for x in ids)
    rows.append((qid, ' '.join(ids)))
ans = pd.DataFrame(rows, columns=['query_id', 'answer'])
assert ans.query_id.is_unique and len(ans) == len(q)
out = sys.argv[1] if len(sys.argv) > 1 else 'answer.csv'
ans.to_csv(ROOT / out, index=False)
print('saved', out, '; empty answers:', (ans.answer == '').sum(), '; mean len:', ans.answer.str.split().str.len().mean())
