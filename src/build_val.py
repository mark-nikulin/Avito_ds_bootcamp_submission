"""Строит кандидатов с признаками для валидационного сплита (seed из argv) и кэширует."""
import sys, time, gc, resource
from common import *
from retriever import Retriever, build_candidates

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
pairs, q, _, train_items = load_data(with_bench=False)
tp, vq, vi, qr = make_split(pairs, train_items, seed=seed)
t = time.time()
# эмбеддинги корпуса валидации — строки emb_train_items по item_id корпуса
E_tr = np.load(CACHE / 'emb_train_items.npy')
E_vi = E_tr[pd.Index(train_items.item_id).get_indexer(vi.item_id)]
R = Retriever().fit(tp, vi, train_items, E_vi, E_tr)
del E_tr, E_vi
del pairs, train_items, tp, vi; gc.collect()
print('fit done', round(time.time() - t), 's, peak RSS GB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, flush=True)
C = build_candidates(R, vq)
C['label'] = C.set_index(['query_id', 'item_id']).index.isin(qr.set_index(['query_id', 'item_id']).index).astype(int)
C.to_parquet(CACHE / f'val_cand_{seed}.parquet')
qr.to_parquet(CACHE / f'val_qrels_{seed}.parquet')
rel = qr.groupby('query_id').size()
print('peak RSS GB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9)
print('seed', seed, 'time', time.time() - t, 'cands/query', len(C) / len(vq),
      'ceiling recall', (C[C.label == 1].groupby('query_id').size() / rel).fillna(0).reindex(rel.index).fillna(0).mean())
