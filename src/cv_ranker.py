"""2-fold CV ранкера по запросам внутри одного валидационного сплита."""
import sys, resource
import numpy as np, pandas as pd
from common import CACHE, recall_at_k
from ranker import add_query_relative, train_ranker, predict_top

seeds = [int(x) for x in sys.argv[1:]] or [0]
C = pd.concat([pd.read_parquet(CACHE / f'val_cand_{s}.parquet').assign(query_id=lambda d, s=s: f's{s}_' + d.query_id) for s in seeds])
qr = pd.concat([pd.read_parquet(CACHE / f'val_qrels_{s}.parquet').assign(query_id=lambda d, s=s: f's{s}_' + d.query_id) for s in seeds])
C = add_query_relative(C)
qids = C.query_id.unique(); rng = np.random.default_rng(0); fold = dict(zip(qids, rng.integers(0, 2, len(qids))))
C['fold'] = C.query_id.map(fold)
pred = {}
for f in (0, 1):
    m = train_ranker(C[C.fold != f])
    pred.update(predict_top(m, C[C.fold == f]))
    del m
# baseline: prescore
base = C.sort_values(['query_id', 'prescore'], ascending=[True, False]).groupby('query_id').item_id.apply(lambda s: list(s.iloc[:50])).to_dict()
print('prescore recall@50', recall_at_k(base, qr), ' ranker recall@50', recall_at_k(pred, qr))
print('peak RSS GB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9)
