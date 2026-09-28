"""Нейросетевые эмбеддинги объявлений и запросов.

Модель: intfloat/multilingual-e5-small (open-source, MIT, 118M параметров, 384-мерные
векторы). Запускается локально (CPU/MPS), внешних API нет; веса скачиваются с
HuggingFace один раз и кэшируются.

E5 требует префиксов: «query: » для запросов и «passage: » для документов.
Эмбеддинги объявлений считаются один раз и сохраняются в cache/*.npy (float16),
в том же порядке строк, что и cache/train_items.parquet / bench_items.parquet.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import CACHE, DATA, build_cache

MODEL_NAME = "intfloat/multilingual-e5-small"
_model = None


def get_model():
    global _model
    if _model is None:
        import torch
        from sentence_transformers import SentenceTransformer
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        _model = SentenceTransformer(MODEL_NAME, device=dev)
        _model.max_seq_length = 128  # заголовка + начала описания достаточно
    return _model


def encode(texts, prefix, batch_size=128):
    m = get_model()
    return m.encode([prefix + t for t in texts], batch_size=batch_size, normalize_embeddings=True,
                    show_progress_bar=False, convert_to_numpy=True).astype(np.float32)


def encode_queries(texts):
    return encode(list(texts), "query: ", batch_size=256)


def _item_texts(raw_path, ids):
    """Тексты «заголовок. начало описания» для item_id в заданном порядке (потоковое чтение)."""
    need = set(ids)
    out = {}
    for b in pq.ParquetFile(raw_path).iter_batches(batch_size=50_000, columns=["item_id", "item_title_raw", "item_description_raw"]):
        df = b.to_pandas()
        df = df[df.item_id.isin(need) & ~df.item_id.isin(out.keys())].drop_duplicates("item_id")
        for i, t, d in zip(df.item_id, df.item_title_raw.fillna(""), df.item_description_raw.fillna("")):
            out[i] = f"{t}. {' '.join(d[:300].split())}"
    return [out[i] for i in ids]


def build_item_embeddings():
    for name, raw in [("bench_items", DATA / "benchmark_items.parquet"), ("train_items", DATA / "train.parquet")]:
        path = CACHE / f"emb_{name}.npy"
        if path.exists():
            continue
        ids = pd.read_parquet(CACHE / f"{name}.parquet", columns=["item_id"]).item_id.tolist()
        texts = _item_texts(raw, ids)
        embs = []
        for s in range(0, len(texts), 20_000):
            embs.append(encode(texts[s:s + 20_000], "passage: ").astype(np.float16))
            print(name, s + len(embs[-1]), "/", len(texts), flush=True)
        np.save(path, np.concatenate(embs))


if __name__ == "__main__":
    build_cache()            # стемминг и компактный кэш данных (если ещё не построен)
    build_item_embeddings()
