"""Общие утилиты: загрузка данных, нормализация текста, валидационный сплит.

Валидация строится только из train.parquet и имитирует бенчмарк:
  * отложенные «запросы» — группы (текст, локация, фильтры, категория), по одной
    группе на уникальный текст запроса (в бенчмарке все тексты уникальны);
  * их строки (и все строки с их позитивными объявлениями) удаляются из обучающей
    части, а сам текст запроса может
    остаться в обучении в других группах (в бенчмарке ~37% текстов есть в train);
  * корпус = все положительные объявления отложенных групп + случайные прочие
    объявления train до размера ~189k, как у benchmark_items.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import Stemmer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "NLP_avito_interns-dataset"
CACHE = ROOT / "cache"
CACHE.mkdir(exist_ok=True)

GROUP_KEYS = ["search_query", "search_location_id", "search_infm_params_text", "search_category"]
ITEM_COLS = [
    "item_id", "item_title_raw", "item_description_raw", "item_infm_params_text",
    "item_category_id", "item_microcat_id", "item_price", "item_rating",
    "item_rating_reviews_count", "item_location_id", "item_latitude", "item_longitude",
    "item_is_phone_hidden", "item_is_message_forbidden",
]

_stemmer = Stemmer.Stemmer("russian")
_token_re = re.compile(r"[a-zа-я0-9]+")


def norm(text: str) -> str:
    """Нижний регистр, ё→е, только буквы/цифры через пробел."""
    if not isinstance(text, str):
        return ""
    return " ".join(_token_re.findall(text.lower().replace("ё", "е")))


def stem_text(text: str) -> str:
    """Нормализация + стемминг Snowball (PyStemmer) — сводит словоформы к основе."""
    toks = _token_re.findall(text.lower().replace("ё", "е")) if isinstance(text, str) else []
    return " ".join(_stemmer.stemWords(toks))


PAIR_COLS = GROUP_KEYS + ["search_is_delivery_search", "item_id", "item_location_id", "item_microcat_id"]
LIGHT_ITEM_COLS = ["item_id", "item_category_id", "item_microcat_id", "item_price", "item_rating",
                   "item_rating_reviews_count", "item_location_id", "item_latitude", "item_longitude"]


def prep_items(items: pd.DataFrame) -> pd.DataFrame:
    """Оставляет лёгкие колонки объявления + стемминговые тексты (сырые тексты выкидываем)."""
    it = items[ITEM_COLS].drop_duplicates("item_id")
    out = it[LIGHT_ITEM_COLS].copy()
    for c in ["item_price", "item_latitude", "item_longitude"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(np.float64)
    out["title_s"] = it.item_title_raw.map(stem_text).values
    # в параметрах много шаблонных повторов («Время для связи, дни недели пн. … вт. …»):
    # оставляем уникальные нецифровые стемы — в 3 раза меньше памяти, смысл сохраняется
    out["params_s"] = it.item_infm_params_text.map(lambda x: " ".join(dict.fromkeys(
        t for t in stem_text(x).split() if not t.isdigit()))).values
    # описания длинные: для ретривала берём начало — там обычно суть услуги
    out["desc_s"] = it.item_description_raw.fillna("").str[:1000].map(stem_text).values
    out["title_n"] = it.item_title_raw.map(norm).values
    return out.reset_index(drop=True)


def _stream_prep(path, pair_cols=None, batch_size=50_000):
    """Потоковая обработка parquet батчами: сырые описания никогда не лежат в памяти целиком.

    Возвращает (пары запрос-объявление или None, уникальные подготовленные объявления).
    """
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    pairs, items, seen = [], [], set()
    for b in pf.iter_batches(batch_size=batch_size):
        df = b.to_pandas()
        if pair_cols:
            pairs.append(df[pair_cols].copy())
        df = df[~df.item_id.isin(seen)].drop_duplicates("item_id")
        seen.update(df.item_id)
        items.append(prep_items(df))
        del df, b
    return (pd.concat(pairs, ignore_index=True) if pair_cols else None), pd.concat(items, ignore_index=True)


def build_cache():
    """Однократная предобработка (стемминг ~10 мин) → компактные parquet в cache/."""
    if not (CACHE / "train_pairs.parquet").exists():
        pairs, items = _stream_prep(DATA / "train.parquet", PAIR_COLS)
        pairs.to_parquet(CACHE / "train_pairs.parquet")
        items.to_parquet(CACHE / "train_items.parquet")
    if not (CACHE / "bench_items.parquet").exists():
        _, items = _stream_prep(DATA / "benchmark_items.parquet")
        items.to_parquet(CACHE / "bench_items.parquet")


def load_data(with_bench: bool = True):
    """(пары train, запросы бенчмарка, подготовленный корпус бенчмарка, подготовленные объявления train).

    with_bench=False — не читать корпус бенчмарка (для валидации он не нужен, экономим память).
    """
    build_cache()
    pairs = pd.read_parquet(CACHE / "train_pairs.parquet")
    q = pd.read_parquet(DATA / "benchmark_queries.parquet")
    bench_items = pd.read_parquet(CACHE / "bench_items.parquet") if with_bench else None
    train_items = pd.read_parquet(CACHE / "train_items.parquet")
    return pairs, q, bench_items, train_items


def make_split(pairs: pd.DataFrame, train_items: pd.DataFrame, n_val: int = 2500,
               corpus_size: int = 189_212, seed: int = 0):
    """Возвращает (train_part, val_queries, val_items, val_qrels).

    pairs — лёгкие пары запрос-объявление из train; train_items — подготовленные
    уникальные объявления train. val_qrels: DataFrame(query_id, item_id).
    """
    rng = np.random.default_rng(seed)
    gid = pairs.groupby(GROUP_KEYS, sort=False).ngroup()
    # по одной группе на каждый выбранный уникальный текст запроса
    first_gid = gid.groupby(pairs.search_query).agg(lambda s: rng.choice(s.unique()))
    texts = rng.choice(first_gid.index.values, size=n_val, replace=False)
    val_gids = set(first_gid.loc[texts].values)
    is_val = gid.isin(val_gids)
    val_rows = pairs[is_val].assign(query_id="v" + gid[is_val].astype(str))
    # убираем из обучения и все прочие строки с позитивами валидации: иначе модель
    # «запомнит» их заголовки через похожие запросы, чего на бенчмарке не будет
    # (там ~90% объявлений корпуса в train не встречаются)
    leak = pairs.item_id.isin(set(val_rows.item_id))
    train_part = pairs[~is_val & ~leak].reset_index(drop=True)
    vq = val_rows.drop_duplicates("query_id")[["query_id"] + GROUP_KEYS + ["search_is_delivery_search"]].reset_index(drop=True)
    qrels = val_rows[["query_id", "item_id"]].drop_duplicates()
    # корпус: позитивы + случайные дистракторы из объявлений train
    pos = train_items.item_id.isin(set(qrels.item_id))
    extra = train_items[~pos].sample(n=corpus_size - int(pos.sum()), random_state=seed)
    val_items = pd.concat([train_items[pos], extra]).reset_index(drop=True)
    return train_part, vq, val_items, qrels


def recall_at_k(pred: dict, qrels: pd.DataFrame, k: int = 50) -> float:
    """pred: query_id -> list(item_id). Метрика Recall@k, усреднённая по запросам."""
    rel = qrels.groupby("query_id").item_id.apply(set)
    return float(np.mean([len(set(pred.get(qid, [])[:k]) & r) / len(r) for qid, r in rel.items()]))
