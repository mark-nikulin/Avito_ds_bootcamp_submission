"""Ранжирующая модель поверх кандидатов.

Используем HistGradientBoostingClassifier (sklearn) — бинарная классификация
«объявление выбрано / не выбрано». Кроме абсолютных скоров добавляем признаки
относительно запроса: ранг скора внутри кандидатов запроса и отставание от
максимума — они переносимы между «лёгкими» и «трудными» запросами.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from retriever import FEATURES

REL_COLS = ["s_title", "s_desc", "s_char", "s_exp_title", "s_exp_params", "s_exp_desc", "p_micro", "prescore", "title_cover",
            "s_emb", "s_emb_exp", "s_emb_exp_txt", "p_micro_emb"]


def add_query_relative(C: pd.DataFrame) -> pd.DataFrame:
    g = C.groupby("query_id", sort=False)
    for c in REL_COLS:
        C[c + "_rank"] = g[c].rank(ascending=False, method="min").astype(np.float32)
        C[c + "_gap"] = (g[c].transform("max") - C[c]).astype(np.float32)
    C["n_cand"] = g["prescore"].transform("size").astype(np.float32)
    return C


def feature_cols():
    return FEATURES + ["prescore"] + [c + s for c in REL_COLS for s in ("_rank", "_gap")] + ["n_cand"]


def train_ranker(C: pd.DataFrame, seed: int = 0):
    cols = feature_cols()
    # даунсэмплинг негативов: позитивов мало, а кандидатов ~700 на запрос
    rng = np.random.default_rng(seed)
    keep = (C.label.values == 1) | (rng.random(len(C)) < 0.15)
    D = C[keep]
    model = HistGradientBoostingClassifier(
        max_iter=600, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=50,
        l2_regularization=1.0, random_state=seed,
    )
    model.fit(D[cols].values, D.label.values)
    return model


def predict_top(model, C: pd.DataFrame, k: int = 50) -> dict:
    C = C.assign(score=model.predict_proba(C[feature_cols()].values)[:, 1])
    C = C.sort_values(["query_id", "score"], ascending=[True, False])
    return C.groupby("query_id", sort=False).item_id.apply(lambda s: list(s.iloc[:k])).to_dict()
