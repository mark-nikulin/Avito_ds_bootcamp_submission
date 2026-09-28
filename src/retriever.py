"""Генерация кандидатов и признаков для пары (запрос, объявление).

Идея: для каждого запроса
  1. Пул = объявления в «допустимых» локациях: сама локация поиска + локации, куда
     пользователи из этой локации поиска ходили в train (матрица переходов).
  2. Для всех объявлений пула считаются дешёвые скоры (разреженные dot-продукты):
     * TF-IDF текста запроса против заголовка / параметров / описания (стемы слов);
     * TF-IDF символьных n-грамм запроса против заголовка (опечатки, морфология);
     * «расширенный запрос»: находим в train похожие тексты запросов (kNN) и берём
       заголовки/параметры объявлений, которые по ним выбирали → профиль в TF-IDF
       пространстве объявлений. Так учимся, что «автоподбор» ≈ «осмотр автомобиля»;
     * априорная вероятность подкатегории объявления (microcat) по kNN-запросам;
     * совпадение фильтров поиска с параметрами объявления;
     * вероятность перехода локаций, популярность локации, рейтинг/отзывы/цена.
  3. Предотбор top-N по простой сумме скоров, затем ранжирующая модель (ranker.py).
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer, TfidfVectorizer


class HashTfidf:
    """TF-IDF поверх HashingVectorizer: не хранит словарь в памяти.

    Обычный TfidfVectorizer сначала собирает полный словарь всех n-грамм
    (миллионы python-строк, пик 8+ ГБ), и лишь потом обрезает его. Хеширование в
    2^20 корзин даёт почти то же качество при в разы меньшем пике памяти.
    """

    def __init__(self, n_features=2 ** 20, **kw):
        self.hv = HashingVectorizer(n_features=n_features, alternate_sign=False, norm=None, dtype=np.float32, **kw)
        self.tt = TfidfTransformer(sublinear_tf=True)

    def fit(self, texts):
        self.tt.fit(self.hv.transform(texts))
        return self

    def transform(self, texts):
        return self.tt.transform(self.hv.transform(texts)).astype(np.float32)
from sklearn.preprocessing import normalize

from common import norm, stem_text

FEATURES = [
    "s_title", "s_params", "s_desc", "s_char", "s_exp_title", "s_exp_params",
    "s_exp_desc", "p_micro", "p_micro_exact", "s_filter", "loc_p", "same_loc",
    "loc_share", "knn_top_sim", "exact_in_train", "rating", "log_reviews",
    "log_price", "q_len", "title_cover",
    # нейросетевые (e5) признаки
    "s_emb", "s_emb_exp", "s_emb_exp_txt", "p_micro_emb", "knn_emb_top",
]


class Retriever:
    def __init__(self, k_neighbors: int = 40):
        self.k = k_neighbors

    # ------------------------------------------------------------------ fit
    def fit(self, train: pd.DataFrame, items: pd.DataFrame, train_items: pd.DataFrame,
            emb_items: np.ndarray, emb_train_items: np.ndarray):
        """train — лёгкие пары запрос-объявление; items — подготовленный корпус
        (common.prep_items), в котором ищем; train_items — подготовленные объявления train.

        Код написан с оглядкой на память (машина 16 ГБ): тексты уже стеммированы и
        закэшированы, векторизуются по уникальным объявлениям, а не по строкам train.
        emb_items / emb_train_items — e5-эмбеддинги, выровненные по строкам items / train_items.
        """
        it = items
        # только объявления, реально встречающиеся в обучающих парах
        used = train_items.item_id.isin(set(train.item_id)).values
        tr_items = train_items[used].reset_index(drop=True)
        E_tr = emb_train_items[used].astype(np.float32)
        self.E_items = np.ascontiguousarray(emb_items, dtype=np.float32)

        # --- TF-IDF по полям корпуса (словарь учим на корпусе + train-объявлениях)
        self.v_title = HashTfidf(ngram_range=(1, 2), token_pattern=r"\S+")
        self.v_title.fit(pd.concat([it.title_s, tr_items.title_s]))
        self.v_params = HashTfidf(n_features=2 ** 19, token_pattern=r"\S+")
        self.v_params.fit(pd.concat([it.params_s, tr_items.params_s]))
        self.v_desc = HashTfidf(token_pattern=r"\S+")
        self.v_desc.fit(it.desc_s)
        self.v_char = HashTfidf(analyzer="char_wb", ngram_range=(3, 5))
        self.v_char.fit(it.title_n)

        self.X_title = self.v_title.transform(it.title_s).tocsr()
        self.X_params = self.v_params.transform(it.params_s).tocsr()
        self.X_desc = self.v_desc.transform(it.desc_s).tocsr()
        self.X_char = self.v_char.transform(it.title_n).tocsr()
        # бинарные матрицы «объявление × стем» — для доли слов запроса / фильтра,
        # встречающихся в заголовке / параметрах (вместо тяжёлых python-множеств)
        self.v_bin = CountVectorizer(binary=True, dtype=np.float32, token_pattern=r"\S+")
        self.v_bin.fit(pd.concat([it.title_s, it.params_s]))
        self.B_title = self.v_bin.transform(it.title_s).tocsr()
        self.B_params = self.v_bin.transform(it.params_s).tocsr()

        # --- профиль «текст train-запроса → выбранные объявления»
        tr = train[["search_query", "item_id", "item_microcat_id"]].copy()
        tr["qn"] = tr.search_query.map(norm)
        qtexts = tr.qn.drop_duplicates().reset_index(drop=True)
        self.qtext_index = pd.Series(np.arange(len(qtexts)), index=qtexts.values)
        qi = self.qtext_index.loc[tr.qn].values
        ii = pd.Index(tr_items.item_id).get_indexer(tr.item_id)
        Pt = normalize(self.v_title.transform(tr_items.title_s))
        Pp = normalize(self.v_params.transform(tr_items.params_s))
        Pd = normalize(self.v_desc.transform(tr_items.desc_s))
        tr_micro = tr_items.item_microcat_id.values
        del tr_items
        # матрица агрегации «текст запроса × уникальное объявление train» (веса = число выборов)
        A = sp.csr_matrix((np.ones(len(tr), np.float32), (qi, ii)), shape=(len(qtexts), Pt.shape[0]))
        self.prof_title = normalize(A @ Pt).tocsr()
        self.prof_params = normalize(A @ Pp).tocsr()
        self.prof_desc = normalize(A @ Pd).tocsr()
        del Pt, Pp, Pd
        # эмбеддинг-профиль запроса: нормированное среднее эмбеддингов выбранных объявлений
        self.prof_emb = normalize(np.asarray(A @ E_tr, dtype=np.float32))
        del E_tr
        # распределение подкатегорий для каждого текста запроса
        micro = pd.Index(np.union1d(it.item_microcat_id.unique(), tr_micro))
        self.micro_index = micro
        mc = micro.get_indexer(tr.item_microcat_id)
        M = sp.csr_matrix((np.ones(len(tr), np.float32), (qi, mc)), shape=(len(qtexts), len(micro)))
        self.prof_micro = normalize(M, norm="l1").tocsr()
        self.item_micro = micro.get_indexer(it.item_microcat_id)

        # тексты больше не нужны — оставляем только лёгкие колонки корпуса
        self.items = it[["item_id", "item_location_id", "item_microcat_id", "item_rating",
                         "item_rating_reviews_count", "item_price"]].copy()
        del it
        it = self.items

        # kNN-индекс по текстам train-запросов: символьные n-граммы + стемы слов
        self.v_qchar = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        self.v_qword = TfidfVectorizer(sublinear_tf=True)
        qs = qtexts.map(stem_text)
        self.Q_train = normalize(sp.hstack([self.v_qchar.fit_transform(qtexts), self.v_qword.fit_transform(qs)]).astype(np.float32)).tocsr()
        # второй kNN-индекс — по e5-эмбеддингам текстов запросов (ловит синонимы)
        from embed import encode_queries
        self.QE_train = encode_queries(qtexts)

        # --- локации: вероятности переходов search_loc -> item_loc
        T = train.groupby(["search_location_id", "item_location_id"]).size().rename("n").reset_index()
        T["p"] = T.n / T.groupby("search_location_id").n.transform("sum")
        self.trans = {s: dict(zip(g.item_location_id, g.p)) for s, g in T.groupby("search_location_id")}
        self.loc_items = it.groupby("item_location_id").indices
        self.item_loc = it.item_location_id.values
        # доля объявлений каждой локации среди выборов train — «популярность» локации
        self.loc_share = it.item_location_id.map(train.item_location_id.value_counts(normalize=True)).fillna(0).values

        self.rating = it.item_rating.fillna(0).values
        self.log_reviews = np.log1p(it.item_rating_reviews_count.fillna(0).values)
        self.log_price = np.log1p(it.item_price.fillna(0).clip(lower=0).values)
        return self

    # ------------------------------------------------------------ helpers
    def _pool(self, loc):
        """Индексы объявлений допустимых локаций и вероятность перехода для каждого."""
        probs = dict(self.trans.get(loc, {}))
        probs.setdefault(loc, 0.0)
        idx, p = [], []
        for l, pr in probs.items():
            ii = self.loc_items.get(l)
            if ii is not None:
                idx.append(ii)
                p.append(np.full(len(ii), pr, np.float32))
        if not idx:
            # фолбэк: в допустимых локациях нет объявлений — ищем по всему корпусу
            return np.arange(len(self.item_loc)), np.zeros(len(self.item_loc), np.float32)
        return np.concatenate(idx), np.concatenate(p)

    def query_features(self, queries: pd.DataFrame, batch: int = 32):
        """Генератор: для каждого запроса (query_id, индексы пула, матрица признаков)."""
        qn = queries.search_query.map(norm)
        qs = queries.search_query.map(stem_text)
        fs = queries.search_infm_params_text.fillna("").map(stem_text)
        Qt = self.v_title.transform(qs).astype(np.float32)
        Qp = self.v_params.transform(qs).astype(np.float32)
        Qd = self.v_desc.transform(qs).astype(np.float32)
        Qc = self.v_char.transform(qn).astype(np.float32)
        Qk = normalize(sp.hstack([self.v_qchar.transform(qn), self.v_qword.transform(qs)]).astype(np.float32)).tocsr()
        # бинарные векторы слов запроса и слов фильтра + их длины
        Qb = self.v_bin.transform(qs).tocsr()
        Fb = self.v_bin.transform(fs).tocsr()
        q_len = np.array([len(set(x.split())) for x in qs], np.float32)
        f_len = np.array([len(set(x.split())) for x in fs], np.float32)
        from embed import encode_queries
        QE = encode_queries(qn)

        for b0 in range(0, len(queries), batch):
            sl = slice(b0, b0 + batch)
            # kNN по train-запросам
            S = (Qk[sl] @ self.Q_train.T).toarray()
            top = np.argsort(-S, axis=1)[:, : self.k]
            W = np.take_along_axis(S, top, axis=1)
            W = W ** 3  # сильнее доверяем близким соседям
            rows = np.repeat(np.arange(S.shape[0]), self.k)
            Wm = sp.csr_matrix((W.ravel(), (rows, top.ravel())), shape=S.shape)
            E_t = normalize(Wm @ self.prof_title)
            E_p = normalize(Wm @ self.prof_params)
            E_d = normalize(Wm @ self.prof_desc)
            Pm = normalize(Wm @ self.prof_micro, norm="l1").toarray()
            # «точное» распределение подкатегорий — только от почти идентичных запросов
            We = sp.csr_matrix(np.where(S > 0.9, S, 0))
            Pme = normalize(We @ self.prof_micro, norm="l1").toarray()
            # kNN по эмбеддингам: соседи-запросы → профиль эмбеддингов и подкатегорий
            qe = QE[sl]
            Se = qe @ self.QE_train.T
            tope = np.argsort(-Se, axis=1)[:, : self.k]
            We_ = np.take_along_axis(Se, tope, axis=1)
            knn_emb_top = We_[:, 0].copy()
            We_ = np.clip(We_, 0, None) ** 8  # e5-сходства сжаты около 0.8–1 — заостряем
            Wem = sp.csr_matrix((We_.ravel(), (np.repeat(np.arange(Se.shape[0]), self.k), tope.ravel())), shape=Se.shape)
            X_emb = normalize(np.asarray(Wem @ self.prof_emb))
            X_emb_txt = normalize(np.asarray(Wm @ self.prof_emb))
            Pm_emb = normalize(Wem @ self.prof_micro, norm="l1").toarray()

            sc = {
                "s_title": (self.X_title @ Qt[sl].T).toarray(),
                "s_params": (self.X_params @ Qp[sl].T).toarray(),
                "s_desc": (self.X_desc @ Qd[sl].T).toarray(),
                "s_char": (self.X_char @ Qc[sl].T).toarray(),
                "s_exp_title": (self.X_title @ E_t.T).toarray(),
                "s_exp_params": (self.X_params @ E_p.T).toarray(),
                "s_exp_desc": (self.X_desc @ E_d.T).toarray(),
                # число слов запроса в заголовке и слов фильтра в параметрах
                "cover": (self.B_title @ Qb[sl].T).toarray(),
                "filt": (self.B_params @ Fb[sl].T).toarray(),
                "s_emb": self.E_items @ qe.T,
                "s_emb_exp": self.E_items @ X_emb.T,
                "s_emb_exp_txt": self.E_items @ X_emb_txt.T,
            }
            for j in range(S.shape[0]):
                qi = b0 + j
                row = queries.iloc[qi]
                pool, lp = self._pool(row.search_location_id)
                if len(pool) == 0:
                    yield row.query_id, pool, np.zeros((0, len(FEATURES)), np.float32)
                    continue
                cover = sc["cover"][pool, j] / max(q_len[qi], 1)
                if f_len[qi] > 0:
                    filt = sc["filt"][pool, j] / f_len[qi]
                else:
                    filt = np.full(len(pool), -1, np.float32)
                mic = self.item_micro[pool]
                F = np.column_stack([
                    sc["s_title"][pool, j], sc["s_params"][pool, j], sc["s_desc"][pool, j], sc["s_char"][pool, j],
                    sc["s_exp_title"][pool, j], sc["s_exp_params"][pool, j], sc["s_exp_desc"][pool, j],
                    Pm[j, mic], Pme[j, mic], filt, lp,
                    (self.item_loc[pool] == row.search_location_id).astype(np.float32),
                    self.loc_share[pool],
                    np.full(len(pool), W[j, 0] ** (1 / 3), np.float32),
                    np.full(len(pool), float(S[j, top[j, 0]] > 0.999), np.float32),
                    self.rating[pool], self.log_reviews[pool], self.log_price[pool],
                    np.full(len(pool), q_len[qi], np.float32), cover,
                    sc["s_emb"][pool, j], sc["s_emb_exp"][pool, j], sc["s_emb_exp_txt"][pool, j],
                    Pm_emb[j, mic], np.full(len(pool), knn_emb_top[j], np.float32),
                ]).astype(np.float32)
                yield row.query_id, pool, F


def prescore(F: np.ndarray) -> np.ndarray:
    """Простая линейная смесь сигналов для предотбора (веса подобраны на валидации)."""
    d = dict(zip(FEATURES, F.T))
    return (d["s_title"] + d["s_char"] + 2 * d["s_exp_title"] + d["s_exp_params"] + d["p_micro"]
            + 0.5 * d["loc_p"] + 0.3 * d["same_loc"]
            + 3 * d["s_emb"] + 3 * d["s_emb_exp"] + d["p_micro_emb"])


def build_candidates(R: Retriever, queries: pd.DataFrame, n_mix: int = 500, n_each: int = 60) -> pd.DataFrame:
    """Кандидаты для ранкера: top-n_mix по смеси ∪ top-n_each по каждому текстовому скору.

    Объединение с отдельными скорами страхует от случаев, когда смесь «топит»
    объявление, сильное лишь по одному сигналу (например, только по описанию).
    """
    fi = {f: i for i, f in enumerate(FEATURES)}
    each = [fi[f] for f in ["s_title", "s_desc", "s_char", "s_exp_title", "s_exp_desc", "title_cover", "s_emb", "s_emb_exp", "s_emb_exp_txt"]]
    ids = R.items.item_id.values
    out = []
    for qid, pool, F in R.query_features(queries):
        if len(pool) == 0:
            continue
        s = prescore(F)
        sel = [np.argpartition(-s, min(n_mix, len(s) - 1))[:n_mix]]
        for j in each:
            sel.append(np.argpartition(-F[:, j], min(n_each, len(s) - 1))[:n_each])
        sel = np.unique(np.concatenate(sel))
        df = pd.DataFrame(F[sel], columns=FEATURES)
        df["prescore"] = s[sel]
        df["query_id"] = qid
        df["item_id"] = ids[pool[sel]]
        out.append(df)
    return pd.concat(out, ignore_index=True)
