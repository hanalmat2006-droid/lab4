"""
AI Security Monitor — полная версия в ОДНОМ файле.

Запуск:  streamlit run app.py
Зависимости: streamlit, pandas, numpy, scikit-learn, matplotlib

Что внутри:
  1. Модуль 1 (сеть): синтетический сетевой набор, Isolation Forest, Random Forest.
  2. Модуль 2 (NLP): нормализация HTTP-запросов, TF-IDF по символьным n-граммам,
     Logistic Regression, сравнение с Naive Bayes, объяснение прогноза.
  3. Единая схема событий.
  4. Панель Streamlit: Обзор, Сеть, HTTP, Качество, О системе.

Модели обучаются при первом запуске (несколько секунд) и кэшируются.
Все результаты — прогнозы моделей, а не подтверждённые атаки.
"""
from __future__ import annotations

import html
import re
import urllib.parse
import random
from urllib.parse import unquote_plus

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.25
MAX_UPLOAD_MB = 20
MAX_UPLOAD_ROWS = 200_000
MAX_REQUEST_LENGTH = 4096
NLP_REVIEW_THRESHOLD = 0.60
BENIGN = "BENIGN"
TEXT_CLASSES = ["Normal", "SQL Injection", "XSS"]
STATUS = "Прогноз модели (не подтверждено)"

NET_FEATURES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Flow Bytes/s",
    "Flow Packets/s", "Flow IAT Mean", "SYN Flag Count", "ACK Flag Count",
    "Average Packet Size",
]
EVENT_COLUMNS = [
    "event_id", "timestamp", "source_module", "model_name", "prediction", "score",
    "is_alert", "needs_review", "ground_truth", "details", "status",
]


# ============================================================================
# ОБЩИЕ УТИЛИТЫ (безопасность CSV, схема событий, метрики)
# ============================================================================
def sanitize_for_csv_export(df: pd.DataFrame) -> pd.DataFrame:
    """Защита от formula injection: ячейки, начинающиеся с = + - @, получают префикс '."""
    out = df.copy()
    for col in out.select_dtypes(include="object").columns:
        out[col] = out[col].map(
            lambda v: "'" + v if isinstance(v, str) and v[:1] in ("=", "+", "-", "@") else v
        )
    return out


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return sanitize_for_csv_export(df).to_csv(index=False).encode("utf-8")


def read_uploaded_csv(uploaded) -> pd.DataFrame:
    """Безопасное чтение: только .csv, <= 20 МБ, <= 200 000 строк."""
    if not uploaded.name.lower().endswith(".csv"):
        raise ValueError("Разрешены только файлы .csv")
    if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"Файл больше {MAX_UPLOAD_MB} МБ")
    df = pd.read_csv(uploaded, low_memory=False)
    if len(df) > MAX_UPLOAD_ROWS:
        raise ValueError(f"Больше {MAX_UPLOAD_ROWS} строк")
    df.columns = df.columns.str.strip()
    return df


def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Нет обязательных столбцов: {missing}")


def make_events(**cols) -> pd.DataFrame:
    df = pd.DataFrame(cols)
    missing = [c for c in EVENT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"В событии нет полей: {missing}")
    if not df["score"].between(0, 1).all():
        raise ValueError("score должен лежать в диапазоне [0, 1]")
    return df[EVENT_COLUMNS]


def false_positive_rate(y_true_bin, y_pred_bin) -> float:
    y_true_bin = np.asarray(y_true_bin).astype(bool)
    y_pred_bin = np.asarray(y_pred_bin).astype(bool)
    fp = int((~y_true_bin & y_pred_bin).sum())
    tn = int((~y_true_bin & ~y_pred_bin).sum())
    return fp / (fp + tn) if (fp + tn) else 0.0


def evaluate(y_true, y_pred, labels) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "labels": list(labels),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


# ============================================================================
# МОДУЛЬ 1. СЕТЬ
# ============================================================================
def generate_network_data(n: int = 6000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Синтетический набор в схеме, похожей на CIC-IDS2017 (учебные данные)."""
    rng = np.random.default_rng(seed)
    classes = [BENIGN, "PortScan", "DoS Hulk", "DDoS", "SSH-Patator"]
    probs = [0.70, 0.08, 0.08, 0.07, 0.07]
    labels = rng.choice(classes, size=n, p=probs)
    rows = []
    for lab in labels:
        if lab == BENIGN:
            port = rng.choice([80, 443, 53, 22, 8080], p=[0.3, 0.4, 0.15, 0.05, 0.1])
            dur = rng.lognormal(11, 1.8)
            fwd, bwd = rng.poisson(8) + 1, rng.poisson(8) + 1
            lf, lb = rng.lognormal(6, 1.1), rng.lognormal(7, 1.4)
            if rng.random() < 0.04:  # «тяжёлая» норма (бэкап) — источник ложных тревог
                fwd, bwd, lf, lb = fwd * 40, bwd * 40, lf * 30, lb * 30
            syn, ack = int(rng.random() < 0.1), int(rng.random() < 0.9)
        elif lab == "PortScan":
            port = int(rng.integers(1, 65535))
            dur = rng.uniform(1, 100)
            fwd, bwd = 1, int(rng.integers(0, 2))
            lf, lb = rng.uniform(0, 6), rng.uniform(0, 6)
            syn, ack = 1, 0
        elif lab == "DoS Hulk":
            port, dur = 80, rng.lognormal(12, 0.8)
            fwd, bwd = int(rng.integers(5, 16)), int(rng.integers(3, 9))
            lf, lb = rng.lognormal(6.5, 0.5), rng.lognormal(7.5, 0.6)
            syn, ack = 0, 1
        elif lab == "DDoS":
            port, dur = 80, rng.lognormal(10.5, 0.8)
            fwd, bwd = int(rng.integers(3, 11)), int(rng.integers(0, 6))
            lf, lb = rng.lognormal(5.5, 0.6), rng.lognormal(5.0, 0.8)
            syn, ack = 1, 1
        else:  # SSH-Patator
            port, dur = 22, rng.lognormal(11.5, 0.4)
            fwd, bwd = int(rng.integers(15, 26)), int(rng.integers(15, 26))
            lf, lb = rng.lognormal(5.0, 0.3), rng.lognormal(5.2, 0.3)
            syn, ack = 0, 1
        total = lf + lb
        pk = fwd + bwd
        rows.append({
            "Destination Port": port, "Flow Duration": dur,
            "Total Fwd Packets": fwd, "Total Backward Packets": bwd,
            "Total Length of Fwd Packets": lf, "Total Length of Bwd Packets": lb,
            "Flow Bytes/s": total / (dur / 1e6 + 1e-6),
            "Flow Packets/s": pk / (dur / 1e6 + 1e-6),
            "Flow IAT Mean": dur / max(pk, 1),
            "SYN Flag Count": syn, "ACK Flag Count": ack,
            "Average Packet Size": total / max(pk, 1), "Label": lab,
        })
    df = pd.DataFrame(rows)
    start = pd.Timestamp("2026-09-01 09:00:00")
    df["Timestamp"] = (start + pd.to_timedelta(rng.integers(0, 8 * 3600, n), unit="s")).astype(str)
    # шум качества данных: inf и пропуски (чистятся ниже)
    idx = rng.choice(n, size=int(n * 0.01), replace=False)
    df.loc[idx, "Flow Bytes/s"] = np.inf
    idx = rng.choice(n, size=int(n * 0.01), replace=False)
    df.loc[idx, "Flow IAT Mean"] = np.nan
    return df


def clean_network_data(df: pd.DataFrame) -> pd.DataFrame:
    """inf -> NaN, числа, удаление дубликатов. Пропуски заполняются внутри Pipeline."""
    out = df.copy()
    out.columns = out.columns.str.strip()
    for c in NET_FEATURES:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out[NET_FEATURES] = out[NET_FEATURES].replace([np.inf, -np.inf], np.nan)
    return out.drop_duplicates().reset_index(drop=True)


def iso_pipeline(contamination: float) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", IsolationForest(n_estimators=200, contamination=contamination,
                                  random_state=RANDOM_STATE, n_jobs=-1)),
    ])


def rf_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                         random_state=RANDOM_STATE, n_jobs=-1)),
    ])


def train_network(df: pd.DataFrame):
    """Разбиение ДО обучения; IF учится только на BENIGN из train; тест — один раз."""
    X, y = df[NET_FEATURES], df["Label"].astype(str)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, stratify=y,
                                              random_state=RANDOM_STATE)
    ytr_bin, yte_bin = (y_tr != BENIGN).astype(int), (y_te != BENIGN).astype(int)

    # подбор contamination на валидационной части train
    X_fit, X_val, y_fit, y_val = train_test_split(X_tr, ytr_bin, test_size=0.25,
                                                  stratify=ytr_bin, random_state=RANDOM_STATE)
    best_c, best_f1 = 0.05, -1
    for c in (0.01, 0.02, 0.05, 0.10, 0.15):
        p = iso_pipeline(c).fit(X_fit[y_fit == 0])
        f = f1_score(y_val, (p.predict(X_val) == -1).astype(int), zero_division=0)
        if f > best_f1:
            best_c, best_f1 = c, f
    iso = iso_pipeline(best_c).fit(X_tr[ytr_bin == 0])
    iso_pred = (iso.predict(X_te) == -1).astype(int)
    iso_m = evaluate(np.where(yte_bin == 1, "Attack", "Normal"),
                     np.where(iso_pred == 1, "Attack", "Normal"), ["Normal", "Attack"])
    iso_m["false_positive_rate"] = round(false_positive_rate(yte_bin, iso_pred), 4)
    iso_m["contamination"] = best_c
    iso_m["detection_rate_by_label"] = {
        str(lab): round(float(iso_pred[(y_te == lab).to_numpy()].mean()), 4) for lab in sorted(y.unique())
    }

    rf = rf_pipeline().fit(X_tr, y_tr)
    rf_pred = rf.predict(X_te)
    labels = sorted(y.unique(), key=lambda s: (s != BENIGN, s))
    rf_m = evaluate(y_te, rf_pred, labels)
    rf_m["false_positive_rate"] = round(false_positive_rate(yte_bin, (rf_pred != BENIGN).astype(int)), 4)
    rf_m["feature_importance"] = dict(sorted(
        zip(NET_FEATURES, rf.named_steps["model"].feature_importances_.round(4)),
        key=lambda t: -t[1]))
    return iso, rf, {"isolation_forest": iso_m, "random_forest": rf_m,
                     "dataset": {"train_rows": len(X_tr), "test_rows": len(X_te)}}


def analyze_network(df: pd.DataFrame, iso: Pipeline, rf: Pipeline) -> pd.DataFrame:
    X = df[NET_FEATURES]
    out = df.copy()
    out["if_anomaly"] = iso.predict(X) == -1
    out["if_score"] = np.clip(-iso.score_samples(X), 0, 1)
    proba = rf.predict_proba(X)
    out["rf_prediction"] = rf.classes_[proba.argmax(axis=1)]
    out["rf_confidence"] = proba.max(axis=1)
    return out


def network_events(a: pd.DataFrame) -> pd.DataFrame:
    n = len(a)
    rf_attack = a["rf_prediction"] != BENIGN
    details = ("port " + a["Destination Port"].fillna(-1).astype(int).astype(str)
               + " | fwd_pkts " + a["Total Fwd Packets"].fillna(0).astype(int).astype(str)
               + " | RF: " + a["rf_prediction"].astype(str)
               + " (" + a["rf_confidence"].round(2).astype(str) + ")")
    ts = (pd.to_datetime(a["Timestamp"], errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S")
          .fillna("").to_numpy() if "Timestamp" in a.columns else [""] * n)
    truth = a["Label"].astype(str).to_numpy() if "Label" in a.columns else [""] * n
    return make_events(
        event_id=[f"net-{i:06d}" for i in range(1, n + 1)], timestamp=ts,
        source_module=["network"] * n, model_name=["IsolationForest"] * n,
        prediction=np.where(a["if_anomaly"], "Anomaly", "Normal"),
        score=a["if_score"].to_numpy(), is_alert=a["if_anomaly"].to_numpy(),
        needs_review=(a["if_anomaly"] != rf_attack).to_numpy(),  # IF и RF расходятся
        ground_truth=truth, details=details.to_numpy(), status=[STATUS] * n)


# ============================================================================
# МОДУЛЬ 2. NLP (HTTP-запросы)
# ============================================================================
def normalize_request(text) -> str:
    """URL-декодирование (до 3 раз) -> HTML-unescape -> нижний регистр -> числа в 0."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text)[:MAX_REQUEST_LENGTH]
    for _ in range(3):
        d = unquote_plus(s)
        if d == s:
            break
        s = d
    s = html.unescape(s).replace("\x00", " ").lower()
    s = re.sub(r"\d+", "0", s)
    return re.sub(r"\s+", " ", s).strip()


def generate_text_data(per_class: int = 500, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Учебный синтетический набор (Normal / SQL Injection / XSS).
    Для реальной работы замените открытыми датасетами (см. ТЗ, раздел 8.2)."""
    rnd = random.Random(seed)
    ch = rnd.choice
    words = ["python", "book", "coffee", "weather", "laptop", "music", "travel", "recipe",
             "football", "news", "holiday", "garden", "movie", "phone", "shoes", "pizza",
             "bike", "camera", "guitar", "hotel", "flight", "tea", "yoga", "chess"]
    names = ["john", "maria", "o'reilly", "d'angelo", "anna", "alex", "sara", "tim", "kate", "omar"]
    tables = ["users", "accounts", "admin", "customers", "orders", "members", "employees"]
    cols = ["username", "password", "email", "token", "card", "login", "secret"]
    normal_t = [
        lambda: f"/search?q={ch(words)}+{ch(words)}+{ch(words)}",
        lambda: f"name={ch(names)}&city={ch(words)}&page={rnd.randint(1, 50)}",
        lambda: f"/product/{ch(words)}-{rnd.randint(1, 9999)}?color={ch(['red', 'blue', 'green', 'black'])}",
        lambda: f"comment=I like {ch(words)} and select the best {ch(words)} for {ch(names)}",
        lambda: f"/api/{ch(words)}/{rnd.randint(1, 999)}/orders?limit={rnd.randint(5, 50)}",
        lambda: f"email={ch(names)}{rnd.randint(1, 99)}@{ch(words)}.com",
        lambda: f"q=how to delete a {ch(words)} from union station {ch(words)}",
        lambda: f"title={ch(words)} & {ch(words)} <3 review by {ch(names)}",
        lambda: f"/category/{ch(words)}/{ch(words)}?sort={ch(['price', 'name', 'date'])}",
    ]

    def sqli() -> str:
        q = ch(["'", "'", '"', "')", ""])
        a = rnd.randint(1, 9)
        kind = rnd.randint(0, 5)
        com = ch(["-- ", "--", "#", "/*"])
        if kind == 0:
            return f"{ch(['1', 'admin', ch(names), ch(words)])}{q} {ch(['or', 'OR', 'Or', 'and'])} {q}{a}{q}={q}{a}{com}"
        if kind == 1:
            return f"{ch(names)}{q} {com}"
        if kind == 2:
            return (f"{rnd.randint(1, 99)}{q} {ch(['union', 'UNION', 'union all'])} select "
                    f"{ch(cols)}, {ch(cols)} from {ch(tables)} {com}")
        if kind == 3:
            return f"{rnd.randint(1, 99)}; {ch(['drop table', 'delete from', 'truncate table'])} {ch(tables)}; {com}"
        if kind == 4:
            return f"{ch(words)}{q} and sleep({rnd.randint(1, 9)}) {com}"
        return f"{q} or {ch(['1=1', 'true', 'a=a', 'null is null'])} {ch(['limit 1', 'order by 1', ''])} {com}"

    def xss() -> str:
        tag = ch(["script", "img", "svg", "body", "iframe", "input", "video", "details", "a"])
        ev = ch(["onerror", "onload", "onclick", "onmouseover", "onfocus", "ontoggle"])
        fn = ch(["alert", "prompt", "confirm"])
        arg = ch(["1", "document.cookie", "'xss'", "document.domain", str(rnd.randint(1, 99))])
        kind = rnd.randint(0, 4)
        if kind == 0:
            return f"<script>{fn}({arg})</script>"
        if kind == 1:
            return f"<{ch(['img', 'svg', 'video', 'input'])} {ch(['src=x', 'autofocus', ''])} {ev}={fn}({arg})>"
        if kind == 2:
            pre = ch(['"', "'", ''])
            return pre + f"><{tag} {ev}={fn}({arg})>"
        if kind == 3:
            return f"javascript:{fn}({arg})"
        return '<iframe src="javascript:' + f"{fn}({arg})" + '"></iframe>'

    def maybe_encode(s: str) -> str:  # часть атак — в URL-кодировке
        r = rnd.random()
        if r < 0.25:
            return urllib.parse.quote(s)
        if r < 0.32:
            return urllib.parse.quote(urllib.parse.quote(s))
        return s

    def collect(gen, label, enc):
        seen, out, tries = set(), [], 0
        while len(out) < per_class and tries < per_class * 50:
            tries += 1
            r = gen()
            key = normalize_request(r)
            if key and key not in seen:
                seen.add(key)
                out.append((maybe_encode(r) if enc else r, label))
        return out

    rows = (collect(lambda: ch(normal_t)(), "Normal", False)
            + collect(sqli, "SQL Injection", True)
            + collect(xss, "XSS", True))
    return pd.DataFrame(rows, columns=["request", "label"]).sample(frac=1, random_state=seed).reset_index(drop=True)


# «трудные» нормальные запросы для проверки ложных срабатываний
HARD_BENIGN = [
    "q=select the best laptop", "comment=union of two sets", "name=o'reilly",
    "q=how to delete a file", "title=Tom & Jerry <3", "q=drop shipping business",
    "text=if x < 5 and y > 3", "q=insert coin to play", "note=update -- later",
    "search=script writing course", "q=1=1 equation", "bio=I'm a developer; I love code",
]


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(preprocessor=normalize_request, lowercase=False, analyzer="char_wb",
                           ngram_range=(2, 5), min_df=2, max_features=50_000, sublinear_tf=True)


def train_nlp(df: pd.DataFrame):
    X, y = df["request"].astype(str), df["label"].astype(str)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, stratify=y,
                                              random_state=RANDOM_STATE)
    models = {
        "logistic_regression": Pipeline([("tfidf", build_vectorizer()),
                                         ("model", LogisticRegression(max_iter=3000, C=10,
                                                                      class_weight="balanced"))]),
        "naive_bayes": Pipeline([("tfidf", build_vectorizer()),
                                 ("model", MultinomialNB(alpha=0.1))]),
    }
    metrics = {}
    for name, pipe in models.items():
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        m = evaluate(y_te, pred, TEXT_CLASSES)
        m["false_positive_rate"] = round(false_positive_rate(y_te != "Normal", pred != "Normal"), 4)
        hard_pred = pipe.predict(pd.Series(HARD_BENIGN))
        m["fpr_on_hard_benign"] = round(float((hard_pred != "Normal").mean()), 4)
        metrics[name] = m
    metrics["dataset"] = {"train_rows": len(X_tr), "test_rows": len(X_te)}
    return models, metrics


def classify_requests(model: Pipeline, requests) -> pd.DataFrame:
    req = pd.Series(list(requests), dtype=str)
    proba = model.predict_proba(req)
    classes = list(model.classes_)
    out = pd.DataFrame({"request": req})
    out["prediction"] = np.array(classes)[proba.argmax(axis=1)]
    out["confidence"] = proba.max(axis=1).round(4)
    for i, c in enumerate(classes):
        out[f"prob_{c}"] = proba[:, i].round(4)
    out["needs_review"] = out["confidence"] < NLP_REVIEW_THRESHOLD
    return out


def explain_request(model: Pipeline, request: str, top_k: int = 8) -> pd.DataFrame:
    """n-граммы с наибольшим вкладом = TF-IDF вес × коэффициент LogReg для класса."""
    clf, vec = model.named_steps["model"], model.named_steps["tfidf"]
    x = vec.transform([request])
    cls = int(np.argmax(clf.predict_proba(x)[0]))
    idx = x.nonzero()[1]
    contrib = x.toarray()[0, idx] * clf.coef_[cls][idx]
    df = pd.DataFrame({"ngram": vec.get_feature_names_out()[idx], "contribution": contrib.round(4)})
    return df.sort_values("contribution", ascending=False).head(top_k).reset_index(drop=True)


def http_events(source: pd.DataFrame, cl: pd.DataFrame) -> pd.DataFrame:
    n = len(cl)
    ts = (pd.to_datetime(source["timestamp"], errors="coerce", format="mixed")
          .dt.strftime("%Y-%m-%dT%H:%M:%S").fillna("").to_numpy()
          if "timestamp" in source.columns else [""] * n)
    truth = source["label"].astype(str).to_numpy() if "label" in source.columns else [""] * n
    return make_events(
        event_id=[f"http-{i:06d}" for i in range(1, n + 1)], timestamp=ts,
        source_module=["http"] * n, model_name=["LogisticRegression"] * n,
        prediction=cl["prediction"].to_numpy(), score=cl["confidence"].to_numpy(),
        is_alert=(cl["prediction"] != "Normal").to_numpy(),
        needs_review=cl["needs_review"].to_numpy(), ground_truth=truth,
        details=cl["request"].str.slice(0, 160).to_numpy(), status=[STATUS] * n)


# ============================================================================
# ОБУЧЕНИЕ ПРИ ПЕРВОМ ЗАПУСКЕ (кэшируется)
# ============================================================================
@st.cache_resource(show_spinner="Обучение моделей (первый запуск, несколько секунд)…")
def get_models():
    net_raw = generate_network_data()
    net_df = clean_network_data(net_raw)
    iso, rf, net_metrics = train_network(net_df)
    text_df = generate_text_data()
    nlp_models, nlp_metrics = train_nlp(text_df)
    metrics = {**net_metrics, **{f"nlp_{k}": v for k, v in nlp_metrics.items()}}
    demo_net = network_events(analyze_network(net_df.sample(1500, random_state=RANDOM_STATE)
                                              .reset_index(drop=True), iso, rf))
    demo_text = text_df.sample(600, random_state=RANDOM_STATE).reset_index(drop=True)
    demo_http = http_events(demo_text, classify_requests(nlp_models["logistic_regression"],
                                                         demo_text["request"]))
    quality = {
        "rows": len(net_raw),
        "duplicates": int(net_raw.duplicated().sum()),
        "inf_values": int(np.isinf(net_raw[NET_FEATURES].apply(pd.to_numeric, errors="coerce")).sum().sum()),
        "missing_values": int(net_raw[NET_FEATURES].isna().sum().sum()),
    }
    return iso, rf, nlp_models, metrics, demo_net, demo_http, quality


def plot_cm(m: dict, title: str):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ConfusionMatrixDisplay(np.array(m["confusion_matrix"]), display_labels=m["labels"]).plot(
        ax=ax, colorbar=False, cmap="Blues", xticks_rotation=30)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


# ============================================================================
# ПАНЕЛЬ STREAMLIT
# ============================================================================
st.set_page_config(page_title="AI Security Monitor", page_icon="🛡️", layout="wide")
st.title("🛡️ AI Security Monitor")
st.caption("Учебный прототип: мониторинг сетевой активности и обнаружение SQLi/XSS. "
           "Все результаты — **прогнозы моделей**, а не подтверждённые атаки.")

iso_model, rf_model, nlp_models, metrics, demo_net, demo_http, quality = get_models()
logreg = nlp_models["logistic_regression"]

if "net_events" not in st.session_state:
    st.session_state.net_events = demo_net
if "http_events" not in st.session_state:
    st.session_state.http_events = demo_http

tab_over, tab_net, tab_http, tab_quality, tab_about = st.tabs(
    ["Обзор", "Сетевой трафик", "HTTP-запросы", "Качество моделей", "О системе"])

# ---------------------------------------------------------------- Обзор ----
with tab_over:
    events = pd.concat([st.session_state.net_events, st.session_state.http_events], ignore_index=True)
    c1, c2, c3, c4 = st.columns(4)
    mod = c1.multiselect("Модуль", ["network", "http"], default=["network", "http"])
    cats = sorted(events["prediction"].unique())
    cat = c2.multiselect("Категория", cats, default=cats)
    result = c3.selectbox("Результат", ["Все", "Только тревоги", "Только норма", "Требуют проверки"])
    has_time = events["timestamp"].astype(str).str.len().gt(0).any()
    if has_time:
        ts = pd.to_datetime(events["timestamp"], errors="coerce")
        lo, hi = ts.min().date(), ts.max().date()
        period = c4.date_input("Период", (lo, hi))
    else:
        c4.caption("Во времени нет данных — фильтр отключён")

    f = events[events["source_module"].isin(mod) & events["prediction"].isin(cat)]
    if result == "Только тревоги":
        f = f[f["is_alert"]]
    elif result == "Только норма":
        f = f[~f["is_alert"]]
    elif result == "Требуют проверки":
        f = f[f["needs_review"]]
    if has_time and isinstance(period, tuple) and len(period) == 2:
        t = pd.to_datetime(f["timestamp"], errors="coerce")
        f = f[t.isna() | ((t.dt.date >= period[0]) & (t.dt.date <= period[1]))]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Всего событий", len(f))
    k2.metric("Тревог", int(f["is_alert"].sum()))
    k3.metric("Доля тревог", f"{(f['is_alert'].mean() if len(f) else 0):.1%}")
    k4.metric("Требуют проверки", int(f["needs_review"].sum()))
    k5.metric("Средний score", f"{(f['score'].mean() if len(f) else 0):.2f}")

    g1, g2 = st.columns(2)
    g1.subheader("События по категориям")
    g1.bar_chart(f["prediction"].value_counts())
    g2.subheader("Тревоги по модулям")
    g2.bar_chart(f.groupby("source_module")["is_alert"].sum())

    st.subheader("Журнал событий")
    st.dataframe(f, use_container_width=True, height=320)
    st.download_button("⬇ Скачать журнал (CSV)", to_csv_bytes(f), "events.csv", "text/csv")
    st.caption(f"Статус всех событий: «{STATUS}». Показаны демонстрационные данные, пока вы не загрузите свои.")

# ---------------------------------------------------------------- Сеть -----
with tab_net:
    st.subheader("Модуль 1. Сетевой трафик")
    st.write("Isolation Forest ищет статистические аномалии, Random Forest определяет тип трафика. "
             "**Аномалия ≠ атака:** если модели расходятся, событие помечается «требует проверки».")
    st.caption("Обязательные столбцы: " + ", ".join(NET_FEATURES))
    up = st.file_uploader("Загрузите CSV с сетевыми потоками", type=["csv"], key="net_up")
    if up is not None:
        try:
            raw = read_uploaded_csv(up)
            require_columns(raw, NET_FEATURES)
            clean = clean_network_data(raw)
            ev = network_events(analyze_network(clean, iso_model, rf_model))
            st.session_state.net_events = ev
            st.success(f"Проанализировано потоков: {len(ev)}")
        except Exception as e:  # понятное сообщение пользователю
            st.error(f"Файл отклонён: {e}")
    ev = st.session_state.net_events
    a, b = st.columns(2)
    a.metric("Аномалий", int(ev["is_alert"].sum()))
    b.metric("Требуют проверки (IF ≠ RF)", int(ev["needs_review"].sum()))
    only = st.checkbox("Только аномалии")
    st.dataframe(ev[ev["is_alert"]] if only else ev, use_container_width=True, height=360)
    st.info("Качество данных (синтетический набор до очистки): "
            f"строк {quality['rows']}, дубликатов {quality['duplicates']}, "
            f"inf {quality['inf_values']}, пропусков {quality['missing_values']}.")

# ---------------------------------------------------------------- HTTP -----
with tab_http:
    st.subheader("Модуль 2. HTTP-запросы (NLP)")
    st.write("Запрос декодируется и нормализуется, затем TF-IDF по символьным n-граммам + Logistic Regression. "
             "Текст **никогда не исполняется** и не отображается как HTML.")
    req = st.text_area("Проверить одиночный запрос", "/search?q=o'reilly python book", height=80)
    if st.button("Проверить"):
        res = classify_requests(logreg, [req]).iloc[0]
        st.write(f"**Класс:** {res['prediction']}  |  **Уверенность:** {res['confidence']:.2f}")
        if res["needs_review"]:
            st.warning("Низкая уверенность — требуется ручная проверка.")
        st.write({c: float(res[f"prob_{c}"]) for c in TEXT_CLASSES})
        st.caption("Уверенность — вероятность выбранного класса, а не вероятность реальной атаки.")
        st.write("Вклад n-грамм в прогноз:")
        st.dataframe(explain_request(logreg, req), use_container_width=True)
        st.code(normalize_request(req), language="text")

    st.divider()
    st.write("Пакетная проверка: CSV со столбцом `request` (необязательно `label`, `timestamp`).")
    up2 = st.file_uploader("Загрузите CSV с запросами", type=["csv"], key="http_up")
    if up2 is not None:
        try:
            raw = read_uploaded_csv(up2)
            require_columns(raw, ["request"])
            raw["request"] = raw["request"].fillna("").astype(str)
            if "label" in raw.columns:
                alias = {"normal": "Normal", "benign": "Normal", "0": "Normal", "sqli": "SQL Injection",
                         "sql injection": "SQL Injection", "xss": "XSS"}
                raw["label"] = raw["label"].map(lambda v: alias.get(str(v).strip().lower(), str(v)))
            cl = classify_requests(logreg, raw["request"])
            st.session_state.http_events = http_events(raw, cl)
            st.success(f"Проанализировано запросов: {len(cl)}")
            st.dataframe(cl, use_container_width=True, height=300)
            st.download_button("⬇ Скачать результат (CSV)", to_csv_bytes(cl), "http_result.csv", "text/csv")
        except Exception as e:
            st.error(f"Файл отклонён: {e}")

# -------------------------------------------------------------- Качество ---
with tab_quality:
    st.subheader("Метрики качества (тестовая выборка, оценка один раз)")
    rows = []
    for key, name in [("isolation_forest", "Isolation Forest"), ("random_forest", "Random Forest"),
                      ("nlp_logistic_regression", "Logistic Regression"), ("nlp_naive_bayes", "Naive Bayes")]:
        m = metrics[key]
        rows.append({"Модель": name, "Accuracy": m["accuracy"], "Precision (macro)": m["precision_macro"],
                     "Recall (macro)": m["recall_macro"], "F1 (macro)": m["f1_macro"],
                     "FPR": m.get("false_positive_rate")})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.caption("FPR на «трудных» нормальных запросах: LogReg = "
               f"{metrics['nlp_logistic_regression']['fpr_on_hard_benign']:.0%}, "
               f"Naive Bayes = {metrics['nlp_naive_bayes']['fpr_on_hard_benign']:.0%}.")
    st.write("Доля записей, помеченных Isolation Forest, по фактической метке:")
    st.bar_chart(pd.Series(metrics["isolation_forest"]["detection_rate_by_label"]))
    st.write("Важность признаков Random Forest:")
    st.bar_chart(pd.Series(metrics["random_forest"]["feature_importance"]))
    cm1, cm2, cm3 = st.columns(3)
    cm1.pyplot(plot_cm(metrics["isolation_forest"], "Isolation Forest"))
    cm2.pyplot(plot_cm(metrics["random_forest"], "Random Forest"))
    cm3.pyplot(plot_cm(metrics["nlp_logistic_regression"], "Logistic Regression"))
    st.warning("Данные синтетические и проще реальных: на CIC-IDS2017 и открытых наборах SQLi/XSS "
               "метрики будут ниже.")

# ---------------------------------------------------------------- О системе -
with tab_about:
    st.subheader("О системе")
    st.markdown("""
**Модуль 1 (сеть).** Isolation Forest обучается только на нормальном трафике; Random Forest классифицирует
известные типы атак. Если модели расходятся — событие уходит на ручную проверку.

**Модуль 2 (NLP).** Нормализация (URL-декодирование до 3 раз, HTML-сущности, нижний регистр, числа → 0),
TF-IDF по символьным n-граммам 2–5, Logistic Regression; Naive Bayes — для сравнения.

**Единая схема событий.** `event_id, timestamp, source_module, model_name, prediction, score, is_alert,
needs_review, ground_truth, details, status`.

**Защита от утечки данных.** Разбиение train/test до обучения, импутация/масштабирование/TF-IDF обучаются
только на train, тест используется один раз.

**Безопасность.** Только `.csv` до 20 МБ и 200 000 строк, HTML не рендерится, выгрузка защищена от formula
injection (`= + - @` → префикс `'`).

**Ограничения.** Прогнозы вероятностны; модели подвержены дрейфу данных; нет потоковой обработки;
сетевой и HTTP-модули не связаны общим ключом (IP/время).
""")
