"""Stage 6 - Semantic similarity (duplicate / overlap detection).

  # 1. sample pairs across the score range for manual labelling
  python -m src.similarity sample --input data/processed/requirements_clean.csv --n 200
  # 2. label column `is_duplicate` (1/0) in data/annotations/pairs_to_label.csv -> save as pairs_labelled.csv
  # 3. sweep thresholds for both representations and choose one
  python -m src.similarity evaluate --pairs data/annotations/pairs_labelled.csv
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support

from . import config


def similarity_matrix(texts: list[str], method: str = "embedding") -> np.ndarray:
    if method == "tfidf":
        X = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english").fit_transform(texts)
        return (X @ X.T).toarray()  # rows are L2-normalised -> cosine
    from .embeddings import encode
    E = encode(texts)
    return E @ E.T


def find_similar(units: pd.DataFrame, threshold: float = config.SIMILARITY_THRESHOLD,
                 method: str = "embedding", scope: str = "all") -> pd.DataFrame:
    """Pairs of NLP units with cosine similarity >= threshold.

    scope: 'within' (same document only) or 'all' (also across documents).
    """
    df = units[units["is_nlp_unit"]].reset_index(drop=True)
    cols = ["doc_a", "req_a", "text_a", "doc_b", "req_b", "text_b", "score", "scope"]
    if len(df) < 2:
        return pd.DataFrame(columns=cols)
    S = similarity_matrix(df["text_clean"].tolist(), method)
    i, j = np.triu_indices(len(df), k=1)
    keep = S[i, j] >= threshold
    rows = []
    for a, b in zip(i[keep], j[keep]):
        same = df.doc[a] == df.doc[b]
        if scope == "within" and not same:
            continue
        rows.append([df.doc[a], df.req_id[a], df.text_clean[a], df.doc[b], df.req_id[b], df.text_clean[b],
                     round(float(S[a, b]), 3), "within-document" if same else "cross-document"])
    return pd.DataFrame(rows, columns=cols).sort_values("score", ascending=False)


def similarity_findings(pairs: pd.DataFrame) -> list[dict]:
    from .rules import finding
    out = []
    for _, p in pairs.iterrows():
        out.append(finding({"doc": p.doc_a, "req_id": p.req_a, "text_clean": p.text_a},
                           "Possible duplicate",
                           f"{p.scope}: similar to {p.req_b} ({p.doc_b}) - \"{p.text_b}\"",
                           "Medium" if p.scope == "within-document" else "Low",
                           source="Similarity", related_req_id=p.req_b, score=p.score))
    return out


# --------------------------------------------------------------------------- evaluation
def sample_pairs(units: pd.DataFrame, n: int = 200, seed: int = config.SEED) -> pd.DataFrame:
    """Stratified sample over embedding-score bins so positives are represented."""
    pairs = find_similar(units, threshold=-1.0)
    bins = pd.cut(pairs["score"], [-1, 0.4, 0.6, 0.7, 0.8, 0.9, 1.01])
    per_bin = max(1, n // bins.nunique())
    sample = (pairs.groupby(bins, observed=True, group_keys=False)
                   .apply(lambda g: g.sample(min(len(g), per_bin), random_state=seed)))
    return sample.drop(columns=["score"]).assign(is_duplicate="").sample(frac=1, random_state=seed)


def evaluate_thresholds(pairs: pd.DataFrame) -> pd.DataFrame:
    pairs = pairs.dropna(subset=["is_duplicate"])
    y = pairs["is_duplicate"].astype(int).values
    rows = []
    for method in ("tfidf", "embedding"):
        texts = pd.concat([pairs.text_a, pairs.text_b]).unique().tolist()
        idx = {t: k for k, t in enumerate(texts)}
        S = similarity_matrix(texts, method)
        scores = np.array([S[idx[a], idx[b]] for a, b in zip(pairs.text_a, pairs.text_b)])
        for th in np.round(np.arange(0.30, 0.96, 0.05), 2):
            p, r, f, _ = precision_recall_fscore_support(y, (scores >= th).astype(int),
                                                         average="binary", zero_division=0)
            rows.append(dict(method=method, threshold=th, precision=p, recall=r, f1=f))
    return pd.DataFrame(rows).round(3)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample"); s.add_argument("--input", default=str(config.PROCESSED_DIR / "requirements_analysed.csv")); s.add_argument("--n", type=int, default=200)
    e = sub.add_parser("evaluate"); e.add_argument("--pairs", default=str(config.ANNOTATION_DIR / "pairs_labelled.csv"))
    a = ap.parse_args()
    if a.cmd == "sample":
        out = config.ANNOTATION_DIR / "pairs_to_label.csv"
        config.ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
        sample_pairs(pd.read_csv(a.input), a.n).to_csv(out, index=False)
        print("Label `is_duplicate` (1/0) in", out)
    else:
        res = evaluate_thresholds(pd.read_csv(a.pairs))
        out = config.RESULTS_DIR / "similarity"; out.mkdir(parents=True, exist_ok=True)
        res.to_csv(out / "threshold_sweep.csv", index=False)
        best = res.loc[res.groupby("method")["f1"].idxmax()]
        (out / "best_thresholds.json").write_text(best.to_json(orient="records", indent=2))
        print(res.to_string(index=False)); print("\nBest F1 per method:\n", best.to_string(index=False))
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 4))
            for m, g in res.groupby("method"):
                ax.plot(g.threshold, g.f1, marker="o", label=f"{m} F1")
                ax.plot(g.threshold, g.precision, ls="--", label=f"{m} precision")
            ax.set_xlabel("cosine threshold"); ax.legend(); fig.tight_layout()
            fig.savefig(out / "threshold_sweep.png", dpi=150)
        except Exception as ex:
            print("Plotting skipped:", ex)


if __name__ == "__main__":
    main()
