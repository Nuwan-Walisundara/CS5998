"""Stage 4 (category) and Stage 5b (quality label) - supervised text classification.

Experiment: TF-IDF + Logistic Regression (baseline) vs. transformer sentence
embeddings + Logistic Regression. Same classifier, same stratified folds,
same metrics - only the representation differs.

  python -m src.classify --labels data/annotations/labelled.csv --target category
  python -m src.classify --labels data/annotations/labelled.csv --target quality_label
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from . import config
from .embeddings import SentenceEmbedder

TEXT_COL = "text_clean"


def build_models() -> dict[str, Pipeline]:
    clf = lambda: LogisticRegression(max_iter=2000, class_weight="balanced", random_state=config.SEED)
    return {
        "tfidf_lr": Pipeline([("rep", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True,
                                                      stop_words="english")), ("clf", clf())]),
        "embed_lr": Pipeline([("rep", SentenceEmbedder()), ("clf", clf())]),
    }


def evaluate(df: pd.DataFrame, target: str, n_splits: int = 5) -> dict:
    df = df.dropna(subset=[TEXT_COL, target])
    df = df[df[target].astype(str).str.strip() != ""]
    counts = df[target].value_counts()
    keep = counts[counts >= 2].index          # stratified CV needs >= 2 per class
    if len(keep) < len(counts):
        print(f"Dropping classes with < 2 examples: {list(counts[counts < 2].index)}")
    df = df[df[target].isin(keep)].reset_index(drop=True)
    X, y = df[TEXT_COL].values, df[target].astype(str).values
    k = int(min(n_splits, df[target].value_counts().min()))
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=config.SEED)
    labels = sorted(set(y))

    results, per_fold = {}, {}
    for name, model in build_models().items():
        preds = np.empty(len(y), dtype=object)
        fold_f1 = []
        for tr, te in cv.split(X, y):
            model.fit(X[tr], y[tr])
            p = model.predict(X[te])
            preds[te] = p
            fold_f1.append(f1_score(y[te], p, average="macro", zero_division=0))
        results[name] = {
            "accuracy": accuracy_score(y, preds),
            "macro_precision": precision_score(y, preds, average="macro", zero_division=0),
            "macro_recall": recall_score(y, preds, average="macro", zero_division=0),
            "macro_f1": f1_score(y, preds, average="macro", zero_division=0),
            "fold_macro_f1_mean": float(np.mean(fold_f1)),
            "fold_macro_f1_std": float(np.std(fold_f1)),
            "report": classification_report(y, preds, zero_division=0, output_dict=True),
            "confusion_matrix": confusion_matrix(y, preds, labels=labels).tolist(),
        }
        per_fold[name] = fold_f1
        df[f"pred_{name}"] = preds
        print(f"\n=== {target} | {name} (k={k}) ===")
        print(classification_report(y, preds, zero_division=0))

    # Paired comparison across folds (Wilcoxon) - indicative only with few folds
    try:
        from scipy.stats import wilcoxon
        stat = wilcoxon(per_fold["embed_lr"], per_fold["tfidf_lr"], zero_method="zsplit")
        results["wilcoxon_p"] = float(stat.pvalue)
    except Exception as e:  # identical scores / scipy missing
        results["wilcoxon_p"] = None
    results["labels"] = labels
    results["n_samples"] = int(len(y))
    results["k_folds"] = k
    results["errors"] = df[[c for c in df.columns if c in ("req_id", TEXT_COL, target) or c.startswith("pred_")]]
    return results


def save_outputs(results: dict, target: str):
    out = config.RESULTS_DIR / target
    out.mkdir(parents=True, exist_ok=True)
    errors = results.pop("errors")
    errors.to_csv(out / "cv_predictions.csv", index=False)  # use for error analysis
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        for name in ("tfidf_lr", "embed_lr"):
            fig, ax = plt.subplots(figsize=(7, 6))
            ConfusionMatrixDisplay(np.array(results[name]["confusion_matrix"]),
                                   display_labels=results["labels"]).plot(ax=ax, xticks_rotation=45, colorbar=False)
            ax.set_title(f"{target} - {name}")
            fig.tight_layout(); fig.savefig(out / f"confusion_{name}.png", dpi=150); plt.close(fig)
    except Exception as e:
        print("Plotting skipped:", e)
    summary = pd.DataFrame({n: {k: results[n][k] for k in ("accuracy", "macro_precision", "macro_recall",
                                                            "macro_f1", "fold_macro_f1_mean", "fold_macro_f1_std")}
                            for n in ("tfidf_lr", "embed_lr")}).T.round(3)
    summary.to_csv(out / "comparison.csv")
    print("\n", summary, f"\nWilcoxon p = {results['wilcoxon_p']}\nSaved -> {out}")


def train_final(df: pd.DataFrame, target: str, model_name: str):
    df = df.dropna(subset=[TEXT_COL, target])
    df = df[df[target].astype(str).str.strip() != ""]
    model = build_models()[model_name].fit(df[TEXT_COL].values, df[target].astype(str).values)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = config.MODEL_DIR / f"{target}.joblib"
    joblib.dump({"model": model, "name": model_name}, path)
    print(f"Final {model_name} model for '{target}' -> {path}")


# --------------------------------------------------------------------------- inference
# Keyword fallback so the pipeline still runs before any model is trained.
_KEYWORD_CATEGORIES = [
    ("Validation/Security", r"\b(validat|verif|signature|authenticat|authori[sz]|token|encrypt|certificate|secur)"),
    ("Error Handling", r"\b(retry|retries|timeout|timed[- ]out|error|fail|exception|revers|rollback)"),
    ("Notification/Callback", r"\b(callback|notif|sms|email|alert|acknowledg)"),
    ("Data/Storage", r"\b(store|save|persist|database|db\b|log|record|table)"),
    ("Integration/Backend", r"\b(post|cbs|backend|backoffice|core banking|send request|invoke|call)"),
    ("API Exposure", r"\b(expose|api|endpoint|resource|receive)"),
]


def keyword_category(text: str) -> str:
    for cat, pat in _KEYWORD_CATEGORIES:
        if re.search(pat, text or "", re.I):
            return cat
    return "Functional/Other"


def predict(units: pd.DataFrame, target: str) -> pd.DataFrame:
    """Adds `<target>`, `<target>_conf`, `<target>_source` columns for NLP units."""
    df = units.copy()
    mask = df["is_nlp_unit"].astype(bool)
    df[target], df[f"{target}_conf"], df[f"{target}_source"] = None, np.nan, None
    if not mask.any():
        return df
    path = config.MODEL_DIR / f"{target}.joblib"
    if path.exists():
        bundle = joblib.load(path)
        model = bundle["model"]
        texts = df.loc[mask, TEXT_COL].values
        df.loc[mask, target] = model.predict(texts)
        df.loc[mask, f"{target}_conf"] = model.predict_proba(texts).max(axis=1).round(3)
        df.loc[mask, f"{target}_source"] = bundle["name"]
    elif target == "category":
        df.loc[mask, target] = df.loc[mask, TEXT_COL].map(keyword_category)
        df.loc[mask, f"{target}_source"] = "keyword-fallback"
    return df


def ml_quality_findings(units: pd.DataFrame) -> list[dict]:
    """Turn predicted quality labels (other than 'Clear') into findings."""
    from .rules import finding
    if "quality_label" not in units:
        return []
    out = []
    flagged = units[units["is_nlp_unit"] & units["quality_label"].notna()
                    & ~units["quality_label"].astype(str).str.lower().isin(["clear", "none", ""])]
    for _, r in flagged.iterrows():
        out.append(finding(r.to_dict(), str(r["quality_label"]),
                           f"ML model predicts '{r['quality_label']}' (confidence {r['quality_label_conf']:.2f})",
                           "Medium", source="ML", score=r["quality_label_conf"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="CSV with text_clean + label columns")
    ap.add_argument("--target", default="category", choices=["category", "quality_label"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--final", default="best", help="model to save: tfidf_lr | embed_lr | best")
    a = ap.parse_args()
    df = pd.read_csv(a.labels)
    res = evaluate(df, a.target, a.folds)
    best = max(("tfidf_lr", "embed_lr"), key=lambda n: res[n]["macro_f1"])
    save_outputs(res, a.target)
    train_final(df, a.target, best if a.final == "best" else a.final)


if __name__ == "__main__":
    main()
