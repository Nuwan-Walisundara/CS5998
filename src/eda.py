"""Exploratory Data Analysis (Milestone 2).   python -m src.eda --input data/raw"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config
from .extract import extract_file, to_frames
from .preprocess import preprocess


def run(input_dir: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in input_dir.glob("*") if p.suffix.lower() in {".md", ".docx"})
    units, mappings = to_frames([extract_file(p) for p in files])
    units = preprocess(units)
    nlp = units[units.is_nlp_unit]

    summary = {
        "documents": len(files),
        "units_total": len(units),
        "requirement_statements": len(nlp),
        "template_fields": int((units.unit_type == "field").sum()),
        "empty_fields_pct": round(100 * units.loc[units.unit_type == "field", "is_empty"].mean(), 1),
        "mapping_rows": len(mappings),
        "mapping_missing_mandatory_flag_pct": round(100 * (mappings["mandatory"].fillna("").str.strip() == "").mean(), 1) if len(mappings) else None,
        "statement_words_mean": round(nlp.n_words.mean(), 1),
        "statement_words_median": float(nlp.n_words.median()),
        "vocabulary_size": len(set(" ".join(nlp.text_clean.str.lower()).split())),
    }
    pd.Series(summary).to_csv(out / "summary.csv", header=["value"])
    print(pd.Series(summary).to_string())

    per_doc = units.groupby("doc").agg(units=("req_id", "size"), statements=("is_nlp_unit", "sum"),
                                       empty_fields=("is_empty", "sum"))
    per_doc.to_csv(out / "per_document.csv")

    empt = (units[units.unit_type == "field"].groupby("field_key")["is_empty"].mean()
            .sort_values(ascending=False).head(25) * 100)
    empt.to_csv(out / "empty_rate_by_field.csv")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    nlp.n_words.plot.hist(bins=30, ax=ax[0], title="Requirement length (words)")
    nlp.field.value_counts().plot.barh(ax=ax[1], title="Statements per source field")
    empt.head(15).sort_values().plot.barh(ax=ax[2], title="% empty by template field")
    fig.tight_layout(); fig.savefig(out / "eda_overview.png", dpi=150)

    labelled = config.ANNOTATION_DIR / "labelled.csv"
    if labelled.exists():
        lab = pd.read_csv(labelled)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        for a, col in zip(ax, ["category", "quality_label"]):
            if col in lab:
                lab[col].value_counts().plot.bar(ax=a, title=f"Class balance: {col}")
        fig.tight_layout(); fig.savefig(out / "class_balance.png", dpi=150)
    print("EDA outputs ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(config.RAW_DIR))
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "eda"))
    a = ap.parse_args()
    run(Path(a.input), Path(a.out))
