"""Stage 3 - Preprocessing: anonymisation, normalisation and derived features."""
from __future__ import annotations

import argparse
import re
from functools import lru_cache

import pandas as pd

from . import config

_PATTERNS = [
    (re.compile(r"https?://\S+", re.I), "<URL>"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b"), "<IP>"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "<EMAIL>"),
    (re.compile(r"\b\+?\d{9,15}\b"), "<NUMBER>"),
    (re.compile(r"\b(?:password|pwd|secret|token|api[_-]?key)\s*[:=]\s*\S+", re.I), "<CREDENTIAL>"),
]


@lru_cache(maxsize=1)
def _custom_terms() -> re.Pattern | None:
    """Organisation-specific names (clients, systems, people) from a private file."""
    if not config.ANON_TERMS_FILE.exists():
        return None
    terms = [t.strip() for t in config.ANON_TERMS_FILE.read_text().splitlines()
             if t.strip() and not t.startswith("#")]
    if not terms:
        return None
    return re.compile(r"\b(" + "|".join(map(re.escape, sorted(terms, key=len, reverse=True))) + r")\b", re.I)


def anonymise(text: str) -> str:
    t = text or ""
    for pat, rep in _PATTERNS:
        t = pat.sub(rep, t)
    if (custom := _custom_terms()) is not None:
        t = custom.sub("<ORG>", t)
    return t


def normalise(text: str) -> str:
    t = re.sub(r"\s+", " ", text or "").strip()
    t = re.sub(r"^\d{1,2}[.)]\s+", "", t)  # leftover numbering
    return t


def is_empty_value(text: str) -> bool:
    t = (text or "").strip().lower()
    return t == "" or t in config.TEMPLATE_PLACEHOLDERS


def looks_like_payload(text: str) -> bool:
    t = (text or "").strip()
    return t.startswith(("{", "[", "<?xml", "HTTP/")) or t.count('":') >= 3


def preprocess(units: pd.DataFrame) -> pd.DataFrame:
    df = units.copy()
    df["text"] = df["text"].fillna("").astype(str)
    df["text_clean"] = df["text"].map(anonymise).map(normalise)
    df["is_empty"] = df["text_clean"].map(is_empty_value)
    df["is_payload"] = df["text_clean"].map(looks_like_payload)
    df["n_words"] = df["text_clean"].str.split().str.len().fillna(0).astype(int)
    df["n_chars"] = df["text_clean"].str.len()
    # Units the NLP models analyse: non-empty natural-language requirement statements
    df["is_nlp_unit"] = (df["unit_type"] == "requirement") & ~df["is_empty"] & ~df["is_payload"]
    return df


def main():
    ap = argparse.ArgumentParser(description="Anonymise and clean extracted units")
    ap.add_argument("--input", default=str(config.PROCESSED_DIR / "requirements_raw.csv"))
    ap.add_argument("--out", default=str(config.PROCESSED_DIR / "requirements_clean.csv"))
    a = ap.parse_args()
    df = preprocess(pd.read_csv(a.input))
    df.to_csv(a.out, index=False)
    # Annotation sheet: label `category` and `quality_label` by hand (see docs/annotation_guidelines.md)
    ann = df.loc[df.is_nlp_unit, ["req_id", "doc", "field", "text_clean"]].assign(category="", quality_label="")
    config.ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    ann.to_csv(config.ANNOTATION_DIR / "to_annotate.csv", index=False)
    print(f"{len(df)} units cleaned, {int(df.is_nlp_unit.sum())} NLP units -> {a.out}")


if __name__ == "__main__":
    main()
