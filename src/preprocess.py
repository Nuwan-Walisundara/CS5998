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
    # Confluence-style @mentions: "@Jane Doe", "@jane.doe (Unlicensed)", "@Unknown User (id)"
    (re.compile(r"@(?:Unknown User \([^)]*\)|[\w.]+(?: [A-Z][a-z]+)?)(?: \(Unlicensed\))?"), "<PERSON>"),
]


@lru_cache(maxsize=1)
def _custom_terms() -> tuple[re.Pattern, dict[str, str]] | None:
    """Organisation-specific names from a private file: `TERM = Replacement`, or `TERM` alone for <ORG>."""
    if not config.ANON_TERMS_FILE.exists():
        return None
    mapping = {}
    for line in config.ANON_TERMS_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        term, _, rep = line.partition("=")
        if term.strip():
            mapping[term.strip().lower()] = rep.strip() or "<ORG>"
    if not mapping:
        return None
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, sorted(mapping, key=len, reverse=True))) + r")\b", re.I)
    return pattern, mapping


def anonymise(text: str) -> str:
    t = text or ""
    # Custom terms first, so listed full names win over the generic @mention pattern
    if (custom := _custom_terms()) is not None:
        pattern, mapping = custom
        t = pattern.sub(lambda m: mapping[m.group(0).lower()], t)
    for pat, rep in _PATTERNS:
        t = pat.sub(rep, t)
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
    # Document names also carry client/system names; "_" would block whole-word matching
    ann["doc"] = ann["doc"].str.replace("_", " ").map(anonymise)
    config.ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    path = config.ANNOTATION_DIR / "to_annotate.csv"
    if path.exists():  # keep labels already entered for unchanged statements
        old = pd.read_csv(path).dropna(subset=["category"]).drop_duplicates("text_clean")
        old = old.set_index("text_clean")[["category", "quality_label"]]
        known = ann["text_clean"].isin(old.index)
        ann.loc[known, ["category", "quality_label"]] = old.loc[ann.loc[known, "text_clean"]].values
    ann.to_csv(path, index=False)
    print(f"{len(df)} units cleaned, {int(df.is_nlp_unit.sum())} NLP units -> {a.out}")


if __name__ == "__main__":
    main()
