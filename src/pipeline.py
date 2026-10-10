"""End-to-end pipeline.

Document -> Extraction -> Preprocessing -> Classification -> Quality/Completeness
         -> Similarity -> Validation Results -> Clarification Questions

  python -m src.pipeline --input data/synthetic            # template questions
  python -m src.pipeline --input data/raw --llm            # + Ollama questions
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import config
from .classify import ml_quality_findings, predict
from .extract import ExtractedDocument, extract_file, parse_markdown, to_frames
from .preprocess import preprocess
from .questions import generate_questions
from .rules import FINDING_COLS, run_rules
from .similarity import find_similar, similarity_findings

SEVERITY_ORDER = {"High": 0, "Medium": 1, "Low": 2}


@dataclass
class AnalysisResult:
    units: pd.DataFrame
    mappings: pd.DataFrame
    findings: pd.DataFrame
    similar_pairs: pd.DataFrame
    questions: pd.DataFrame = field(default_factory=pd.DataFrame)
    question_info: dict = field(default_factory=dict)


def analyse(docs: list[ExtractedDocument], use_similarity: bool = True, sim_method: str = "embedding",
            sim_threshold: float = config.SIMILARITY_THRESHOLD, sim_scope: str = "all",
            questions: bool = True, use_llm: bool = False) -> AnalysisResult:
    # 2. Extraction (already parsed) -> frames
    units, mappings = to_frames(docs)
    # 3. Preprocessing
    units = preprocess(units)
    # 4. Classification - what kind of requirement is it?
    units = predict(units, "category")
    # 5. Quality / completeness - is something wrong or missing?
    units = predict(units, "quality_label")          # ML (only if a model has been trained)
    payload = {d.doc: d.payload_keys for d in docs}
    rule_f = run_rules(units, mappings, payload)
    ml_f = pd.DataFrame(ml_quality_findings(units))
    # 6. Similarity - does another requirement mean nearly the same thing?
    pairs = find_similar(units, sim_threshold, sim_method, sim_scope) if use_similarity else pd.DataFrame()
    sim_f = pd.DataFrame(similarity_findings(pairs)) if len(pairs) else pd.DataFrame()
    # 7. Validation results
    findings = pd.concat([f for f in (rule_f, ml_f, sim_f) if len(f)], ignore_index=True) \
        if any(len(f) for f in (rule_f, ml_f, sim_f)) else pd.DataFrame(columns=FINDING_COLS)
    if "category" in units and len(findings):
        findings = findings.merge(units[["req_id", "category"]], on="req_id", how="left")
    findings = (findings.assign(_o=findings["severity"].map(SEVERITY_ORDER))
                        .sort_values(["doc", "_o"]).drop(columns="_o").reset_index(drop=True))
    findings.insert(0, "finding_id", [f"F{i:04d}" for i in range(1, len(findings) + 1)])
    res = AnalysisResult(units, mappings, findings, pairs)
    # 8. Clarification questions
    if questions:
        res.questions, res.question_info = generate_questions(findings, use_llm=use_llm)
    return res


def analyse_files(paths, **kw) -> AnalysisResult:
    return analyse([extract_file(Path(p)) for p in paths], **kw)


def analyse_text(markdown: str, name: str = "uploaded", **kw) -> AnalysisResult:
    return analyse([parse_markdown(markdown, name)], **kw)


def main():
    ap = argparse.ArgumentParser(description="Run the full validation pipeline")
    ap.add_argument("--input", default=str(config.SYNTHETIC_DIR))
    ap.add_argument("--out", default=str(config.PROCESSED_DIR))
    ap.add_argument("--llm", action="store_true", help="generate questions with Ollama")
    ap.add_argument("--sim-method", default="embedding", choices=["embedding", "tfidf"])
    ap.add_argument("--threshold", type=float, default=config.SIMILARITY_THRESHOLD)
    a = ap.parse_args()
    files = sorted(p for p in Path(a.input).glob("*") if p.suffix.lower() in {".md", ".docx"})
    res = analyse_files(files, use_llm=a.llm, sim_method=a.sim_method, sim_threshold=a.threshold)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res.units.to_csv(out / "requirements_analysed.csv", index=False)
    res.mappings.to_csv(out / "mappings.csv", index=False)
    res.findings.to_csv(out / "findings.csv", index=False)
    res.similar_pairs.to_csv(out / "similar_pairs.csv", index=False)
    res.questions.to_csv(out / "clarification_questions.csv", index=False)
    print(f"{len(files)} documents | {len(res.units)} units | {len(res.findings)} findings | "
          f"{len(res.similar_pairs)} similar pairs | {len(res.questions)} questions -> {out}")
    print(res.findings.groupby(["issue_type", "source"]).size().to_string())


if __name__ == "__main__":
    main()
