"""Stage 8 - Clarification question generation.

Layer 1: deterministic template questions (always available, evaluation baseline).
Layer 2: optional local LLM (Ollama) that rewrites/groups the detected findings into
         customer-ready questions. The LLM never decides *what* is an issue - it only
         phrases questions for findings produced by stages 5-6. If Ollama is not
         reachable or returns invalid output, Layer 1 is used.
"""
from __future__ import annotations

import json
import time

import pandas as pd
import requests

from . import config

QUESTION_COLS = ["doc", "req_ids", "topic", "question", "priority", "source"]

TEMPLATES = {
    "Missing field": "The template field '{field}' is missing. Could you please provide this information?",
    "Missing value": "Could you please provide the '{field}'? ({detail})",
    "Missing mapping info": "{detail}. Could you please confirm this mapping detail?",
    "Missing section": "{detail}. Could you please provide these details?",
    "Ambiguous term": "Requirement {req_id} uses the term '{term}'. What measurable value or condition is expected? (\"{text}\")",
    "Weak requirement": "Is requirement {req_id} mandatory or optional? (\"{text}\")",
    "Unquantified behaviour": "For requirement {req_id}, please specify the exact values for '{term}' (e.g. number of retries, interval, limit) and the behaviour after the final failure. (\"{text}\")",
    "Incomplete requirement": "Requirement {req_id} appears incomplete: \"{text}\". Could you please elaborate the expected behaviour?",
    "Inconsistency": "{detail}. Could you please confirm the correct information?",
    "Unmapped field": "The request payload contains '{term}', but it is not in any mapping table. How should this field be mapped and is it mandatory?",
    "Possible duplicate": "Requirements {req_id} and {related_req_id} look very similar ({score_pct}). Do they describe the same requirement, or how do they differ?",
    "Ambiguous": "Requirement {req_id} may be ambiguous: \"{text}\". Could you please clarify the expected behaviour and acceptance condition?",
    "Incomplete": "Requirement {req_id} may be missing information: \"{text}\". What additional details (values, conditions, systems) apply?",
}
_PRIORITY = {"High": "High", "Medium": "Medium", "Low": "Low"}


def _fmt(f: dict) -> str:
    tpl = TEMPLATES.get(f.get("issue_type"), "Please clarify: {detail} (\"{text}\")")
    score = f.get("score")
    vals = {k: ("" if v is None or (isinstance(v, float) and pd.isna(v)) else v) for k, v in f.items()}
    vals.update(field_l=str(vals.get("field", "")).lower(), term=vals.get("term", ""),
                score_pct=f"{float(score):.0%}" if vals.get("score") != "" else "")
    vals["text"] = str(vals.get("text", ""))[:200]
    try:
        return tpl.format(**vals)
    except KeyError:
        return f"Please clarify: {f.get('detail')}"


def template_questions(findings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, f in findings.iterrows():
        d = f.to_dict()
        rows.append(dict(doc=d.get("doc"), req_ids=str(d.get("req_id") or d.get("field")),
                         topic=d.get("issue_type"), question=_fmt(d),
                         priority=_PRIORITY.get(d.get("severity"), "Medium"), source="Template"))
    q = pd.DataFrame(rows, columns=QUESTION_COLS)
    return q.drop_duplicates(subset=["doc", "question"]).reset_index(drop=True)


# --------------------------------------------------------------------------- LLM layer
PROMPT = """You are a senior business analyst on a WSO2 API/integration project.
An automated validation tool analysed the requirement document "{doc}" (use case: {use_case}).
It detected the issues listed below. Each issue has an ID in square brackets.

ISSUES:
{issues}

Write clarification questions to send to the CUSTOMER so the gaps can be closed.
Rules:
- Ask ONLY about the listed issues. Do not invent new requirements, systems or values.
- Merge issues about the same topic into one question; every issue ID must be covered.
- Be specific and polite; reference the requirement/field; where useful suggest an answer format (e.g. "e.g. 50 TPS").
- priority: High = blocks design/implementation, Medium = needed before testing, Low = documentation.
Return ONLY JSON: {{"questions": [{{"issue_ids": ["I1"], "topic": "...", "question": "...", "priority": "High|Medium|Low"}}]}}"""


def ollama_available() -> bool:
    try:
        return requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=3).ok
    except requests.RequestException:
        return False


def llm_questions(findings: pd.DataFrame, use_case: str = "", model: str = config.OLLAMA_MODEL,
                  max_issues: int = 40) -> tuple[pd.DataFrame, dict]:
    """Returns (questions, run_info). Falls back to templates on any failure."""
    info = {"model": model, "latency_s": None, "fallback": False, "dropped_unknown_ids": 0, "uncovered": 0}
    if findings.empty:
        return pd.DataFrame(columns=QUESTION_COLS), info
    out = []
    for doc, fd in findings.groupby("doc", sort=False):
        fd = fd.head(max_issues).reset_index(drop=True)
        ids = {f"I{k+1}": r for k, r in fd.iterrows()}
        issues = "\n".join(f"[{i}] {r['issue_type']} | field: {r['field']} | req: {r['req_id']} | "
                           f"{r['detail']} | text: \"{str(r['text'])[:200]}\"" for i, r in ids.items())
        t0 = time.time()
        try:
            resp = requests.post(f"{config.OLLAMA_URL}/api/generate", timeout=config.OLLAMA_TIMEOUT, json={
                "model": model, "prompt": PROMPT.format(doc=doc, use_case=use_case or doc, issues=issues),
                "format": "json", "stream": False, "options": {"temperature": 0, "seed": config.SEED}})
            resp.raise_for_status()
            qs = json.loads(resp.json()["response"])["questions"]
        except Exception as e:
            info.update(fallback=True, error=str(e))
            out.append(template_questions(fd))
            continue
        info["latency_s"] = round(time.time() - t0, 1)
        covered, rows = set(), []
        for q in qs:
            valid = [i for i in q.get("issue_ids", []) if i in ids]       # hallucination guard
            info["dropped_unknown_ids"] += len(q.get("issue_ids", [])) - len(valid)
            if not valid or not q.get("question"):
                continue
            covered |= set(valid)
            rows.append(dict(doc=doc, req_ids=", ".join(sorted({str(ids[i]["req_id"] or ids[i]["field"]) for i in valid})),
                             topic=q.get("topic", ""), question=q["question"].strip(),
                             priority=q.get("priority", "Medium"), source=f"LLM ({model})"))
        missed = [i for i in ids if i not in covered]
        info["uncovered"] += len(missed)
        df = pd.DataFrame(rows, columns=QUESTION_COLS)
        if missed:  # guarantee coverage: template question for every issue the LLM skipped
            df = pd.concat([df, template_questions(fd.loc[[int(i[1:]) - 1 for i in missed]])])
        out.append(df)
    return pd.concat(out, ignore_index=True), info


def generate_questions(findings: pd.DataFrame, use_llm: bool = False, use_case: str = "") -> tuple[pd.DataFrame, dict]:
    if use_llm and ollama_available():
        return llm_questions(findings, use_case)
    return template_questions(findings), {"fallback": use_llm, "model": None}


def to_email_text(questions: pd.DataFrame, doc: str) -> str:
    order = {"High": 0, "Medium": 1, "Low": 2}
    q = questions.assign(_o=questions["priority"].map(order).fillna(3)).sort_values("_o")
    lines = [f"Subject: Clarifications required - {doc}", "", "Dear team,", "",
             "While reviewing the integration requirements we identified the following points that need clarification:", ""]
    for n, (_, r) in enumerate(q.iterrows(), 1):
        lines.append(f"{n}. [{r['priority']}] {r['question']}  (Ref: {r['req_ids']})")
    lines += ["", "Kind regards,"]
    return "\n".join(lines)
