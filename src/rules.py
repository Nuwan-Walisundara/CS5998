"""Stage 5a - Deterministic structural / completeness / wording checks.

Every finding is a dict with a common schema so rule, ML and similarity findings
can be merged in the validation results (stage 7).
"""
from __future__ import annotations

import re

import pandas as pd

from . import config

FINDING_COLS = ["doc", "req_id", "field", "issue_type", "detail", "severity", "source", "text",
                "related_req_id", "score"]

_VAGUE = re.compile(r"\b(" + "|".join(config.VAGUE_TERMS) + r")\b", re.I)
_MODAL = re.compile(config.MODAL_VERBS, re.I)
_UNQUANT = re.compile(config.UNQUANTIFIED_PATTERNS, re.I)
_NUMBER = re.compile(r"\d")


def finding(row, issue_type, detail, severity="Medium", source="Rule", **extra) -> dict:
    f = dict(doc=row.get("doc", ""), req_id=row.get("req_id", ""), field=row.get("field", ""),
             issue_type=issue_type, detail=detail, severity=severity, source=source,
             text=row.get("text_clean", row.get("text", "")), related_req_id="", score=None)
    f.update(extra)
    return f


# --------------------------------------------------------------------------- completeness
def check_required_fields(doc_df: pd.DataFrame) -> list[dict]:
    out, doc = [], doc_df["doc"].iloc[0]
    present = set(doc_df["field_key"])
    filled = set(doc_df.loc[~doc_df["is_empty"], "field_key"])
    for key, sev in config.REQUIRED_FIELDS.items():
        if key not in present:
            out.append(finding({"doc": doc, "field": key}, "Missing field",
                               f"Template field '{key}' is not present in the document", sev))
        elif key not in filled:
            row = doc_df[doc_df.field_key == key].iloc[0].to_dict()
            out.append(finding(row, "Missing value", f"'{row['field']}' is empty", sev))
    return out


def check_backends(doc_df: pd.DataFrame) -> list[dict]:
    out, doc = [], doc_df["doc"].iloc[0]
    started = []
    for sec in config.BACKEND_SECTIONS:
        part = doc_df[doc_df["subsection"].fillna("").eq(sec) | doc_df["section"].fillna("").eq(sec)]
        if part.empty or part["is_empty"].all():
            continue
        started.append(sec)
        filled = set(part.loc[~part["is_empty"], "field_key"])
        for req in config.BACKEND_REQUIRED.get(sec, []):
            if req not in filled:
                rows = part[part.field_key == req]
                row = rows.iloc[0].to_dict() if len(rows) else {"doc": doc, "field": req}
                out.append(finding(row, "Missing value",
                                   f"Backend section '{sec}' is partially filled but '{req}' is empty", "High"))
    if not started:
        out.append(finding({"doc": doc, "field": "Backend details"}, "Missing section",
                           "No backend (REST/SOAP, DB or file system) details are provided", "High"))
    return out


def check_consistency(doc_df: pd.DataFrame) -> list[dict]:
    out = []
    get = lambda k: doc_df.loc[doc_df.field_key == k, "text_clean"].str.cat(sep=" ").strip()
    status, date = get("approval status"), get("approval date")
    if re.search(r"approved", status, re.I) and not date:
        row = doc_df[doc_df.field_key == "approval date"].iloc[0].to_dict() if (doc_df.field_key == "approval date").any() else {"doc": doc_df.doc.iloc[0]}
        out.append(finding(row, "Inconsistency", "Status is approved but approval date is empty", "Low"))
    return out


# --------------------------------------------------------------------------- wording quality
def check_wording(doc_df: pd.DataFrame) -> list[dict]:
    out = []
    for _, r in doc_df[doc_df["is_nlp_unit"]].iterrows():
        row, t = r.to_dict(), r["text_clean"]
        if m := _VAGUE.search(t):
            out.append(finding(row, "Ambiguous term", f"Vague term '{m.group(0)}' has no measurable criterion",
                               "Medium", term=m.group(0)))
        if r["field_key"] == "acceptance criteria" and not _MODAL.search(t):
            out.append(finding(row, "Weak requirement", "No modal verb (shall/should/must) - obligation unclear", "Low"))
        if _UNQUANT.search(t) and not _NUMBER.search(t):
            out.append(finding(row, "Unquantified behaviour",
                               f"'{_UNQUANT.search(t).group(0)}' is mentioned without a count, interval or limit",
                               "Medium", term=_UNQUANT.search(t).group(0)))
        if r["n_words"] < 4:
            out.append(finding(row, "Incomplete requirement", "Requirement statement is too short to be testable", "Medium"))
    return out


# --------------------------------------------------------------------------- mappings
def check_mappings(mappings: pd.DataFrame, payload_keys: set[str], doc: str) -> list[dict]:
    out = []
    if mappings is None or mappings.empty:
        out.append(finding({"doc": doc, "field": "Mappings"}, "Missing section",
                           "No source-to-target mapping table found", "High"))
        return out
    for _, m in mappings.iterrows():
        label = f"{m['mapping_table']} row {m['row_no']} ({m['source_field'] or m['target_field']})"
        base = {"doc": doc, "req_id": f"{doc}:M{m['row_no']}", "field": m["mapping_table"],
                "text": f"{m['source']}.{m['source_field']} -> {m['target']}.{m['target_field']}"}
        if not str(m["mandatory"]).strip():
            out.append(finding(base, "Missing mapping info", f"Optional/Mandatory not specified in {label}", "Medium"))
        if not str(m["target_field"]).strip():
            out.append(finding(base, "Missing mapping info", f"Target field missing in {label}", "High"))
    mapped = {str(v).split("/")[-1].strip().lower()
              for col in ("source_field", "target_field") for v in mappings[col] if str(v).strip()}
    for key in sorted(payload_keys):
        if key.lower() not in mapped:
            out.append(finding({"doc": doc, "req_id": f"{doc}:payload", "field": "Request mapping", "text": key}, "Unmapped field",
                               f"Payload field '{key}' appears in the request payload but in no mapping table",
                               "Medium", term=key))
    return out


def run_rules(units: pd.DataFrame, mappings: pd.DataFrame | None = None,
              payload_keys: dict[str, set[str]] | None = None) -> pd.DataFrame:
    findings = []
    for doc, doc_df in units.groupby("doc", sort=False):
        findings += check_required_fields(doc_df)
        findings += check_backends(doc_df)
        findings += check_consistency(doc_df)
        findings += check_wording(doc_df)
        if mappings is not None:
            doc_map = mappings[mappings["doc"] == doc] if not mappings.empty else mappings
            findings += check_mappings(doc_map, (payload_keys or {}).get(doc, set()), doc)
    df = pd.DataFrame(findings)
    for c in FINDING_COLS:
        if c not in df:
            df[c] = None
    return df
