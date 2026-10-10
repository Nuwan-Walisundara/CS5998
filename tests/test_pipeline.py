import pandas as pd
import pytest

from src import config
from src.extract import extract_file, split_statements, to_frames
from src.preprocess import anonymise, preprocess
from src.questions import template_questions
from src.rules import run_rules

SAMPLE = config.SYNTHETIC_DIR / "SYN-01 Account Balance Inquiry API.md"


@pytest.fixture(scope="module")
def doc():
    return extract_file(SAMPLE)


@pytest.fixture(scope="module")
def units(doc):
    u, _ = to_frames([doc])
    return preprocess(u)


def test_split_numbered_list():
    assert split_statements("1. ESB will receive 2. ESB will validate 3. ESB will post") == [
        "ESB will receive", "ESB will validate", "ESB will post"]


def test_split_keeps_decimals():
    assert split_statements("1. Respond within 2.5 seconds 2. Retry 3 times") == [
        "Respond within 2.5 seconds", "Retry 3 times"]


def test_extract_units_and_mappings(doc):
    fields = {u["field_key"] for u in doc.units}
    assert {"business logic", "acceptance criteria", "anticipated peak time tps"} <= fields
    assert sum(u["field_key"] == "acceptance criteria" for u in doc.units) == 6
    assert len(doc.mappings) == 4
    assert {"accountNo", "accountType", "channelRef"} <= doc.payload_keys


def test_anonymise():
    t = anonymise("Call https://x.y/z from 10.1.2.3 or mail a@b.com, phone 255764399970")
    assert "<URL>" in t and "<IP>" in t and "<EMAIL>" in t and "<NUMBER>" in t


def test_rules_detect_known_issues(doc, units):
    _, maps = to_frames([doc])
    f = run_rules(units, maps, {doc.doc: doc.payload_keys})
    issues = set(zip(f.issue_type, f.field.str.lower()))
    assert ("Missing value", "anticipated peak time duration") in issues
    assert ("Missing value", "error responses(including http status code, payload)") in issues
    assert "Ambiguous term" in set(f.issue_type)            # "quickly", "user friendly"
    assert "Unquantified behaviour" in set(f.issue_type)    # retry without count
    assert ("Unmapped field", "request mapping") in issues  # accountType
    assert (f.issue_type == "Missing mapping info").any()   # channelRef has no Optional/Mandatory


def test_every_finding_gets_a_question(doc, units):
    _, maps = to_frames([doc])
    f = run_rules(units, maps, {doc.doc: doc.payload_keys})
    q = template_questions(f)
    assert len(q) > 0 and q["question"].str.len().min() > 20
