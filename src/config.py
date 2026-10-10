"""Central configuration for the requirements validation pipeline."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"              # confidential documents (git-ignored)
SYNTHETIC_DIR = ROOT / "data" / "synthetic"  # public, reproducible examples
PROCESSED_DIR = ROOT / "data" / "processed"
ANNOTATION_DIR = ROOT / "data" / "annotations"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
ANON_TERMS_FILE = ROOT / "config" / "anonymise_terms.txt"  # one client/system name per line

SEED = 42

# --- Extraction -----------------------------------------------------------
# Template fields whose value is free-text requirements (split into units).
TEXT_REQUIREMENT_FIELDS = {
    "use-case description",
    "business logic",
    "acceptance criteria",
    "any other (if required)",
    "validations",
    "additional comments",
}
# Fields that should be split on "1. 2. 3." numbering.
LIST_FIELDS = {"business logic", "acceptance criteria", "validations"}
PAYLOAD_FIELD_PREFIX = "request payload parameters"
MAPPING_HEADER_KEYS = {"source", "target"}

# --- Structural validation -----------------------------------------------------
# field (lower-case) -> severity when the field is empty
REQUIRED_FIELDS = {
    "use-case name": "High",
    "use-case description": "High",
    "business logic": "High",
    "acceptance criteria": "High",
    "api consumers": "Medium",
    "approval status": "Medium",
    "approval date": "Low",
    "anticipated peak time tps": "High",
    "anticipated peak time duration": "Medium",
    "protocol(restful,soap,file)": "High",
    "path parameters": "High",
    "request payload parameters(including http status code)": "High",
    "response payload parameters(including http status code)": "High",
    "error responses(including http status code, payload)": "High",
}
BACKEND_SECTIONS = {"restfull/soap backend", "dbms system", "file system"}
# Within a backend section that has been started, these must also be filled.
BACKEND_REQUIRED = {
    "restfull/soap backend": ["name of the backend", "protocol", "url/ip", "security details"],
    "dbms system": ["name of the db", "url/s", "security details"],
    "file system": ["file type", "url/s", "security details"],
}
# Placeholder values left from the template, treated as empty.
TEMPLATE_PLACEHOLDERS = {"sql/nosql", "oracle,mysql", "n/a", "tbd", "-", "xxx"}

VAGUE_TERMS = [
    "quickly", "fast", "slow", "appropriate", "appropriately", "properly", "as required",
    "as needed", "if possible", "etc", "user[- ]friendly", "efficient", "efficiently",
    "adequate", "some", "several", "various", "normal", "reasonable", "robust",
    "seamless", "flexible", "minimal", "maximum possible", "as soon as possible", "asap",
]
MODAL_VERBS = r"\b(shall|should|must|will|needs? to|has to)\b"
UNQUANTIFIED_PATTERNS = r"\b(retry|retries|retried|timeout|timed[- ]out|time[- ]out|expire|threshold|limit)\b"

# --- Models ------------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.80"))  # tune with similarity.py evaluate

# --- LLM (optional) ------------------------------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))
