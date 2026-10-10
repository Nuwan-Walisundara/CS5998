# CS5998 – Intelligent Requirements Analysis and Validation for WSO2 Integration Projects

Capstone project (Master of Data Science & AI). The system analyses WSO2 integration requirement
documents, detects gaps and quality issues using rules, classical NLP/ML and transformer embeddings,
and generates **clarification questions for the customer**.

```
Requirement Document → Extraction → Preprocessing → Classification → Quality/Completeness Analysis
   → Similarity Analysis → Validation Results → Clarification Questions (template + optional Ollama LLM)
```

| Stage | Module | Technique |
|---|---|---|
| 2. Extraction | `src/extract.py` | Structure-aware parsing of template tables (Markdown / .docx), numbered-list splitting |
| 3. Preprocessing | `src/preprocess.py` | Anonymisation (URL, IP, e-mail, numbers, credentials, custom names), normalisation, features |
| 4. Classification | `src/classify.py` | **TF-IDF + LR baseline vs. sentence-embedding + LR**, stratified k-fold CV |
| 5a. Completeness/structure | `src/rules.py` | Required fields, backend sections, mapping checks, payload↔mapping consistency, vague terms |
| 5b. Quality (ML) | `src/classify.py --target quality_label` | Same experiment for `Clear / Ambiguous / Incomplete` |
| 6. Similarity | `src/similarity.py` | Cosine similarity (embeddings vs. TF-IDF), threshold tuned on labelled pairs |
| 7. Validation results | `app.py` | Streamlit dashboard |
| 8. Clarification questions | `src/questions.py` | Template questions (baseline/fallback) + Ollama LLM with ID check |
| EDA | `src/eda.py` | Milestone 2 statistics and figures |

The LLM **never decides what an issue is**. It only phrases questions for findings produced by stages 5–6.
Question IDs it returns are checked against the findings, and any finding it skips gets a template question.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```
Optional LLM: install [Ollama](https://ollama.com), then `ollama pull llama3.1:8b` (configure with the `OLLAMA_MODEL` / `OLLAMA_URL` env vars).

## Quick start (synthetic documents)
```bash
python -m src.pipeline --input data/synthetic        # writes data/processed/*.csv
streamlit run app.py
```

## Full workflow on real documents
1. Put documents in `data/raw/` (git-ignored). Copy `config/anonymise_terms.example.txt` to `config/anonymise_terms.txt` and list client/system names.
2. Extract and clean: `python -m src.extract && python -m src.preprocess`. This also creates `data/annotations/to_annotate.csv`.
3. EDA: `python -m src.eda --input data/raw`, outputs in `results/eda/`.
4. Annotate `category` and `quality_label` following `docs/annotation_guidelines.md`, and save as `data/annotations/labelled.csv`.
5. Classification experiments (baseline vs. transformer):
   ```bash
   python -m src.classify --labels data/annotations/labelled.csv --target category
   python -m src.classify --labels data/annotations/labelled.csv --target quality_label
   ```
   Results go to `results/<target>/` (comparison.csv, metrics.json, confusion matrices, cv_predictions.csv for error analysis). The best model is saved to `models/`, and the pipeline uses it automatically. Before a model exists, categories come from a keyword fallback.
6. Similarity threshold:
   ```bash
   python -m src.pipeline --input data/raw
   python -m src.similarity sample --n 200      # label is_duplicate → data/annotations/pairs_labelled.csv
   python -m src.similarity evaluate
   ```
   Set the chosen value with `SIMILARITY_THRESHOLD=0.xx`.
7. Run the dashboard: `streamlit run app.py`, upload documents, review findings, generate and edit questions, and export CSV or e-mail text.

Try the pipeline end to end on `data/annotations/labelled_synthetic.csv` (48 invented examples).

## Reproducibility
- Fixed seed `SEED = 42` (`src/config.py`): CV folds, classifiers, Ollama sampling (`temperature 0`, `seed`).
- Pinned minimum versions in `requirements.txt`. Embedding model: `sentence-transformers/all-MiniLM-L6-v2`.
- Confidential documents, processed data, annotations, models and results are git-ignored. Only synthetic documents are published.

## Repository layout
```
app.py                    Streamlit dashboard
src/                      pipeline stages (see table)
data/synthetic/           public synthetic requirement documents
data/annotations/         labelled_synthetic.csv (public); real labels are git-ignored
docs/annotation_guidelines.md
config/anonymise_terms.example.txt
tests/                    pytest unit tests
```
