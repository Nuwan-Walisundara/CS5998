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


## Data preprocessing
|Step|	Script|	What it does|
|---|---|---|
|Text conversion		|extract.py			|Reads .md and .docx documents|
|Structure parsing		|extract.py			|Reads the template tables and normalises field names|
|Segmentation			|extract.py			|Splits sections into individual requirement statements and numbered items|
|Mapping extraction		|extract.py			|One row per source-to-target mapping, plus the JSON keys from payloads|
|Anonymisation			|preprocess.py		|Replaces URLs, IPs, e-mails, long numbers and credentials, plus your custom names, Predefine mappings defined at config/anonymise_terms.txt |
|Normalisation			|preprocess.py		|Collapses whitespace and removes leftover numbering|
|Filtering				|preprocess.py		|Marks empty template fields and payload blocks, and keeps only real statements for NLP|
|Features				|preprocess.py		|Word and character counts|




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


**Mostly yes.** Training and production use the **same preprocessing code and the same saved model**, but there's one weak point: the training file is a snapshot.

## The two paths

**Training** (`python -m src.classify`):
```
data/raw → extract.py → preprocess.py → to_annotate.csv → (you label) → labelled.csv
                                                                          + labelled_synthetic.csv
                                                                          → labelled_all.csv → classify.py → models/*.joblib
```

**Production** (`src.pipeline` and the dashboard):
```
new document → extract.py → preprocess.py → NLP statements → models/*.joblib → predictions
```

## Where they match ✅
- **The same extraction and cleaning code** (`extract.py`, `preprocess.py`): anonymisation, normalisation and the same filter that keeps only real requirement statements (`is_nlp_unit`). The 52 rows you labelled are exactly the statements production would analyse.
- **The same input column:** both read `text_clean`.
- **The same text-to-numbers step:** the saved model file contains both the TF-IDF or MiniLM step *and* the classifier. Production can't accidentally vectorise text differently from training.
- **The same model in the command line and the dashboard:** both call `analyse()`.

## Where they can drift apart ⚠️
1. **The labelled CSV is a frozen copy.** Training reads `text_clean` from the CSV; it doesn't re-run preprocessing. So if preprocessing changes, the training text goes out of date until you rebuild the file. That's exactly what happened with anonymisation: production started outputting *"Real-Time Retail Payment Switch"* while the models had been trained on *"TIPS"*. I fixed the CSV by hand, and **retraining** will bring the models back in line.
2. **Synthetic rows skip extraction.** The 48 rows in `labelled_synthetic.csv` were written by hand, so they never went through `extract.py`. They're also shorter and cleaner than real statements. That's a known difference between training and production data; say so in the limitations.
3. **There's no automatic check.** Nothing warns you if a model was trained on text from an older preprocessing version.

## Make them match fully
The robust fix is to have `classify.py` **re-apply `anonymise()` + `normalise()` to `text_clean`** before training. Then every training run uses the current preprocessing rules automatically, and the labelled CSV can't go out of date. It's a small change of about 2 lines.

For your report, the point to make: *"Preprocessing is shared code between training and inference, and the vectoriser is saved inside the model, which prevents training/serving skew; the labelled dataset is re-cleaned at training time so that changes to preprocessing rules propagate."* That last part is true once the fix is in.

Should I add it?


-----------------
To add data, put more labelled rows in a CSV and run the training command again. The model trains from scratch on whatever CSV you give it, so it isn't updated in steps.

## Option A – Add rows by hand
Open `data/annotations/labelled_synthetic.csv` (or a copy of it) in Excel or VS Code and add rows in the same 4 columns:

```
req_id,text_clean,category,quality_label
N001,ESB should retry the CBS call 3 times at 30 s interval on timeout,Error Handling,Clear
N002,The API should respond quickly,API Exposure,Ambiguous
```

## Option B – Take requirements from your WSO2 documents (recommended)
1. Copy the documents (`.md` or `.docx`) into `data/raw/`.
2. Extract and anonymise them:
   ```bash
   python -m src.extract && python -m src.preprocess
   ```
   This creates `data/annotations/to_annotate.csv`, with one requirement per row.
3. Fill in `category` and `quality_label` for each row, following `docs/annotation_guidelines.md`. Save the file as `data/annotations/labelled.csv`.
4. Optionally, combine it with the sample data so you have more examples:
   ```bash
   python -c "import pandas as pd; pd.concat([pd.read_csv('data/annotations/labelled_synthetic.csv'), pd.read_csv('data/annotations/labelled.csv')]).drop_duplicates('text_clean').to_csv('data/annotations/labelled_all.csv', index=False)"
   ```

## Retrain
```bash
python -m src.classify --labels data/annotations/labelled_all.csv --target category
```
```bash
python -m src.classify --labels data/annotations/labelled_all.csv --target quality_label
```
Each run replaces the files in `models/` and `results/`. The pipeline and dashboard pick up the new model automatically.

## Labelling tips
- **Aim for at least 30–50 examples per category.** Training uses 5-fold cross-validation, so any category with fewer than 5 rows will cause problems.
- **Keep the classes reasonably balanced**, or the model will mostly predict the largest category.
- **The guidelines define a 7th category, `Functional/Other`, that has no sample rows yet.** Add examples of it, or the model can't learn it.
- **Use exactly the label spellings from the guidelines.** "Error handling" and "Error Handling" count as two different classes.
- **Get a second person to label about 50 rows** and report Cohen's kappa. The guidelines ask for this, and it supports your report.

----
`annotation_guidelines.md` already exists at [docs/annotation_guidelines.md](docs/annotation_guidelines.md), so there's nothing to create. That's the file I summarised in my last answer. You can click the link or open it with:

```bash
open docs/annotation_guidelines.md
```

If you meant something different, tell me which:
- **A copy in another folder**, for example next to `to_annotate.csv` or in the parent capstone folder
- **An expanded version**, with more WSO2 examples per category, edge cases, or a section for your second labeller
- **A Word or PDF version** for your report or supervisor


-------------------------------


You now have 74 documents and 539 rows to label. 88 rows match ones you already labelled, so I'll carry those labels over by text and label the rest myself.

All **539 rows** in [to_annotate.csv](data/annotations/to_annotate.csv) now have a `category` and a `quality_label`. Your `data/raw/` folder grew to 74 documents, so the file was regenerated with many new rows.

- **88 rows** matched sentences you'd already labelled, so they got the same labels.
- **416 new distinct sentences** I labelled following `docs/annotation_guidelines.md`. The remaining rows repeat sentences from those two groups and got the same labels.

The file as it was before is backed up at `/private/tmp/claude-502/-Users-nuwanw-personal-MDANDAI-S3-CS5998-Capstone-Project/28bafb5d-d5b0-4281-8575-80c1d03270e2/scratchpad/backup/to_annotate.before.csv`.

| category | n | | quality_label | n |
|---|---|---|---|---|
| Functional/Other | 154 | | Clear | 412 |
| API Exposure | 111 | | Incomplete | 97 |
| Validation/Security | 69 | | Ambiguous | 30 |
| Integration/Backend | 69 | | | |
| Notification/Callback | 52 | | | |
| Data/Storage | 50 | | | |
| Error Handling | 34 | | | |

Every category now has at least 34 real examples, which is enough for training without synthetic data.

**How I handled the less obvious rows:**
- **Pointers like "Refer to the attached…\<URL\>", section headers ("MESSAGE PROCESSING", "1.", "Refer:")** → Functional/Other, Incomplete. They contain no requirement.
- **SWIFT message identification** (message type, direction, clearing type) → Functional/Other. Duplicate checks, signature checks and account validation → Validation/Security.
- **MT202 returns, reversals and "retry if timed out"** → Error Handling.
- **"Retry until acknowledged" or "send until acknowledged", with no limit or interval** → Incomplete.
- **Rows with open analyst questions** (`< @… can you pls …>`, `* Pls state the criteria…`) → Incomplete.
- **"Appropriate", "accordingly", "as per requirement", "desired response", "properly"** → Ambiguous.

**⚠️ Fix before you train or submit:**
1. **Real people's names are still in the text:** *@Tonny Kayage, @chanaka sisira kumara, @Anuradha Prasanna, @Solomon.Komba*. Other organisation names are there too: *KCBL, TACOBA, Montran, ZECO*, and the BIC code *CORUTZTZ*. Add them to `config/anonymise_terms.txt`, for example `KCBL = Partner Bank`, `Montran = Clearing Gateway`, or a name alone so it becomes `<ORG>`. Also consider a rule that removes `@names` automatically.
2. **There are duplicate documents.** *GePG Sweeping* and *Transfer List API* each appear twice under different file names, giving 73 repeated sentences. Remove the duplicates from `data/raw/`, or remove duplicate texts before training. Otherwise the same sentence can land in both the training and test folds and inflate your scores.
3. **Review the labels.** As before, these are my draft. Check the Functional/Other and Incomplete groups especially.

**Next steps:**
```bash
cd /Users/nuwanw/personal/MDANDAI/S3/CS5998-Capstone_Project/CS5998 && source .venv/bin/activate
```
```bash
cp data/annotations/to_annotate.csv data/annotations/labelled.csv
```
```bash
python -c "import pandas as pd; pd.concat([pd.read_csv('data/annotations/labelled_synthetic.csv'), pd.read_csv('data/annotations/labelled.csv')]).drop_duplicates('text_clean').to_csv('data/annotations/labelled_all.csv', index=False)"
```

Then retrain both models with `--target category` and `--target quality_label`.
