# 1. Data Preprocessing

## 1.1 Overview
Raw WSO2 integration requirement documents are semi-structured (template tables, numbered lists,
free text, JSON payloads and mapping tables). Preprocessing converts them into a clean, anonymised,
labelled dataset of individual requirement statements that can be used to train and evaluate the
classification models. All steps are implemented as reproducible scripts:

```
python -m src.extract && python -m src.preprocess
```

| Step | Script | Purpose |
|---|---|---|
| 1. Extraction | `src/extract.py` | Parse template tables and sections; split into requirement units and mapping rows |
| 2. Anonymisation | `src/preprocess.py` + `config/anonymise_terms.txt` | Remove confidential names and identifiers |
| 3. Cleaning | `src/preprocess.py` | Normalise whitespace, markup and leftover numbering |
| 4. Filtering | `src/preprocess.py` | Keep only non-empty natural-language requirement statements |
| 5. Labelling | Manual, `docs/annotation_guidelines.md` | Assign `category` and `quality_label` |
| 6. Dataset assembly | — | Combine real and synthetic labelled data into `labelled_all.csv` |

## 1.2 Steps

**Step 1 – Extraction.** Each document (Markdown or .docx) is parsed using its template structure.
Template fields (e.g. *Security details*, *Anticipated peak time TPS*) become one unit each; free-text
sections (*Business logic*, *Acceptance criteria*, *Use-case description*) are split into individual
statements; source-to-target mapping tables become separate mapping rows; JSON keys are collected from
request payloads for later consistency checks. Each unit keeps its source document and field for traceability.

**Step 2 – Anonymisation.** Because the documents are confidential, sensitive content is replaced before analysis:
- Pattern-based: URLs → `<URL>`, IP addresses → `<IP>`, e-mails → `<EMAIL>`, long numbers → `<NUMBER>`,
  credentials → `<CREDENTIAL>`.
- Term mapping (`config/anonymise_terms.txt`, kept private): organisation-specific names are replaced by
  **general fintech terms** rather than a blank placeholder, so that the text keeps its meaning for the models:

| Original | Replaced with |
|---|---|
| CRDB | Bank |
| BOT | Central Bank |
| TISS | Real-Time Gross Settlement (RTGS) system |
| GePG | National Digital Payment Infrastructure (NDPI) |
| TIPS | Real-Time Retail Payment Switch |
| PAYSTREAM | Salary processing System |
| TRA | Tax Authority |

**Step 3 – Cleaning.** Whitespace is collapsed, Markdown escapes and bold markers are removed, and leftover
list numbering (e.g. `1.`, `2)`) is stripped.

**Step 4 – Filtering.** Units are flagged as *empty* (blank or template placeholder text), *payload*
(JSON/XML/HTTP blocks) or *template field*. Only non-empty, natural-language requirement statements
(`is_nlp_unit = True`) are passed to the NLP models. Empty fields are not discarded: they feed the
rule-based completeness checks.

**Step 5 – Labelling.** Each NLP statement was labelled with a `category` (7 classes) and a
`quality_label` (Clear / Ambiguous / Incomplete) following `docs/annotation_guidelines.md`.
<!-- TODO: add second-annotator agreement (Cohen's kappa on 50 rows) when available -->

**Step 6 – Dataset assembly.** The 52 real labelled statements were combined with 48 synthetic,
anonymous example statements (`labelled_synthetic.csv`) to give enough examples per class for 5-fold
stratified cross-validation. Duplicated texts were removed.

## 1.3 Data flow and counts

| Stage | Count |
|---|---|
| Input documents | 9 |
| Extracted units | 352 (262 template fields + 90 requirement statements) |
| Mapping rows (separate table) | 221 |
| Empty units (blank / placeholder) | 174 |
| Payload blocks | 40 |
| **NLP requirement statements (after filtering)** | **52** |
| Synthetic labelled statements added | 48 |
| **Final labelled dataset (`labelled_all.csv`)** | **100** |

```
9 documents → 352 units → 90 requirement statements → 52 NLP statements → +48 synthetic → 100 labelled rows
```

Label distribution of the final dataset:

| category | n | | quality_label | n |
|---|---|---|---|---|
| API Exposure | 23 | | Clear | 61 |
| Integration/Backend | 22 | | Incomplete | 22 |
| Validation/Security | 12 | | Ambiguous | 17 |
| Functional/Other | 12 | | | |
| Notification/Callback | 11 | | | |
| Error Handling | 10 | | | |
| Data/Storage | 10 | | | |

## 1.4 Before-and-after example

| | Text |
|---|---|
| **Raw (extracted)** | This request is used to requests transfer of funds or payments in **TIPS** from the payer FSP to payee FSP |
| **Clean (anonymised)** | This request is used to requests transfer of funds or payments in **Real-Time Retail Payment Switch** from the payer FSP to payee FSP |
| **Labels** | category = Functional/Other, quality_label = Clear |

## 1.5 Output
The final dataset is `data/annotations/labelled_all.csv` (100 rows), ready for model training:

| Column | Description |
|---|---|
| `req_id` | Unique ID (`R…` real, `S…` synthetic) |
| `text_clean` | Anonymised, cleaned requirement text – model input |
| `category` | Requirement type – target 1 |
| `quality_label` | Clear / Ambiguous / Incomplete – target 2 |
| `doc`, `field` | Source document and template section (traceability; empty for synthetic rows) |

Confidential raw documents, the term-mapping file and the real labelled data are excluded from the public
repository; only synthetic data is published for reproducibility.

## 1.6 Limitations
- The real labelled set is small (52 statements); synthetic data is shorter and cleaner than real text,
  so results on the combined set may be optimistic.
- Some statements still contain embedded analyst Q&A text, which makes them long (up to ~160 words).
- One of the 9 documents (`7.2_Batch-Payment`) produced no units because its layout differs from the template.
