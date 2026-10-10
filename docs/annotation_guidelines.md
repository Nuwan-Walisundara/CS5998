# Annotation Guidelines

Annotate each row of `data/annotations/to_annotate.csv` (one requirement statement) with two labels.
Read the statement on its own, together with its source field. Do not use knowledge of the implementation.

## 1. `category`: what kind of requirement is it?
| Label | Definition | Example |
|---|---|---|
| Validation/Security | Input/signature/token validation, authentication, authorisation, encryption | "ESB should validate the request signature" |
| Error Handling | Retries, timeouts, reversals, error responses, failure behaviour | "ESB should reverse the debit if the credit fails" |
| Notification/Callback | Callbacks, acknowledgements, SMS/e-mail/alerts to parties | "ESB should send callback to TIPS after completion" |
| Data/Storage | Persisting, logging, auditing, archiving data | "ESB should save the transaction in the data store" |
| Integration/Backend | Calling/transforming to/from backend systems (CBS, biller, GePG) | "ESB should post the transaction to CBS" |
| API Exposure | Exposing/receiving an API, resources, channels | "ESB should expose a REST API for transfers" |
| Functional/Other | Anything else (business rules, descriptions) | "The use case allows customers to pay bills" |

If a statement fits several categories, choose its **primary action** (the main verb).

## 2. `quality_label`: is there a potential problem?
| Label | Rule | Example |
|---|---|---|
| Clear | Testable: actor + action + object, any needed values or conditions are given | "Retry 3 times at 30 s interval on timeout" |
| Ambiguous | Uses vague or unmeasurable wording (quickly, properly, as required, user friendly, secure) | "The API should respond quickly" |
| Incomplete | Missing an essential element: value, condition, target system, or failure behaviour | "ESB should retry posting in case of timeout" |

If both Ambiguous and Incomplete apply, choose **Ambiguous**.

## 3. Similar pairs: `is_duplicate`
`1` if the two statements require the same behaviour, so implementing one satisfies the other. Otherwise `0`.
Business-logic and acceptance-criteria statements describing the same step count as `1`.

## Agreement
A second practitioner labels a random 50 rows on their own. Report Cohen's kappa
(`sklearn.metrics.cohen_kappa_score`) for each label, and resolve disagreements by discussion before training.
