# Public Safety ML — Toronto KSI Collision Analysis

A supervised machine learning project analyzing Toronto's **Killed or Seriously Injured (KSI)** traffic collision dataset to understand what drives fatal outcomes on Toronto's roads, and to predict collision severity from the conditions present at the scene.

---

## 1. The Problem

Every year, thousands of traffic collisions occur across the City of Toronto. Most result in property damage or minor injury — but a subset end in death or serious, life-altering injury. These are the collisions captured in the **KSI (Killed or Seriously Injured)** dataset published by the Toronto Police Service.

The core problem this project addresses:

> **Given the conditions of a collision — location, time, road/weather conditions, who and what was involved, and driver behaviour — can we predict whether it will be fatal, and which factors matter most?**

This is fundamentally a **road-safety and public-policy problem**, not just a data problem. Traffic fatalities are largely preventable — they result from a combination of infrastructure design, enforcement gaps, environmental conditions, and human behaviour. Understanding *which combinations of factors* push a collision from "serious" to "fatal" is the first step toward preventing it.

## 2. Why This Matters — The Business Perspective

Toronto operates under **Vision Zero**, the city's road-safety plan aiming to eliminate traffic fatalities and serious injuries. A model like this one supports that goal by turning historical collision records into **actionable risk intelligence**. Several stakeholders have a direct interest in this analysis:

| Stakeholder | Business Need | How This Project Helps |
|---|---|---|
| **Toronto Police Service / City of Toronto (Vision Zero)** | Prioritize enforcement and infrastructure spending where it saves the most lives | Identifies the strongest predictors of fatality (e.g. speeding, pedestrian involvement, lighting/road conditions) to target interventions |
| **Urban planners / Transportation Services** | Decide where to invest in traffic calming, better lighting, redesigned intersections | Neighbourhood- and location-level patterns highlight high-risk areas and conditions |
| **Insurance providers** | Better understand risk factors that correlate with severe outcomes | Feature importance from the model reflects real-world severity drivers, not just claim frequency |
| **Emergency services** | Anticipate when/where KSI-severity collisions are more likely | Time-of-day and seasonal patterns inform resource and ambulance staffing |
| **Public awareness campaigns** | Target messaging (e.g. anti-speeding, anti-impaired-driving campaigns) | Quantifies how much specific behaviours (speeding, alcohol, red-light running) contribute to fatal outcomes |

The economic and human cost of a fatal collision is enormous — beyond the immediate loss of life, there are long-term healthcare costs, legal and insurance costs, lost productivity, and lasting impact on families and communities. Even modest improvements in identifying and mitigating high-risk conditions translate into real lives saved and costs avoided.

## 3. Project Goal

Frame this as a **binary classification problem**:

- **Target variable:** `ACCLASS_BINARY` — `1 = Fatal`, `0 = Non-Fatal` (property-damage and non-fatal-injury outcomes are combined, since the business question is "fatal vs. not")
- **Objective:** Train a model that predicts the likelihood a given collision will be fatal, based on features known at (or shortly after) the time of the incident, while avoiding any feature that would leak the outcome itself (e.g. injury severity fields).

## 4. Dataset

**`KSI_data.csv`** — Toronto Police Service KSI collision records, spanning 2006–present.

**Row structure — important:** each row represents one *person* involved in a collision, not one collision. A crash with a driver, a passenger, and a pedestrian produces three rows sharing the same `ACCNUM` (collision ID), each with a different `INVTYPE`. The dataset's ~18,900 rows correspond to roughly half that many actual distinct collisions. This matters throughout the pipeline — most importantly for the train/test split (Section 6).

Key feature groups:
- **Collision details:** `ACCLASS`, `ACCLOC`, `IMPACTYPE`, `ROAD_CLASS`, `TRAFFCTL`
- **Environmental conditions:** `LIGHT`, `VISIBILITY`, `RDSFCOND`
- **Involvement details:** `INVTYPE`, `INJURY`, `PEDESTRIAN`, `CYCLIST`, `AUTOMOBILE`, `MOTORCYCLE`, `TRUCK`
- **Driver behaviour flags:** `SPEEDING`, `AG_DRIV` (aggressive driving), `REDLIGHT`, `ALCOHOL`, `DISABILITY`
- **Location:** `DISTRICT`, `HOOD_158`/`HOOD_140` (neighbourhood codes), `LATITUDE`, `LONGITUDE`

A significant part of the work in this project is *not* the modelling itself, but the disciplined handling of missingness — in this dataset, "missing" rarely means "unknown." It usually means "not applicable" (e.g. a pedestrian-only field left blank for a driver) or "did not occur" (e.g. `ALCOHOL` blank means alcohol was not a factor). Treating every blank field as generically missing would destroy the signal in the data, so the preprocessing pipeline applies a different, justified strategy to each semantic group of columns rather than one blanket approach.

## 5. Project Structure

```
GroupProject/
├── KSI_data.csv                       # Toronto KSI collision dataset
├── public_safety_ml.py                # Exploration -> Visualization -> Modelling -> Export
│
├── app.py                             # Flask analytics API (Deliverable 5)
├── model_service.py                   # Bundle loading + request-to-DataFrame preparation
├── templates/index.html               # Jinja browser client — form over every model feature
├── client.py                          # Python API client, tested on held-out data
├── make_postman_collection.py         # Regenerates the Postman collection from the bundle
├── postman_collection.json            # Importable Postman collection (8 requests)
├── requirements.txt                   # Pinned dependencies
├── DELIVERABLE_5_GUIDE.md             # Deployment walkthrough — run it, read the UI
│
├── model_bundle.pkl                   # Pickled serving pipeline + metadata  (generated)
├── test_samples.csv                   # Held-out test rows for client testing (generated)
├── model_metadata.json                # Human-readable bundle contents       (generated)
├── part4_report_numbers.md            # Part 4 table + confusion matrices    (generated)
│
├── 1_collisions_by_year_severity.png  # Yearly trend, by severity
├── 2_collisions_by_hour.png           # Hour-of-day risk profile
├── 3_contributing_factors.png         # Frequency of behavioural risk factors
├── 4_correlation_heatmap.png          # Relationships between contributing factors
├── 5_top_neighbourhoods.png           # Highest-collision neighbourhoods
├── 6_roc_curves_all_models.png        # ROC comparison across all tuned models
├── 7_feature_importance_rf.png        # Top 20 features (Random Forest)
├── 8_threshold_selection.png          # Decision-threshold selection on validation
└── README.md                          # Project documentation
```

The five `(generated)` artifacts are written by `public_safety_ml.py`; the API
will not start without `model_bundle.pkl`.

## 6. Methodology

The pipeline runs in five deliberate stages, in this order. Each stage builds on
the in-memory output of the one before it — nothing reloads the CSV or repeats
an earlier transformation, so a decision justified once cannot be silently
undone later:

### Stage 1 — Data Exploration & Cleaning
- Profiles the dataset (shape, types, missingness, ranges) to understand what we're working with before touching it.
- Applies a **domain-driven missing-value strategy**, split into six groups by *why* the data is missing rather than treating every NaN the same way:

| Group | Example Columns | Strategy | Business Rationale |
|-------|---------|----------|-----------|
| **1 — Binary involvement flags** | `PEDESTRIAN`, `ALCOHOL`, `SPEEDING` | `fillna("No")` | Missing = factor was not present, not unknown |
| **2 — Role-conditional attributes** | `PEDTYPE`, `PEDACT`, `CYCLISTYPE` | `fillna("Not Applicable")` | Only relevant to pedestrians/cyclists — doesn't apply to a driver-only record |
| **3 — Event-conditional (`FATAL_NO`)** | `FATAL_NO` | `fillna(0)` | Only populated for fatal collisions |
| **4 — Genuinely missing reports** | `ACCLOC`, `INITDIR`, `ROAD_CLASS` | Drop (<3%) or `"Unknown"` (~28%) | Real reporting gaps, handled by size of the gap |
| **5 — Identifiers / low-value fields** | `OFFSET`, `STREET2`, `ACCNUM` | Fill or drop | Location/ID descriptors, not analytical signal. Missing `ACCNUM` values each get a unique placeholder rather than a shared one, so unrelated records are never mistaken for the same collision downstream |
| **6 — Role-dependent `INJURY`** | `INJURY` | Conditional fill by `INVTYPE` | "None" for people present at the scene; "Not Applicable" for bystander roles |

### Stage 2 — Data Visualization
Once the data is clean, exploratory charts surface the patterns a business stakeholder would actually ask about:

- **Collisions by year and severity** — Total KSI volume trends downward over the period, from ~1,400+ collisions/year in 2006–2007 to roughly 600–650/year by 2020–2022 (with a step-down starting in 2020, likely reflecting reduced traffic during COVID). However, the **fatal (red) band barely shrinks** even as total volume roughly halves — fatal counts hover in the 100–200/year range across almost the entire period. In other words: fewer collisions overall, but a similar number are still deadly, meaning the *share* of KSI collisions that are fatal has been creeping upward. This is a critical finding for Vision Zero — raw collision reduction alone isn't translating into a proportional fatality reduction.
- **Collisions by hour of day** — Non-fatal collision volume climbs steadily from a 3–4am low to a clear evening-commute peak around 5–7pm, tracking overall traffic volume. Fatal collisions don't follow that same curve — they stay comparatively substantial even during low-traffic overnight hours. That's a classic road-safety pattern: fewer crashes at night, but a higher proportion of them are severe, consistent with higher speeds, lower visibility, and more impaired driving overnight. This has direct implications for enforcement and emergency-resource scheduling.
- **Contributing factors** — Aggressive driving (`AG_DRIV`) dominates, appearing in roughly 4x as many KSI collisions as the next-most-common factor, speeding (~2,400), followed by red-light running (~1,600), then alcohol and disability much further behind (under 1,000 each). This ranks where enforcement and public-awareness spending would have the most reach — aggressive driving as a category outweighs the other behavioural factors combined.
- **Correlation heatmap** — Most risk factors are only weakly related to each other, which is itself informative: these are largely *independent* risk pathways rather than one "bad driver" profile causing everything. The exceptions worth noting: speeding and aggressive driving (0.39 correlation), and aggressive driving and red-light running (0.29), cluster together — suggesting a genuine "reckless driving" pattern rather than three unrelated issues, and a case for combined enforcement targeting all three at once. Pedestrian involvement is mildly negatively correlated with cyclist and motorcycle involvement (-0.28, -0.23), indicating these tend to be distinct collision types rather than co-occurring.
- **Top neighbourhoods** — Waterfront Communities-The Island leads by a wide margin (~730 collisions), ahead of West Humber-Clairville (~570) and Bay Street Corridor (~440). The list mixes two distinct neighbourhood profiles: dense downtown cores with heavy pedestrian exposure (Waterfront Communities, Bay Street Corridor, Church-Yonge Corridor) and outer/suburban areas with higher-speed arterials (West Humber-Clairville, Woburn, Rouge, Wexford/Maryvale). This matters for policy — a downtown intersection redesign and a suburban speed-enforcement push are different interventions, but both neighbourhood types show up as high-KSI zones and likely need tailored responses.

### Stage 3 — Data Modelling
Builds directly on the cleaned exploration output (no re-loading or re-imputing):

1. **Target preparation** — collapses the 3-class outcome into a binary `Fatal` vs `Non-Fatal` target.
2. **Leakage removal** — drops any column that describes the outcome itself (`INJURY`, `FATAL_NO`), since including these would let the model "cheat" rather than predict from causal/contextual factors.
3. **Feature selection with justification** — every dropped and kept column is documented with a business/statistical reason (sparsity, redundancy, or genuine predictive relevance).
4. **Feature engineering** — extracts `HOUR` from raw `TIME` and `YEAR` from `DATE`, groups rare categories (<1%) into `"Other"` to avoid noisy one-hot columns, and encodes binary flags as 0/1.
5. **Crash-grouped 80/20 train/test split** — because each row is a *person*, not a collision, a plain random split can place different people from the *same crash* on both sides of the split, letting the model see near-duplicate rows (identical time, location, weather) during training and testing. To prevent this, the split uses `GroupShuffleSplit` on `ACCNUM`, so every row belonging to a given collision lands entirely in train or entirely in test, never both. This is a deliberate departure from a plain stratified split — it costs the ability to guarantee an exact class ratio in both sets, but it removes a real leakage path that a row-level split would otherwise hide.
6. **Preprocessing pipeline** — numeric features are median-imputed and standardized; categorical features are mode-imputed and one-hot encoded, all inside a single `ColumnTransformer` fit only on training data (no test-set leakage).
7. **Class imbalance handling (SMOTE)** — fatal collisions are a small minority of all KSI records (roughly 6:1 non-fatal to fatal). SMOTE is applied **only to the training set** to avoid inflating evaluation metrics with synthetic test data, since a model that just predicts "non-fatal" every time would score misleadingly well on accuracy without being useful to any stakeholder.

### Stage 4 — Model Building & Evaluation

Five algorithms are trained on the SMOTE-balanced training set and evaluated on
the untouched, naturally imbalanced test set: logistic regression, decision
tree, SVM, random forest, and a neural network (MLP). Hyperparameters are tuned
with `GridSearchCV` where the search space is small and cheap (logistic
regression, decision tree) and `RandomizedSearchCV` where an exhaustive grid
would be impractical (SVM, random forest, MLP).

Tuning is scored on **F1 for the Fatal class**, not accuracy. With a ~14% fatal
rate, a model that predicts "Non-Fatal" for everything scores 86% accuracy
while being operationally worthless.

### Stage 5 — Deployment

The winning model is exported and served as a local analytics API. Three
problems had to be solved to get from "a trained model in a script" to "a
model an API can serve":

**1. The preprocessor and the models were fitted separately.** Section 7 fits
the `ColumnTransformer` on its own, and every estimator is then fitted on an
already-encoded matrix. Pickling an estimator alone would produce something
that only accepts a ~200-column encoded array — useless to an endpoint
receiving JSON. Both fitted objects are therefore wrapped into a single
`Pipeline`, which accepts raw feature rows. Neither component is refitted:
sklearn only refits on `.fit()`, so the deployed model is exactly the evaluated
one.

**2. Rare-category grouping is data-dependent.** Feature engineering folds
categories below 1% frequency into `"Other"` using frequencies computed across
the whole training set. A single incoming API row has no frequency
distribution, so recomputing would group nothing — a category that trained as
`"Other"` would arrive raw and be silently one-hot encoded as all-zeros,
producing a wrong prediction with no error raised. The surviving category lists
are captured at training time, shipped in the bundle, and replayed on every
request.

**3. The decision threshold.** See below.

Everything the API needs travels in **one pickle** (`model_bundle.pkl`) — the
pipeline, the threshold, the feature order, the rare-category maps, the form
schema, and the metrics — so the served model and the metadata describing it
cannot drift apart.

#### Choosing the decision threshold

The default 0.5 cut-off is arbitrary here. The project's whole framing is that
a **false negative — a fatal collision predicted non-fatal — is the expensive
error**, so the deployed API should not inherit a threshold that treats both
error types as equally costly.

The obvious fix — maximising F-beta with beta=2 to weight recall more heavily —
is a **poor criterion on this dataset**. Precision on the Fatal class is bounded
below by the class prevalence (~15%), while recall can always be driven toward
1.0 by lowering the threshold, so F2 rewards ever-lower cut-offs. Its optimum
falls below 0.01, off the bottom of the searched range, and scores 0.505 against
**0.474 for a classifier that simply labels every record Fatal**. That narrow
margin is the problem: most of F2's score at its optimum comes from the same
indiscriminate direction as the all-positive baseline rather than from genuine
discrimination. The script computes the all-positive baseline explicitly and
prints the comparison rather than asserting the conclusion.

**Youden's J** (`TPR − FPR`) is used instead. It cannot degenerate — J = 0 for
both all-positive and all-negative predictions, so its maximum is necessarily
an interior, genuinely discriminating point. It is also the standard cut-point
criterion for the ROC curve already plotted in Stage 4, and it is
prevalence-independent, which matters because the model was fitted on
SMOTE-balanced data but scores naturally imbalanced rows.

J is maximised using `roc_curve()`, which evaluates **every distinct predicted
probability** as a candidate cut-point, rather than a fixed grid. This matters
for the same reason F2 was rejected: a grid can only report an optimum at one
of its own points, so a maximum sitting near the edge is indistinguishable from
one the grid is simply too coarse to resolve. The exact search removes grid
resolution as a factor. `8_threshold_selection.png` plots how every criterion
behaves across the threshold range, including the all-positive F2 reference
line that makes the degeneracy visible.

Critically, the threshold is tuned on a **validation split carved out of the
training data, never on the test set**. `X_train` is split again by `ACCNUM`, a
clone of the winning estimator is refitted on the sub-training half, and the
threshold is selected on the held-out validation half at its natural class
balance. Choosing a cut-off that maximises a score on the test set and then
reporting that same score would be leakage; because the test set played no part
in the selection, the reported figures remain an honest held-out estimate.

#### Deployment results

The **neural network (MLP)** is deployed. It had both the highest test F1
(0.340) and the highest ROC-AUC (0.736) — the latter matters most here, because
ROC-AUC is threshold-independent, so the best-ranking model stays the right
choice after the cut-off is retuned.

**It is served with isotonic calibration** (`CalibratedClassifierCV`, 3-fold),
applied at the deployment stage rather than during model comparison. The raw MLP
pinned 65% of its predictions below 0.001 or above 0.999, which made the served
probability useless — the form returned "0.0%" for nearly every input.
Calibration cuts that to 19%. Isotonic mapping is monotonic, so it cannot
reorder records: ROC-AUC is unchanged at 0.740 and the Deliverable 3 model
comparison still stands. Note this means the served estimator is not
byte-identical to the one evaluated in Deliverable 3, since calibration refits
the base model across folds.

Selected threshold: **0.0197** (validation J = 0.354, chosen from 415 candidate
cut-points).

Test-set performance, on data untouched by both training and threshold
selection:

| Threshold | Accuracy | Precision | Recall | F1 | F2 |
|-----------|----------|-----------|--------|-------|-------|
| 0.50 (sklearn default) | 85.4% | 44.3% | 21.0% | 0.285 | 0.235 |
| **0.0197 (deployed)** | 68.2% | 25.3% | **66.1%** | **0.365** | **0.499** |

Retuning the threshold **catches 223 more fatalities** on the test set (missed
fatalities fall from 391 to 168), at the cost of 837 additional false alarms
(131 → 968).

Confusion matrix at the deployed threshold:

|  | Predicted Non-Fatal | Predicted Fatal |
|--|--------------------|-----------------|
| **Actual Non-Fatal** | 2112 | 968 |
| **Actual Fatal** | 168 | 327 |

**Read the accuracy figure carefully.** 68.2% is *below* the 86% a model would
score by predicting "Non-Fatal" for everything — because that trivial model
catches zero fatalities, which is the entire point of building this. The
deployed configuration finds two thirds of them. This is a deliberate operating
point, not a weak result, but it must be presented as such rather than buried.

Validation F1 is essentially flat between thresholds 0.02 and 0.20
(0.385–0.399), so the choice of cut-off is not really a choice about overall
quality — it only decides where on the recall/precision curve the tool sits.
Youden's J places it at the high-recall end, consistent with the
false-negative-is-costlier framing used throughout.

## 7. Requirements

```bash
pip install -r requirements.txt
```

`scikit-learn` is **pinned to an exact version**. A pickled estimator is not
portable across scikit-learn versions — loading `model_bundle.pkl` under a
different version either warns and misbehaves or fails outright, so the API
must run on the same version that trained the model.

## 8. Usage

> For deployment specifically, **[DELIVERABLE_5_GUIDE.md](DELIVERABLE_5_GUIDE.md)**
> is the step-by-step walkthrough: what to run, what every part of the interface
> means, how to read the client output, and the design decisions behind the
> deployed threshold.

### Step 1 — Train the models and export the deployable bundle

```bash
python public_safety_ml.py
```

This runs the entire pipeline end to end and:
1. Prints exploration and missing-value diagnostics.
2. Saves the eight visualization PNGs.
3. Trains and tunes all five algorithms, printing the comparison table.
4. Writes `model_bundle.pkl`, `test_samples.csv`, and `model_metadata.json`.

Both the fitted preprocessor and the fitted models must be in memory at export
time, so the export runs at the end of a **single full training run** — it is
not a separate script. Expect several minutes, dominated by the hyperparameter
searches.

### Step 2 — Start the API

```bash
python app.py
```

Serves on `http://127.0.0.1:5000`. The bundle is unpickled once at start-up.

### Step 3 — Test it

**Browser** — open `http://127.0.0.1:5000`, pick a record from the
*held-out test record* dropdown to fill the form, and submit. The page shows
the predicted class, the fatality probability against the decision threshold,
and the record's recorded outcome for comparison.

**Python client** — with the API running, in a second terminal:

```bash
python client.py
```

Runs six checks against held-out data: health, schema retrieval, a single
prediction, a 25-record batch with a confusion breakdown, a partial-input
request, and four error cases.

**Postman** — import `postman_collection.json`
(*File → Import*). Eight requests, each with test assertions, runnable
via the Collection Runner. Regenerate after retraining with
`python make_postman_collection.py`.

**curl**:

```bash
curl http://127.0.0.1:5000/health
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"INVTYPE": "Pedestrian", "SPEEDING": 1, "ALCOHOL": 0, "LIGHT": "Dark"}'
```

## 9. API Reference

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET`  | `/` | Browser client — HTML form covering every model feature |
| `POST` | `/predict-form` | Form submission; re-renders the page with the result |
| `POST` | `/predict` | JSON in, JSON out — the main analytics endpoint |
| `POST` | `/predict/batch` | Scores a list of records in one request |
| `GET`  | `/schema` | Every accepted field, with valid options and ranges |
| `GET`  | `/samples` | Held-out test records with their recorded outcomes |
| `GET`  | `/health` | Liveness, loaded model, threshold, and test metrics |

### Request

`POST /predict` accepts a JSON object of feature values. Field names are
case-insensitive.

```json
{
  "INVTYPE": "Pedestrian",
  "SPEEDING": 1,
  "ALCOHOL": 0,
  "LIGHT": "Dark",
  "LATITUDE": 43.7,
  "LONGITUDE": -79.4
}
```

Three conveniences are built in, each mirroring a training-time transformation:

- **Partial bodies are accepted.** Any field not supplied is filled with the
  training-set default, and every substitution is listed in `warnings` — the
  API never silently invents input.
- **`TIME` is accepted in place of `HOUR`** (e.g. `1430` → hour 14), matching
  the `TIME // 100` feature engineering.
- **Binary flags accept `1`/`0`, `"Yes"`/`"No"`, or `true`/`false`**, all
  normalised to the 0/1 the model was trained on.

### Response

```json
{
  "prediction": 1,
  "label": "Fatal",
  "probability_fatal": 0.8123,
  "probability_non_fatal": 0.1877,
  "threshold": 0.11,
  "model": "Neural Network",
  "warnings": ["..."]
}
```

`prediction` applies the deployed threshold to `probability_fatal` — it is not
sklearn's default 0.5 `.predict()` output.

### Errors

Invalid input returns HTTP 400 with an explanatory `error` string rather than a
stack trace or a silent guess. Values outside the training range, and
categories never seen during training, are accepted but flagged in `warnings`,
since the model can still score them — the caller is told the prediction is an
extrapolation.

## 10. Limitations

- **Person-level, not crash-level.** Each row in the dataset is one person
  involved in a collision, and features like `INVTYPE` and `INVAGE` describe
  that person. The API therefore predicts whether *a given involved person* is
  likely to be killed, not whether a collision as a whole will be fatal.
- **Screening aid, not a verdict.** Precision on the Fatal class is modest, so
  a substantial share of flagged records will not be fatal. The threshold was
  chosen deliberately to trade precision for recall — appropriate for
  prioritising records for review, not for any automated decision about an
  individual.
- **The decision surface is not well-behaved, and this is the most important
  limitation.** Flipping a single risk flag from No to Yes on a real record
  moves the prediction in the *wrong* direction 27.6% of the time. The
  behaviour is also inconsistent: the same flag can raise the estimate on one
  record and lower it on the next. In practice this means a record with
  speeding, alcohol, aggressive driving and red-light running can score lower
  than the same record with speeding alone. The model is usable for **ranking
  records**, which is what ROC-AUC measures and what the screening use case
  needs, but it is **not usable for counterfactual reasoning** — "what happens
  to the risk if this one factor changes" is a question it cannot answer
  reliably. See §11 for the investigation.

- **Probabilities were saturated before calibration.** The raw MLP pinned 65%
  of its test predictions below 0.001 or above 0.999. Isotonic calibration at
  the deployment stage reduced that to 19%, which is why the served
  `probability_fatal` now varies meaningfully across inputs. Treat it as a
  calibrated ranking score, not a literal actuarial likelihood.

- **One feature is a data artifact.** Setting `DIVISION` to `NSA` (police
  division not recorded) alone drives the prediction to ~0.99. The model has
  learned that a missing administrative field predicts fatality, almost
  certainly reflecting how those reports are filed rather than anything about
  the collision itself.
- **Historical scope.** The model reflects 2006–2022 Toronto KSI reporting
  practice and does not account for subsequent road or policy changes.

## 11. Investigating the unstable decision surface

The instability above was found by using the deployed form, not by reading
metrics — setting speeding, alcohol, aggressive driving and red-light running
all to Yes produced a *lower* fatality probability than speeding alone. What
follows is the investigation, including the hypotheses that turned out to be
wrong, because the negative results are what narrowed it down.

**Measurement.** For 200 real test records, each of the four behavioural risk
flags currently set to No was flipped to Yes, one at a time, and the change in
predicted probability recorded. Turning a risk factor *on* should not *reduce*
predicted fatality risk; every case where it does is counted as a monotonicity
violation. Saturation is the share of test predictions pinned below 0.001 or
above 0.999.

| Variant | Violations | ROC-AUC | Saturation |
|---|---|---|---|
| SMOTE on encoded data (original) | 33.6% | 0.7363 | 65.1% |
| SMOTENC on raw features | 36.8% | 0.7281 | 54.7% |
| No resampling at all | 36.7% | 0.7304 | 58.3% |
| MLP with lower learning rate, more L2 | 42.7% | 0.7487 | 31.0% |
| **MLP + isotonic calibration (deployed)** | **27.6%** | **0.7398** | **18.8%** |
| Logistic regression | 17.1% | 0.6819 | 0.0% |

**Hypothesis 1 — SMOTE on one-hot data. Wrong.** SMOTE runs after the
`ColumnTransformer`, so it interpolates in the encoded space and produces
synthetic rows with fractional categorical values (`PEDESTRIAN = 0.37`), which
no real record can have. `SMOTENC` handles categorical columns by taking the
neighbourhood mode instead, so it should have removed those impossible points.
It made the violation rate slightly *worse*, and removing resampling entirely
did not help either. Resampling is not the cause.

**Hypothesis 2 — the MLP is over-trained. Also wrong.** Cutting the learning
rate tenfold and raising L2 a hundredfold reduced saturation substantially but
made the violation rate *worse* (42.7%).

**What the logistic regression comparison revealed.** Its 113 violations turned
out to be exactly the 113 probe rows where `AG_DRIV` was 0 — the fitted
coefficient for aggressive driving is negative (−0.262), so it lowers the
estimate for *every* record, consistently. That is an explainable statistical
result (aggressive-driving collisions, conditional on everything else, are
typically lower-speed urban intersection crashes), not instability.

That reframed the finding: **the problem is not the violation rate, it is the
inconsistency.** A linear model applies one rule in one direction to every
record. The MLP applies a different rule per record, because its ~200
dimensional decision surface is shaped by a training set that occupies only a
small region of that space — and single-flag edits walk into corners no real
record ever occupied.

**What was adopted.** Isotonic calibration, applied at the deployment stage.
It cuts saturation from 65% to 19% and leaves ROC-AUC unchanged (isotonic
mapping is monotonic, so record ranking is preserved by construction). It
reduces violations from 33.6% to 27.6% but does **not** resolve them.

Logistic regression is the only variant that behaves consistently, and it was
not adopted because ROC-AUC falls from 0.736 to 0.682 — it would contradict the
model comparison in Deliverable 3 while still not being a strong model. The
honest conclusion is that this dataset supports record *ranking* far better
than it supports per-record *explanation*, and the deployed API is documented
accordingly.

## 12. Course

**COMP247 — Supervised Learning**
Centennial College — Semester 4