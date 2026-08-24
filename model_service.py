"""
DELIVERABLE 5 — Model serving layer.

Loads the pickled bundle produced by the Deliverable 5 section of
public_safety_ml.py and turns loose user input (JSON body or HTML form) into
the exact DataFrame shape the fitted pipeline expects.

This module is deliberately separate from app.py so that the Flask routes stay
thin and the input handling can be imported by the test client and exercised
without starting a server.
"""

import os
import pickle

import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE_PATH = os.environ.get("MODEL_BUNDLE", os.path.join(PROJECT_DIR, "model_bundle.pkl"))
SAMPLES_PATH = os.path.join(PROJECT_DIR, "test_samples.csv")

# Values that a form or JSON body may use for the binary involvement flags.
# The model was trained on 0/1 integers, so everything is normalised to that.
TRUTHY = {"1", "yes", "y", "true", "t", "on"}
FALSY = {"0", "no", "n", "false", "f", "off", ""}


class InputError(ValueError):
    """Raised when a request body cannot be turned into a valid model row."""


def load_bundle(path=BUNDLE_PATH):
    """Deserialize the model bundle written by public_safety_ml.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model bundle not found at {path}. Run `python public_safety_ml.py` "
            "first — its Deliverable 5 section writes model_bundle.pkl."
        )
    with open(path, "rb") as f:
        bundle = pickle.load(f)

    # A pickled estimator is not portable across scikit-learn versions. A
    # mismatch can unpickle "successfully" and then behave differently, so it
    # is surfaced at start-up rather than left to show up as odd predictions.
    import sklearn
    trained_with = bundle.get("sklearn_version")
    if trained_with and trained_with != sklearn.__version__:
        print(
            f"WARNING: model_bundle.pkl was created with scikit-learn "
            f"{trained_with} but {sklearn.__version__} is installed. "
            "Predictions may differ from the reported metrics. "
            "Install the pinned version from requirements.txt, or retrain."
        )

    return bundle


def load_test_samples(path=SAMPLES_PATH):
    """Held-out rows exported at training time, for the demo dropdown/client."""
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def _coerce_binary(name, value):
    """Accept 1/0, 'Yes'/'No', true/false for an involvement flag."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value in (0, 1):
            return int(value)
        raise InputError(f"{name}: binary flag must be 0 or 1, got {value!r}")
    text = str(value).strip().lower()
    if text in TRUTHY:
        return 1
    if text in FALSY:
        return 0
    raise InputError(f"{name}: cannot interpret {value!r} as a 0/1 flag")


def _coerce_numeric(name, value):
    try:
        return float(value)
    except (TypeError, ValueError):
        raise InputError(f"{name}: expected a number, got {value!r}")


def prepare_input(payload, bundle):
    """
    Turn a dict of user-supplied values into a one-row DataFrame the fitted
    pipeline can consume.

    Three training-time behaviours have to be replayed here, because the
    pipeline itself only starts at the ColumnTransformer:

      1. HOUR was engineered from TIME (TIME // 100) in section 5a. Callers may
         send either; TIME is converted when HOUR is absent.
      2. The involvement flags were converted from 'Yes'/'No' to 1/0 in 5d.
      3. Rare categories were folded into 'Other' in 5c using frequencies
         computed over the whole training set. Those frequencies cannot be
         recovered from a single row, so the saved category lists are replayed
         instead. Skipping this would let a rare value reach the encoder
         unmapped and be silently encoded as all-zeros — a wrong prediction
         with no error raised.

    Returns (DataFrame, warnings).
    """
    if not isinstance(payload, dict):
        raise InputError("Request body must be a JSON object of feature values.")

    schema = {f["name"]: f for f in bundle["field_schema"]}
    feature_order = bundle["feature_order"]
    rare_maps = bundle.get("rare_category_maps", {})
    warnings = []

    # Case-insensitive lookup so 'invage' and 'INVAGE' both work.
    supplied = {str(k).strip().upper(): v for k, v in payload.items()}

    # Convenience: derive HOUR from a TIME value like 1430 when HOUR is absent.
    if "HOUR" in schema and "HOUR" not in supplied and "TIME" in supplied:
        supplied["HOUR"] = _coerce_numeric("TIME", supplied["TIME"]) // 100
        warnings.append("HOUR derived from TIME (TIME // 100), matching training-time feature engineering.")

    unknown_keys = [k for k in supplied if k not in schema and k != "TIME"]
    if unknown_keys:
        warnings.append(f"Ignored fields not used by the model: {', '.join(sorted(unknown_keys))}")

    row = {}
    defaulted = []

    for name in feature_order:
        spec = schema[name]
        raw = supplied.get(name, None)

        # Missing or blank -> training-set default, so a partial body still works.
        if raw is None or (isinstance(raw, str) and raw.strip() == ""):
            row[name] = spec["default"]
            defaulted.append(name)
            continue

        if spec["type"] == "binary":
            row[name] = _coerce_binary(name, raw)

        elif spec["type"] == "numeric":
            value = _coerce_numeric(name, raw)
            lo, hi = spec["min"], spec["max"]
            if not (lo <= value <= hi):
                warnings.append(
                    f"{name}={value:g} falls outside the training range "
                    f"[{lo:g}, {hi:g}] — the prediction is an extrapolation."
                )
            row[name] = value

        else:  # categorical
            value = str(raw).strip()
            # Replay section 5c rare-category grouping.
            kept = rare_maps.get(name)
            if kept is not None and value not in kept:
                warnings.append(f"{name}: '{value}' was a rare category at training time — mapped to 'Other'.")
                value = "Other"
            if value not in spec["options"]:
                warnings.append(
                    f"{name}: '{value}' was never seen during training — the encoder "
                    "treats it as all-zeros, so it contributes no signal."
                )
            row[name] = value

    if defaulted:
        warnings.append(
            f"{len(defaulted)} field(s) not supplied, filled with the training default: "
            f"{', '.join(defaulted)}"
        )

    frame = pd.DataFrame([row])[feature_order]

    # Force numeric dtypes: a one-row frame built from a dict can otherwise
    # infer `object`, which SimpleImputer(strategy='median') rejects.
    for name in feature_order:
        if schema[name]["type"] in ("numeric", "binary"):
            frame[name] = pd.to_numeric(frame[name], errors="coerce")

    return frame, warnings


def predict_frame(frame, bundle):
    """Run the fitted pipeline over a prepared frame and apply the tuned threshold."""
    pipeline = bundle["pipeline"]
    threshold = bundle["threshold"]
    labels = bundle["target_labels"]

    proba = pipeline.predict_proba(frame)[:, 1]
    preds = (proba >= threshold).astype(int)

    return [
        {
            "prediction": int(p),
            "label": labels[int(p)],
            "probability_fatal": round(float(pr), 4),
            "probability_non_fatal": round(float(1 - pr), 4),
        }
        for p, pr in zip(preds, proba)
    ]


def predict_one(payload, bundle):
    """Full path for a single record: validate -> prepare -> predict."""
    frame, warnings = prepare_input(payload, bundle)
    result = predict_frame(frame, bundle)[0]
    result["threshold"] = bundle["threshold"]
    result["model"] = bundle["model_name"]
    if warnings:
        result["warnings"] = warnings
    return result


def samples_as_records(bundle, limit=None):
    """Held-out test rows as dicts, with their true labels kept alongside."""
    samples = load_test_samples()
    if samples.empty:
        return []
    if limit:
        samples = samples.head(limit)

    feature_order = bundle["feature_order"]
    records = []
    for position, (_, row) in enumerate(samples.iterrows()):
        features = {c: row[c] for c in feature_order if c in samples.columns}
        # numpy scalars are not JSON-serialisable
        features = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in features.items()}
        records.append({
            # ORIGINAL_INDEX traces a row back to the source dataset; fall back
            # to position if an older CSV without that column is present.
            "id": int(row["ORIGINAL_INDEX"]) if "ORIGINAL_INDEX" in samples.columns else position,
            "actual": int(row["ACTUAL_ACCLASS_BINARY"]),
            "actual_label": str(row["ACTUAL_LABEL"]),
            "features": features,
        })
    return records
