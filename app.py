"""
DELIVERABLE 5 — Flask analytics API for the KSI fatality model.

Serves the pickled model bundle from public_safety_ml.py on localhost:

    GET  /                 Jinja form covering every model feature (browser client)
    POST /predict-form     Form submission -> re-renders the page with the result
    POST /predict          JSON in, JSON out  (Postman / requests / curl)
    POST /predict/batch    List of records -> list of predictions
    GET  /schema           Field definitions the model accepts
    GET  /samples          Held-out test records the model never trained on
    GET  /health           Liveness + which model/threshold is loaded

Run:  python app.py        ->  http://127.0.0.1:5000
"""

import os
import traceback

from flask import Flask, jsonify, render_template, request

import model_service as svc

app = Flask(__name__)

# Load the bundle once at start-up rather than per request — unpickling a
# fitted neural network on every call would dominate the response time.
BUNDLE = svc.load_bundle()
SAMPLES = svc.samples_as_records(BUNDLE)

print(f"[startup] Loaded {BUNDLE['model_name']} "
      f"(trained {BUNDLE['trained_at']}, sklearn {BUNDLE['sklearn_version']})")
print(f"[startup] Decision threshold: {BUNDLE['threshold']:.4f}")
print(f"[startup] Expecting {len(BUNDLE['feature_order'])} features")
print(f"[startup] {len(SAMPLES)} held-out test records available at /samples")


# ---------------------------------------------------------------------------
# Form layout
# ---------------------------------------------------------------------------
# The form exposes every feature the model consumes, which is a lot of inputs
# to face as one flat list. Grouping is presentation only — it never changes
# what gets sent. Any feature not named here still renders, under "Other",
# so a change to feature selection can't silently drop a field from the form.

FIELD_GROUPS = [
    ("Location", "Where the collision happened",
     ["LATITUDE", "LONGITUDE", "DISTRICT", "DIVISION", "HOOD_158", "ACCLOC", "ROAD_CLASS"]),
    ("Time & environment", "Conditions at the time of the collision",
     ["HOUR", "LIGHT", "VISIBILITY", "RDSFCOND"]),
    ("Collision details", "How the collision occurred",
     ["IMPACTYPE", "TRAFFCTL", "INITDIR", "VEHTYPE"]),
    ("Person involved", "The model scores one involved person per request",
     ["INVTYPE", "INVAGE"]),
]


def _grouped_schema():
    schema = {f["name"]: f for f in BUNDLE["field_schema"]}
    assigned = set()
    groups = []

    for title, blurb, names in FIELD_GROUPS:
        fields = [schema[n] for n in names if n in schema]
        assigned.update(f["name"] for f in fields)
        if fields:
            groups.append({"title": title, "blurb": blurb, "fields": fields})

    flags = [f for f in BUNDLE["field_schema"]
             if f["type"] == "binary" and f["name"] not in assigned]
    assigned.update(f["name"] for f in flags)
    if flags:
        groups.append({
            "title": "Contributing factors",
            "blurb": "Involvement and behaviour flags — tick every one that applies",
            "fields": flags,
        })

    leftover = [f for f in BUNDLE["field_schema"] if f["name"] not in assigned]
    if leftover:
        groups.append({"title": "Other features", "blurb": "", "fields": leftover})

    return groups


GROUPED_SCHEMA = _grouped_schema()


# ---------------------------------------------------------------------------
# Browser client — Jinja template
# ---------------------------------------------------------------------------

def _render(result=None, error=None, submitted=None):
    return render_template(
        "index.html",
        groups=GROUPED_SCHEMA,
        schema=BUNDLE["field_schema"],
        samples=SAMPLES,
        model_name=BUNDLE["model_name"],
        threshold=BUNDLE["threshold"],
        threshold_note=BUNDLE.get("threshold_note", ""),
        metrics=BUNDLE.get("test_metrics", {}),
        trained_at=BUNDLE["trained_at"],
        sklearn_version=BUNDLE["sklearn_version"],
        n_features=len(BUNDLE["feature_order"]),
        result=result,
        error=error,
        submitted=submitted or {},
    )


@app.route("/", methods=["GET"])
def index():
    return _render()


@app.route("/predict-form", methods=["POST"])
def predict_form():
    payload = request.form.to_dict()
    try:
        result = svc.predict_one(payload, BUNDLE)
        return _render(result=result, submitted=payload)
    except svc.InputError as exc:
        return _render(error=str(exc), submitted=payload), 400
    except Exception:
        traceback.print_exc()
        return _render(error="Unexpected server error while scoring the record.",
                       submitted=payload), 500


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({
            "error": "Request body must be JSON. Send Content-Type: application/json "
                     "with an object of feature values.",
            "hint": "GET /schema lists every accepted field; GET /samples returns ready-made bodies.",
        }), 400
    try:
        return jsonify(svc.predict_one(payload, BUNDLE))
    except svc.InputError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Unexpected server error while scoring the record."}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be JSON."}), 400

    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        return jsonify({
            "error": "Send a JSON list of records, or an object with a 'records' list."
        }), 400

    try:
        results = [svc.predict_one(rec, BUNDLE) for rec in records]
    except svc.InputError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Unexpected server error while scoring the batch."}), 500

    return jsonify({
        "count": len(results),
        "model": BUNDLE["model_name"],
        "threshold": BUNDLE["threshold"],
        "predictions": results,
    })


@app.route("/schema", methods=["GET"])
def schema():
    return jsonify({
        "model": BUNDLE["model_name"],
        "threshold": BUNDLE["threshold"],
        "feature_order": BUNDLE["feature_order"],
        "fields": BUNDLE["field_schema"],
    })


@app.route("/samples", methods=["GET"])
def samples():
    return jsonify({
        "count": len(SAMPLES),
        "note": "Held-out rows from the test split — never seen during training.",
        "samples": SAMPLES,
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": BUNDLE["model_name"],
        "threshold": BUNDLE["threshold"],
        "threshold_note": BUNDLE.get("threshold_note"),
        "n_features": len(BUNDLE["feature_order"]),
        "trained_at": BUNDLE["trained_at"],
        "sklearn_version": BUNDLE["sklearn_version"],
        "test_metrics": BUNDLE.get("test_metrics"),
    })


@app.errorhandler(404)
def not_found(_):
    return jsonify({
        "error": "Unknown endpoint.",
        "endpoints": ["/", "/predict-form", "/predict", "/predict/batch",
                      "/schema", "/samples", "/health"],
    }), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Deployed on localhost as the project requires. debug=False keeps the
    # bundle from being unpickled twice by the reloader.
    app.run(host="127.0.0.1", port=port, debug=False)
