"""
DELIVERABLE 5 — Generates the Postman collection for the KSI prediction API.

The request bodies are built from the exported bundle and the held-out test
rows rather than typed by hand, so the collection can never drift out of sync
with the model's actual input contract: if feature selection changes and the
model is retrained, re-running this script regenerates valid requests.

Usage:
    python make_postman_collection.py     ->  writes postman_collection.json
"""

import json
import os

import model_service as svc

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(PROJECT_DIR, "postman_collection.json")
BASE_URL = "{{baseUrl}}"


def url(path):
    segments = [s for s in path.split("/") if s]
    return {"raw": f"{BASE_URL}/{'/'.join(segments)}", "host": [BASE_URL], "path": segments}


def get(name, path, description, tests):
    return {
        "name": name,
        "event": [{"listen": "test", "script": {"type": "text/javascript", "exec": tests}}],
        "request": {"method": "GET", "header": [], "url": url(path), "description": description},
    }


def post(name, path, body, description, tests):
    return {
        "name": name,
        "event": [{"listen": "test", "script": {"type": "text/javascript", "exec": tests}}],
        "request": {
            "method": "POST",
            "header": [{"key": "Content-Type", "value": "application/json"}],
            "body": {"mode": "raw", "raw": json.dumps(body, indent=2),
                     "options": {"raw": {"language": "json"}}},
            "url": url(path),
            "description": description,
        },
    }


def main():
    bundle = svc.load_bundle()
    samples = svc.samples_as_records(bundle)
    if not samples:
        raise SystemExit("No test samples found — run public_safety_ml.py first.")

    fatal = next((s for s in samples if s["actual"] == 1), samples[0])
    nonfatal = next((s for s in samples if s["actual"] == 0), samples[-1])

    ok = ['pm.test("status is 200", () => pm.response.to.have.status(200));']
    has_pred = ok + [
        'const body = pm.response.json();',
        'pm.test("returns a prediction", () => pm.expect(body).to.have.property("prediction"));',
        'pm.test("returns a probability", () => pm.expect(body.probability_fatal).to.be.a("number"));',
        'console.log(`${body.label} — P(fatal)=${body.probability_fatal}`);',
    ]

    partial = {k: fatal["features"][k] for k in
               ["INVTYPE", "SPEEDING", "ALCOHOL", "LIGHT", "PEDESTRIAN"]
               if k in fatal["features"]}

    collection = {
        "info": {
            "name": "KSI Collision Fatality API",
            "description": (
                "Deliverable 5 — client tests for the Flask analytics API serving the "
                f"{bundle['model_name']} model at threshold {bundle['threshold']:.4f}.\n\n"
                "Start the API with `python app.py`, then run this collection.\n\n"
                "The /predict bodies are real held-out test records that were never "
                "used to train the model; each request's description gives the "
                "recorded outcome so the prediction can be checked against it."
            ),
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
        },
        "variable": [{"key": "baseUrl", "value": "http://127.0.0.1:5000", "type": "string"}],
        "item": [
            get("1. Health check", "/health",
                "Confirms the API is up and reports which model and threshold are loaded.",
                ok + ['pm.test("model is loaded", () => pm.expect(pm.response.json().status).to.eql("ok"));']),

            get("2. Field schema", "/schema",
                f"Lists all {len(bundle['feature_order'])} features the model accepts, "
                "with valid categorical options and numeric ranges.",
                ok + [f'pm.test("all features described", () => '
                      f'pm.expect(pm.response.json().feature_order.length).to.eql({len(bundle["feature_order"])}));']),

            get("3. Held-out samples", "/samples",
                "Returns test-split records the model never trained on, with their true outcomes.",
                ok + ['pm.test("samples returned", () => pm.expect(pm.response.json().count).to.be.above(0));']),

            post("4. Predict — record with a FATAL outcome", "/predict", fatal["features"],
                 f"Held-out record #{fatal['id']}. Recorded outcome: {fatal['actual_label']}.",
                 has_pred),

            post("5. Predict — record with a NON-FATAL outcome", "/predict", nonfatal["features"],
                 f"Held-out record #{nonfatal['id']}. Recorded outcome: {nonfatal['actual_label']}.",
                 has_pred),

            post("6. Predict — partial input", "/predict", partial,
                 "Only a handful of fields are supplied. The API fills the rest with "
                 "training-set defaults and lists what it defaulted in `warnings`.",
                 has_pred + ['pm.test("warns about defaults", () => '
                             'pm.expect(pm.response.json().warnings).to.be.an("array"));']),

            post("7. Predict — batch", "/predict/batch",
                 {"records": [s["features"] for s in samples[:5]]},
                 "Scores five held-out records in one request.",
                 ok + ['const body = pm.response.json();',
                       'pm.test("five predictions", () => pm.expect(body.count).to.eql(5));']),

            post("8. Predict — invalid input is rejected", "/predict",
                 {"LATITUDE": "not-a-number"},
                 "Bad input must produce a clear 400, not a stack trace or a silent guess.",
                 ['pm.test("status is 400", () => pm.response.to.have.status(400));',
                  'pm.test("explains the problem", () => '
                  'pm.expect(pm.response.json().error).to.be.a("string"));']),
        ],
    }

    with open(OUT_PATH, "w") as f:
        json.dump(collection, f, indent=2)

    print(f"Wrote {OUT_PATH}")
    print(f"  {len(collection['item'])} requests, built against {bundle['model_name']} "
          f"@ threshold {bundle['threshold']:.4f}")
    print(f"  Fatal example: held-out record #{fatal['id']}")
    print(f"  Non-fatal example: held-out record #{nonfatal['id']}")
    print("\nImport into Postman: File > Import > postman_collection.json")


if __name__ == "__main__":
    main()
