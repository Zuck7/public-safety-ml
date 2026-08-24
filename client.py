"""
DELIVERABLE 5 — Test client for the KSI fatality prediction API.

Exercises the deployed Flask service the way the project spec asks: using the
held-out test data that was never seen during training. The rows come from
test_samples.csv, written by public_safety_ml.py from X_test — the grouped
train/test split kept whole crashes out of training, so these are genuinely
unseen records with known outcomes.

Usage:
    python app.py          # in one terminal
    python client.py       # in another
    python client.py --url http://127.0.0.1:5000
"""

import argparse
import json
import sys

import requests

DEFAULT_URL = "http://127.0.0.1:5000"
RULE = "=" * 78


def section(title):
    print(f"\n{RULE}\n{title}\n{RULE}")


def check_health(base):
    section("1. HEALTH CHECK  —  GET /health")
    r = requests.get(f"{base}/health", timeout=10)
    r.raise_for_status()
    info = r.json()
    print(f"  Status:      {info['status']}")
    print(f"  Model:       {info['model']}")
    print(f"  Threshold:   {info['threshold']:.4f}")
    print(f"  Features:    {info['n_features']}")
    print(f"  Trained at:  {info['trained_at']}  (scikit-learn {info['sklearn_version']})")
    return info


def fetch_samples(base):
    section("2. FETCH HELD-OUT TEST RECORDS  —  GET /samples")
    r = requests.get(f"{base}/samples", timeout=10)
    r.raise_for_status()
    payload = r.json()
    print(f"  Retrieved {payload['count']} records the model has never seen.")
    return payload["samples"]


def single_prediction(base, sample):
    section("3. SINGLE PREDICTION  —  POST /predict")
    print(f"  Sending record #{sample['id']} (recorded outcome: {sample['actual_label']})")
    r = requests.post(f"{base}/predict", json=sample["features"], timeout=30)
    r.raise_for_status()
    result = r.json()
    print(json.dumps({k: v for k, v in result.items() if k != "warnings"}, indent=2))
    if result.get("warnings"):
        print("  Warnings:")
        for w in result["warnings"]:
            print(f"    - {w}")
    return result


def batch_predictions(base, samples):
    section("4. BATCH PREDICTION  —  POST /predict/batch")
    records = [s["features"] for s in samples]
    r = requests.post(f"{base}/predict/batch", json={"records": records}, timeout=120)
    r.raise_for_status()
    payload = r.json()
    preds = payload["predictions"]

    print(f"  Scored {payload['count']} held-out records with {payload['model']} "
          f"at threshold {payload['threshold']:.4f}\n")
    print(f"  {'#':>3}  {'ACTUAL':<10}  {'PREDICTED':<10}  {'P(fatal)':>9}  RESULT")
    print(f"  {'-'*3}  {'-'*10}  {'-'*10}  {'-'*9}  {'-'*7}")

    tp = fp = tn = fn = 0
    for sample, pred in zip(samples, preds):
        actual, predicted = sample["actual"], pred["prediction"]
        hit = "correct" if actual == predicted else "MISS"
        if actual == 1 and predicted == 1:
            tp += 1
        elif actual == 1:
            fn += 1
        elif predicted == 1:
            fp += 1
        else:
            tn += 1
        print(f"  {sample['id']:>3}  {sample['actual_label']:<10}  {pred['label']:<10}  "
              f"{pred['probability_fatal']:>9.4f}  {hit}")

    total = tp + fp + tn + fn
    correct = tp + tn
    print(f"\n  Confusion on these {total} held-out rows:")
    print(f"    True Fatal  predicted Fatal      (TP): {tp}")
    print(f"    True Fatal  predicted Non-Fatal  (FN): {fn}   <- the costly error")
    print(f"    True Non-F. predicted Fatal      (FP): {fp}")
    print(f"    True Non-F. predicted Non-Fatal  (TN): {tn}")
    print(f"\n  Accuracy on this sample: {correct}/{total} = {correct / total * 100:.1f}%")
    if tp + fn:
        print(f"  Recall on Fatal cases:   {tp}/{tp + fn} = {tp / (tp + fn) * 100:.1f}%")
    if tp + fp:
        print(f"  Precision on Fatal:      {tp}/{tp + fp} = {tp / (tp + fp) * 100:.1f}%")
    print("\n  Note: this is a small, deliberately fatal-enriched slice for demonstration.")
    print("  The honest held-out metrics are the full-test-set numbers in /health.")
    return preds


def partial_payload(base, sample):
    section("5. PARTIAL INPUT  —  POST /predict with only a few fields")
    subset = {k: sample["features"][k] for k in
              ["INVTYPE", "SPEEDING", "ALCOHOL", "LIGHT", "PEDESTRIAN"]
              if k in sample["features"]}
    print(f"  Sending only: {json.dumps(subset)}")
    r = requests.post(f"{base}/predict", json=subset, timeout=30)
    r.raise_for_status()
    result = r.json()
    print(f"  -> {result['label']} (P(fatal) = {result['probability_fatal']:.4f})")
    print("  The service filled the remaining fields with training-set defaults and said so:")
    for w in result.get("warnings", []):
        print(f"    - {w}")


def error_handling(base):
    section("6. ERROR HANDLING  —  invalid input is rejected, not guessed at")

    print("  a) Non-numeric value in a numeric field:")
    r = requests.post(f"{base}/predict", json={"LATITUDE": "not-a-number"}, timeout=10)
    print(f"     HTTP {r.status_code} -> {r.json().get('error')}")

    print("\n  b) Unparseable value in a binary flag:")
    r = requests.post(f"{base}/predict", json={"SPEEDING": "maybe"}, timeout=10)
    print(f"     HTTP {r.status_code} -> {r.json().get('error')}")

    print("\n  c) Body that isn't JSON at all:")
    r = requests.post(f"{base}/predict", data="TIME=1430", timeout=10)
    print(f"     HTTP {r.status_code} -> {r.json().get('error')}")

    print("\n  d) Unknown endpoint:")
    r = requests.get(f"{base}/predikt", timeout=10)
    print(f"     HTTP {r.status_code} -> {r.json().get('error')}")


def main():
    parser = argparse.ArgumentParser(description="Test client for the KSI prediction API")
    parser.add_argument("--url", default=DEFAULT_URL, help="Base URL of the running API")
    args = parser.parse_args()
    base = args.url.rstrip("/")

    print(f"Testing the KSI fatality prediction API at {base}")

    try:
        check_health(base)
    except requests.exceptions.ConnectionError:
        print(f"\nERROR: nothing is listening on {base}.")
        print("Start the API first:  python app.py")
        sys.exit(1)

    samples = fetch_samples(base)
    if not samples:
        print("\nERROR: the API returned no held-out samples. "
              "Re-run public_safety_ml.py to regenerate test_samples.csv.")
        sys.exit(1)

    single_prediction(base, samples[0])
    batch_predictions(base, samples)
    partial_payload(base, samples[0])
    error_handling(base)

    section("ALL CLIENT TESTS COMPLETED")


if __name__ == "__main__":
    main()
