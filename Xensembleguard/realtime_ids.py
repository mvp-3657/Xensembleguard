"""
=============================================================
NOVELTY 4 — REAL-TIME INTRUSION DETECTION ENGINE
Drift-Aware Explainable Ensemble Learning for
Adaptive Network Intrusion Detection
=============================================================
Streams network traffic sample-by-sample, producing:
  - Live attack/normal decision per packet
  - Real-time SHAP explanation for every alert
  - Continuous ADWIN drift monitoring
  - Zero-day flagging via Isolation Forest
  - Live alert log with timestamps
=============================================================
Run: python realtime_ids.py
"""

import numpy as np
import pandas as pd
import joblib, os, time, threading, warnings
from datetime import datetime
from collections import deque
warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score
from river.drift import ADWIN
import shap

# ── Load all trained models ────────────────────────────────
print("\n" + "="*60)
print("  REAL-TIME IDS ENGINE — Loading Models...")
print("="*60)

MODELS_DIR = "models"
models = {}
for name, fname in [
    ("LightGBM",      "LightGBM.pkl"),
    ("XGBoost",       "XGBoost.pkl"),
    ("CatBoost",      "CatBoost.pkl"),
    ("RandomForest",  "RandomForest.pkl"),
    ("MetaModel",     "meta_model.pkl"),
    ("IsolationForest","isolation_forest.pkl"),
]:
    path = f"{MODELS_DIR}/{fname}"
    if os.path.exists(path):
        models[name] = joblib.load(path)
        print(f"  ✅ Loaded {name}")
    else:
        print(f"  ⚠️  {name} not found — run Main.py first")

print()


# ── Load test data as the live stream ─────────────────────
from data_preprocessing import DataPreprocessor
data   = DataPreprocessor("UNSW_NB15").load()
X_all  = data["X_test"]
y_all  = data["y_test_bin"]
feat_names = data["feature_names"]

# Build meta feature names for SHAP
base_names   = ["LightGBM","XGBoost","CatBoost","RandomForest"]
class_names  = ["Normal","Attack"]
meta_feat_names = [f"{m}→P({c})" for m in base_names
                   for c in class_names] + list(feat_names)[:42]

# SHAP explainer
print("  Setting up SHAP explainer...")
_bg_probas = [models[m].predict_proba(X_all[:100]) for m in base_names if m in models]
_bg_stacked = np.hstack(_bg_probas)
_bg_meta    = np.hstack([_bg_stacked, X_all[:100]])
explainer   = shap.TreeExplainer(models["MetaModel"], _bg_meta)
print("  ✅ SHAP explainer ready\n")


# ── Alert log ─────────────────────────────────────────────
alert_log  = []
adwin      = ADWIN(delta=0.002)
acc_window = deque(maxlen=50)
stats = {
    "total": 0, "attacks": 0, "normals": 0,
    "zeroday": 0, "correct": 0, "drift_events": 0
}


# ── Predict one sample ────────────────────────────────────
def predict_one(x, y_true):
    """Full real-time prediction pipeline for one sample."""
    X = x.reshape(1, -1)

    # Step 1 — Zero-Day check
    iso_label = models["IsolationForest"].predict(X)[0]
    is_zeroday = (iso_label == -1)

    # Step 2 — Ensemble prediction
    probas  = [models[m].predict_proba(X) for m in base_names if m in models]
    stacked = np.hstack(probas)
    meta_in = np.hstack([stacked, X])
    pred    = models["MetaModel"].predict(meta_in)[0]
    proba   = models["MetaModel"].predict_proba(meta_in)[0]
    confidence = float(max(proba)) * 100

    # Step 3 — SHAP (only for attacks or zero-day)
    shap_top = []
    if pred == 1 or is_zeroday:
        try:
            sv = explainer.shap_values(meta_in)
            if isinstance(sv, list):
                vals = sv[1][0]
            elif hasattr(sv, 'ndim') and sv.ndim == 3:
                vals = sv[0, :, -1] if sv.shape[0] == 1 else sv[-1, 0, :]
            else:
                vals = sv[0]
            top_idx  = np.argsort(np.abs(vals))[::-1][:3]
            shap_top = [(meta_feat_names[i] if i < len(meta_feat_names)
                         else f"feat_{i}", float(vals[i]))
                        for i in top_idx]
        except Exception:
            shap_top = []

    # Step 4 — ADWIN drift update
    correct  = int(pred == y_true)
    adwin.update(correct)
    drift    = adwin.drift_detected

    return {
        "pred":       int(pred),
        "label":      "Attack" if pred==1 else "Normal",
        "confidence": confidence,
        "is_zeroday": is_zeroday,
        "shap_top":   shap_top,
        "correct":    correct,
        "drift":      drift,
        "prob_attack": float(proba[1]),
    }


# ── Real-time stream engine ────────────────────────────────
def run_stream(n_samples=200, delay=0.3, verbose=True):
    """
    Streams n_samples from the test set with `delay` seconds
    between each, simulating live network traffic.
    """
    global stats

    print("="*60)
    print("  🚀 REAL-TIME IDS STREAM STARTED")
    print(f"  Streaming {n_samples} samples at {delay}s interval")
    print(f"  Press Ctrl+C to stop")
    print("="*60 + "\n")

    idx_list = np.random.choice(len(X_all), n_samples, replace=False)

    for i, idx in enumerate(idx_list):
        x      = X_all[idx]
        y_true = y_all[idx]

        result = predict_one(x, y_true)
        ts     = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        stats["total"]   += 1
        stats["correct"] += result["correct"]
        if result["pred"] == 1:
            stats["attacks"] += 1
        else:
            stats["normals"] += 1
        if result["is_zeroday"]:
            stats["zeroday"] += 1
        if result["drift"]:
            stats["drift_events"] += 1

        acc_window.append(result["correct"])
        live_acc = np.mean(acc_window) * 100

        # Log entry
        entry = {
            "timestamp":  ts,
            "sample_id":  int(idx),
            "prediction": result["label"],
            "confidence": result["confidence"],
            "zero_day":   result["is_zeroday"],
            "drift":      result["drift"],
            "shap_top":   result["shap_top"],
            "true_label": "Attack" if y_true==1 else "Normal",
            "correct":    result["correct"],
        }
        alert_log.append(entry)

        if verbose:
            # ── Print alert ───────────────────────────────────
            icon = ("⚠️ " if result["is_zeroday"] else
                    "🚨" if result["pred"]==1 else "✅")
            drift_flag = " | ⚡DRIFT!" if result["drift"] else ""
            print(f"[{ts}] #{i+1:>4} {icon} {result['label']:<8} "
                  f"({result['confidence']:>5.1f}%) "
                  f"| LiveAcc={live_acc:.1f}%{drift_flag}")

            # Show SHAP for attacks
            if result["shap_top"] and result["pred"]==1:
                for feat, val in result["shap_top"]:
                    direction = "→ ATTACK" if val > 0 else "→ NORMAL"
                    print(f"         📌 {feat[:30]:<30} {val:+.4f} {direction}")

            # Stats every 25 samples
            if (i+1) % 25 == 0:
                overall_acc = stats["correct"]/stats["total"]*100
                print(f"\n  ── Stats @ sample {i+1} ──────────────────")
                print(f"  Total       : {stats['total']}")
                print(f"  Attacks     : {stats['attacks']}")
                print(f"  Normal      : {stats['normals']}")
                print(f"  Zero-Day    : {stats['zeroday']}")
                print(f"  Drift Events: {stats['drift_events']}")
                print(f"  Accuracy    : {overall_acc:.2f}%\n")

        time.sleep(delay)

    # ── Final summary ─────────────────────────────────────
    print("\n" + "="*60)
    print("  ✅ STREAM COMPLETE — FINAL SUMMARY")
    print("="*60)
    overall_acc = stats["correct"]/stats["total"]*100
    print(f"  Total Samples Processed : {stats['total']}")
    print(f"  Attacks Detected        : {stats['attacks']}")
    print(f"  Normal Traffic          : {stats['normals']}")
    print(f"  Zero-Day Alerts         : {stats['zeroday']}")
    print(f"  Drift Events            : {stats['drift_events']}")
    print(f"  Overall Accuracy        : {overall_acc:.2f}%")

    # Save alert log
    df_log = pd.DataFrame(alert_log)
    df_log.to_csv("results/realtime_alert_log.csv", index=False)
    print(f"  📄 Alert log saved → results/realtime_alert_log.csv")
    print("="*60)
    return df_log


# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=100,
                    help="Number of samples to stream (default 100)")
    ap.add_argument("--delay",   type=float, default=0.2,
                    help="Seconds between each sample (default 0.2)")
    ap.add_argument("--fast",    action="store_true",
                    help="Fast mode: no delay between samples")
    args = ap.parse_args()

    delay = 0.0 if args.fast else args.delay
    run_stream(n_samples=args.samples, delay=delay)
