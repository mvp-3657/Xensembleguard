"""
=============================================================
NOVELTY 3 — ZERO-DAY ATTACK DETECTION
xEnsembleGuard: Isolation Forest Anomaly Detection Layer
=============================================================
Detects completely NEW attacks that the classifier has
never seen during training (zero-day / novel attacks).

Workflow:
  1. Train Isolation Forest ONLY on normal traffic
  2. At inference: check if sample is anomalous
     - Known traffic  → pass to Ensemble IDS
     - Unknown/Novel  → flag as Zero-Day Alert
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_curve, auc)
import joblib

PLOTS_DIR   = "plots"
RESULTS_DIR = "results"
MODELS_DIR  = "models"
os.makedirs(PLOTS_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)


class AnomalyDetector:
    """
    Isolation Forest-based Zero-Day Attack Detector.
    Trained ONLY on normal traffic. Flags anything that
    deviates from normal as a potential zero-day attack.
    """

    def __init__(self, contamination=0.05, n_estimators=200, random_state=42):
        self.contamination = contamination
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        self.anomaly_scores = None

    def fit(self, X_train, y_train):
        """Train ONLY on normal (label=0) samples."""
        X_normal = X_train[y_train == 0]
        print(f"\n{'='*55}")
        print(f"  NOVELTY 3 — ZERO-DAY DETECTION (Isolation Forest)")
        print(f"{'='*55}")
        print(f"  Training on normal traffic only...")
        print(f"  Normal samples used    : {len(X_normal):,}")
        print(f"  Contamination rate     : {self.contamination}")
        self.model.fit(X_normal)
        joblib.dump(self.model, f"{MODELS_DIR}/isolation_forest.pkl")
        print(f"  ✅ Isolation Forest trained & saved")
        return self

    def detect(self, X):
        """Returns labels (+1=normal, -1=anomaly) and scores."""
        labels = self.model.predict(X)
        scores = self.model.decision_function(X)
        self.anomaly_scores = scores
        return labels, scores

    def evaluate(self, X_test, y_test):
        """Evaluate zero-day detection performance."""
        labels, scores = self.detect(X_test)
        y_pred_iso = np.where(labels == -1, 1, 0)

        total       = len(y_test)
        n_attacks   = int(y_test.sum())
        n_normal    = total - n_attacks
        n_flagged   = int(y_pred_iso.sum())
        n_correct   = int(((y_test == 1) & (y_pred_iso == 1)).sum())
        n_missed    = int(((y_test == 1) & (y_pred_iso == 0)).sum())
        n_false_pos = int(((y_test == 0) & (y_pred_iso == 1)).sum())

        precision = precision_score(y_test, y_pred_iso, zero_division=0)
        recall    = recall_score(y_test, y_pred_iso, zero_division=0)
        f1        = f1_score(y_test, y_pred_iso, zero_division=0)

        print(f"\n  Total test samples     : {total:,}")
        print(f"  True attacks           : {n_attacks:,}")
        print(f"  True normal            : {n_normal:,}")
        print(f"  Flagged as anomaly     : {n_flagged:,}")
        print(f"  Correctly caught (TP)  : {n_correct:,}")
        print(f"  Missed attacks (FN)    : {n_missed:,}")
        print(f"  False alarms (FP)      : {n_false_pos:,}")
        print(f"  ─────────────────────────────────────")
        print(f"  Precision              : {precision*100:.2f}%")
        print(f"  Recall (Detection Rate): {recall*100:.2f}%")
        print(f"  F1-Score               : {f1*100:.2f}%")

        results = {
            "total_samples": total, "true_attacks": n_attacks,
            "flagged": n_flagged, "true_positives": n_correct,
            "false_negatives": n_missed, "false_positives": n_false_pos,
            "precision": round(precision, 4),
            "recall": round(recall, 4), "f1": round(f1, 4),
        }
        pd.DataFrame([results]).to_csv(
            f"{RESULTS_DIR}/zeroday_results.csv", index=False)
        return y_pred_iso, scores, results

    def simulate_zeroday(self, X_test, y_test_multi, class_names, holdout_classes=None):
        """Simulate zero-day: treat rarest classes as unseen attacks."""
        if holdout_classes is None:
            unique, counts = np.unique(y_test_multi, return_counts=True)
            holdout_classes = unique[np.argsort(counts)][:2].tolist()

        holdout_names = [class_names[i] for i in holdout_classes if i < len(class_names)]
        print(f"\n  Zero-Day Simulation — treating as unseen: {holdout_names}")

        zeroday_mask = np.isin(y_test_multi, holdout_classes)
        X_zeroday = X_test[zeroday_mask]

        if len(X_zeroday) == 0:
            print("  No zero-day samples found.")
            return {}

        labels, _ = self.detect(X_zeroday)
        n_detected = int((labels == -1).sum())
        rate = n_detected / len(X_zeroday)

        print(f"  Zero-day samples       : {len(X_zeroday)}")
        print(f"  Detected as anomaly    : {n_detected}")
        print(f"  Zero-Day Detection Rate: {rate*100:.2f}%")

        return {"classes": holdout_names, "samples": len(X_zeroday),
                "detected": n_detected, "rate": round(rate, 4)}

    def plot_anomaly_scores(self, scores, y_test, save=True):
        """Plot anomaly score distribution: normal vs attack."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        normal_scores = scores[y_test == 0]
        attack_scores = scores[y_test == 1]

        ax1.hist(normal_scores, bins=60, alpha=0.7, color="#2E75B6",
                 label=f"Normal Traffic (n={len(normal_scores):,})", density=True)
        ax1.hist(attack_scores, bins=60, alpha=0.7, color="#e74c3c",
                 label=f"Attack Traffic (n={len(attack_scores):,})", density=True)
        ax1.axvline(0, color="black", linestyle="--", lw=1.5,
                    label="Decision Boundary")
        ax1.set_xlabel("Anomaly Score", fontsize=12)
        ax1.set_ylabel("Density", fontsize=12)
        ax1.set_title("Isolation Forest — Anomaly Score Distribution", fontsize=12, fontweight="bold")
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)

        ax2.axis("off")
        txt = (
            "  Zero-Day Detection Pipeline\n"
            "  ───────────────────────────\n\n"
            "  Network Traffic\n"
            "        ↓\n"
            "  Isolation Forest\n"
            "  (trained on normal only)\n"
            "        ↓\n"
            "   Anomalous?\n"
            "   /         \\\n"
            "  NO          YES\n"
            "  ↓            ↓\n"
            "Ensemble    Zero-Day\n"
            "  IDS         Alert\n"
            "  ↓\n"
            "SHAP + ADWIN"
        )
        ax2.text(0.1, 0.95, txt, transform=ax2.transAxes, fontsize=11,
                 verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="#f0f4ff", alpha=0.8))

        plt.suptitle("xEnsembleGuard — Novelty 3: Zero-Day Attack Detection",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        if save:
            path = f"{PLOTS_DIR}/zeroday_detection.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  📊 Zero-day plot saved → {path}")
        plt.close()

    def plot_roc_zeroday(self, scores, y_test, save=True):
        """ROC curve for anomaly detection."""
        fpr, tpr, _ = roc_curve(y_test, -scores)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, color="#e74c3c", lw=2.2,
                label=f"Isolation Forest (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curve — Zero-Day Detection\nIsolation Forest",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        if save:
            path = f"{PLOTS_DIR}/zeroday_roc.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  📊 Zero-day ROC saved → {path}")
        plt.close()
        return roc_auc
