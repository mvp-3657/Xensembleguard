"""
=============================================================
STEP 6 — EVALUATION & PAPER RESULTS GENERATOR
xEnsembleGuard: SHAP + Concept Drift Extension
=============================================================
Generates:
  • Classification report per dataset
  • Comparison table (vs EnsembleGuard baseline)
  • Per-attack-class F1 heatmap
  • ROC curves
  • All paper-ready plots
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_curve, auc,
    RocCurveDisplay
)
from sklearn.preprocessing import label_binarize

PLOTS_DIR   = "plots"
RESULTS_DIR = "results"
os.makedirs(PLOTS_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================
# HELPER — Full metrics dict
# =============================================================
def compute_metrics(y_true, y_pred, name: str) -> dict:
    return {
        "model"    : name,
        "accuracy" : accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall"   : recall_score   (y_true, y_pred, average="macro", zero_division=0),
        "f1"       : f1_score       (y_true, y_pred, average="macro", zero_division=0),
    }


# =============================================================
# CLASS: PaperEvaluator
# =============================================================
class PaperEvaluator:
    """
    Collects results from all models + your proposed novelties
    and generates all figures/tables needed for the paper.
    """

    def __init__(self, class_names: list, dataset_name: str = "UNSW-NB15"):
        self.class_names  = class_names
        self.dataset_name = dataset_name
        self.all_results  = []

    # ----------------------------------------------------------
    def add_result(self, name: str, y_true, y_pred):
        """Register a model's predictions for comparison."""
        r = compute_metrics(y_true, y_pred, name)
        self.all_results.append(r)
        print(f"  Added: {name:<25} | Acc: {r['accuracy']:.4f} | F1: {r['f1']:.4f}")
        return r

    # ----------------------------------------------------------
    def print_per_class_report(self, y_true, y_pred, model_name: str):
        """Detailed per-class breakdown — goes into paper tables."""
        print(f"\n{'='*60}")
        print(f"  Per-Class Report: {model_name} on {self.dataset_name}")
        print(f"{'='*60}")
        print(classification_report(y_true, y_pred,
                                    target_names=self.class_names,
                                    zero_division=0, digits=4))

    # ----------------------------------------------------------
    def plot_comparison_table(self, save: bool = True):
        """
        Publication-quality bar chart comparing all models.
        Your xEnsembleGuard should be visually the highest.
        """
        if not self.all_results:
            print("  No results added yet.")
            return

        df = pd.DataFrame(self.all_results)
        df = df.sort_values("f1", ascending=True)

        fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(df)*0.5)))
        metrics = [("accuracy", "Accuracy", "#2E75B6"),
                   ("f1",       "F1-Score", "#ED7D31")]

        for ax, (col, label, color) in zip(axes, metrics):
            # Highlight proposed model
            colors = []
            for model in df["model"]:
                if "xEnsembleGuard" in model or "Proposed" in model:
                    colors.append("#C00000")
                else:
                    colors.append(color)

            bars = ax.barh(df["model"], df[col], color=colors,
                           edgecolor="white", linewidth=0.5)
            ax.set_xlim(df[col].min() - 0.05, 1.01)
            ax.set_xlabel(label, fontsize=12)
            ax.set_title(f"{label} Comparison\n{self.dataset_name}",
                         fontsize=12, fontweight="bold")
            ax.grid(axis="x", alpha=0.3)

            for bar in bars:
                ax.text(bar.get_width() + 0.002,
                        bar.get_y() + bar.get_height()/2,
                        f"{bar.get_width():.4f}",
                        va="center", ha="left", fontsize=8)

        plt.suptitle("Model Performance Comparison — All Approaches",
                     fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()

        if save:
            path = f"{PLOTS_DIR}/final_comparison_{self.dataset_name}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  📊 Comparison chart saved → {path}")
        plt.close()

    # ----------------------------------------------------------
    def plot_per_class_f1_heatmap(self, models_dict: dict, save: bool = True):
        """
        Heatmap: rows = models, cols = attack classes, values = F1.
        Shows which model excels at detecting which attack type.
        """
        rows = []
        for model_name, (y_true, y_pred) in models_dict.items():
            f1s = f1_score(y_true, y_pred, average=None, zero_division=0)
            row = {"Model": model_name}
            for i, cls in enumerate(self.class_names):
                row[cls] = round(f1s[i], 4) if i < len(f1s) else 0.0
            rows.append(row)

        df = pd.DataFrame(rows).set_index("Model")

        fig, ax = plt.subplots(figsize=(max(10, len(self.class_names)*1.5),
                                        max(6, len(models_dict)*0.7)))
        sns.heatmap(df, annot=True, fmt=".3f", cmap="YlOrRd",
                    linewidths=0.5, linecolor="white",
                    vmin=0.7, vmax=1.0, ax=ax,
                    annot_kws={"size": 9})
        ax.set_title(f"Per-Class F1 Score Heatmap — {self.dataset_name}",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Attack Class", fontsize=11)
        ax.set_ylabel("Model",        fontsize=11)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        if save:
            path = f"{PLOTS_DIR}/f1_heatmap_{self.dataset_name}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  📊 F1 heatmap saved → {path}")
        plt.close()

    def plot_roc_curves(self, y_true, y_pred_proba: np.ndarray,
                        model_name: str = "xEnsembleGuard", save: bool = True):
        """
        ROC curves: binary (single curve) or multiclass One-vs-Rest.
        AUC values go into your paper's Table.
        """
        n_cls = len(self.class_names)
        fig, ax = plt.subplots(figsize=(9, 7))

        if n_cls == 2:
            # ── Binary classification: single ROC curve ──────────
            pos_proba = y_pred_proba[:, 1] if y_pred_proba.shape[1] > 1 else y_pred_proba[:, 0]
            fpr, tpr, _ = roc_curve(y_true, pos_proba)
            roc_auc = auc(fpr, tpr)
            mean_auc = roc_auc
            ax.plot(fpr, tpr, color="crimson", lw=2.2,
                    label=f"Attack vs Normal (AUC = {roc_auc:.4f})")
        else:
            # ── Multiclass: One-vs-Rest ───────────────────────────
            y_bin = label_binarize(y_true, classes=list(range(n_cls)))
            colors = plt.cm.Set1(np.linspace(0, 1, n_cls))
            mean_auc = 0
            n_plotted = 0
            for i, (cls_name, color) in enumerate(zip(self.class_names, colors)):
                if i >= y_pred_proba.shape[1] or i >= y_bin.shape[1]:
                    break
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                mean_auc += roc_auc
                n_plotted += 1
                ax.plot(fpr, tpr, color=color, lw=1.8,
                        label=f"{cls_name} (AUC = {roc_auc:.4f})")
            mean_auc /= max(n_plotted, 1)

        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate",  fontsize=12)
        ax.set_title(f"ROC Curves — {model_name}\n{self.dataset_name} | "
                     f"Mean AUC = {mean_auc:.4f}",
                     fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if save:
            path = f"{PLOTS_DIR}/roc_curves_{self.dataset_name}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  📊 ROC curves saved → {path}")
        plt.close()
        return mean_auc


    # ----------------------------------------------------------
    def generate_paper_table(self, save: bool = True) -> pd.DataFrame:
        """
        Produces the main results table for your paper.
        Columns: Model | Precision | Recall | F1 | Accuracy
        Rows   : All models in all_results
        """
        if not self.all_results:
            print("  No results to tabulate.")
            return pd.DataFrame()

        df = pd.DataFrame(self.all_results)
        for col in ["precision", "recall", "f1", "accuracy"]:
            df[col] = df[col].round(4)
        df.columns = ["Model", "Precision", "Recall", "F1-Score", "Accuracy"]
        df = df.sort_values("Accuracy", ascending=False)

        print(f"\n{'='*65}")
        print(f"  PAPER RESULTS TABLE — {self.dataset_name}")
        print(f"{'='*65}")
        print(df.to_string(index=False))

        if save:
            path = f"{RESULTS_DIR}/paper_table_{self.dataset_name}.csv"
            df.to_csv(path, index=False)
            print(f"\n  📄 Table saved → {path}")
        return df

    # ----------------------------------------------------------
    def plot_radar_chart(self, save: bool = True):
        """
        Radar chart showing Precision / Recall / F1 / Accuracy
        for all models simultaneously.
        """
        metrics   = ["Precision", "Recall", "F1-Score", "Accuracy"]
        n_metrics = len(metrics)
        angles    = np.linspace(0, 2*np.pi, n_metrics, endpoint=False).tolist()
        angles   += angles[:1]

        fig, ax = plt.subplots(figsize=(9, 9),
                               subplot_kw=dict(polar=True))

        colors = plt.cm.tab10(np.linspace(0, 1, len(self.all_results)))

        for result, color in zip(self.all_results, colors):
            values = [result["precision"], result["recall"],
                      result["f1"],        result["accuracy"]]
            values += values[:1]
            ax.plot(angles, values, lw=2, color=color,
                    label=result["model"])
            ax.fill(angles, values, alpha=0.07, color=color)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), metrics, fontsize=12)
        ax.set_ylim(0.85, 1.01)
        ax.set_title(f"Model Comparison Radar\n{self.dataset_name}",
                     fontsize=13, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save:
            path = f"{PLOTS_DIR}/radar_chart_{self.dataset_name}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  📊 Radar chart saved → {path}")
        plt.close()


# =============================================================
# QUICK TEST
# =============================================================
if __name__ == "__main__":
    np.random.seed(42)
    y_true = np.random.randint(0, 4, 1000)
    class_names = ["Normal", "DoS", "Probe", "R2L"]

    evaluator = PaperEvaluator(class_names, "UNSW-NB15")

    # Simulate 3 models
    for model_name, noise in [("LightGBM", 0.05), ("XGBoost", 0.04),
                               ("xEnsembleGuard (Proposed)", 0.02)]:
        y_pred = y_true.copy()
        flip = np.random.rand(1000) < noise
        y_pred[flip] = np.random.randint(0, 4, flip.sum())
        evaluator.add_result(model_name, y_true, y_pred)

    evaluator.plot_comparison_table()
    evaluator.generate_paper_table()
    evaluator.plot_radar_chart()
    print("\nEvaluation module test passed ✅")