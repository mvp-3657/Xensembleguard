"""
=============================================================
STEP 3 — EnsembleGuard META-MODEL
xEnsembleGuard: SHAP + Concept Drift Extension
=============================================================
Meta-model: XGBoost (high-accuracy stacking combiner)
Input     : Stacked predictions from all base models
=============================================================
"""

import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from xgboost              import XGBClassifier
from sklearn.tree         import DecisionTreeClassifier, export_text
from sklearn.metrics      import (
    accuracy_score, precision_score,
    recall_score, f1_score, classification_report,
    confusion_matrix
)
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

MODELS_DIR  = "models"
RESULTS_DIR = "results"
PLOTS_DIR   = "plots"
for d in [MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)


# =============================================================
# CLASS: MetaModel
# =============================================================
class MetaModel:
    """
    Decision Tree meta-model that takes stacked base model
    predictions as input and produces final attack classification.
    Supports hyperparameter tuning, text rule export,
    and confusion matrix visualization.
    """

    def __init__(self, class_names: list):
        self.class_names = class_names
        self.model       = None
        self.best_params = {}
        self.metrics     = {}

    # ----------------------------------------------------------
    def tune_and_train(self, X_meta_train, y_train):
        """
        XGBoost meta-model with GridSearchCV tuning.
        """
        print(f"\n{'='*55}")
        print(f"  META-MODEL HYPERPARAMETER TUNING (XGBoost)")
        print(f"{'='*55}")

        n_classes = len(np.unique(y_train))
        param_grid = {
            "n_estimators"   : [400, 600],
            "learning_rate"  : [0.03, 0.05],
            "max_depth"      : [6, 8, 10],
            "subsample"      : [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }

        xgb = XGBClassifier(
            use_label_encoder = False,
            eval_metric       = "mlogloss",
            random_state      = 42,
            n_jobs            = -1,
            verbosity         = 0,
            objective         = "multi:softprob" if n_classes > 2 else "binary:logistic",
            num_class         = n_classes if n_classes > 2 else None,
        )
        grid = GridSearchCV(
            xgb, param_grid,
            cv=3, scoring="f1_macro",
            n_jobs=-1, verbose=1
        )
        grid.fit(X_meta_train, y_train)

        self.best_params = grid.best_params_
        print(f"\n  Best parameters: {self.best_params}")

        self.model = grid.best_estimator_
        joblib.dump(self.model, f"{MODELS_DIR}/meta_model.pkl")
        print(f"  ✅ XGBoost meta-model tuned and saved")
        return self

    # ----------------------------------------------------------
    def train_simple(self, X_meta_train, y_train, max_depth: int = 8):
        """
        XGBoost meta-model — strong combiner targeting ≥95% accuracy.
        Input: stacked base model predictions + original features.
        """
        n_classes = len(np.unique(y_train))
        is_binary = (n_classes == 2)
        print(f"\n  Training XGBoost meta-model ({n_classes} classes, {X_meta_train.shape[1]} features) ...")

        xgb_params = dict(
            n_estimators      = 600,
            learning_rate     = 0.05,
            max_depth         = max_depth,
            subsample         = 0.8,
            colsample_bytree  = 0.8,
            min_child_weight  = 1,
            gamma             = 0.0,
            reg_alpha         = 0.05,
            reg_lambda        = 1.0,
            use_label_encoder = False,
            eval_metric       = "logloss" if is_binary else "mlogloss",
            random_state      = 42,
            n_jobs            = -1,
            verbosity         = 0,
            objective         = "binary:logistic" if is_binary else "multi:softprob",
        )
        if not is_binary:
            xgb_params["num_class"] = n_classes

        self.model = XGBClassifier(**xgb_params)
        self.model.fit(
            X_meta_train, y_train,
            eval_set  = [(X_meta_train, y_train)],
            verbose   = False,
        )
        joblib.dump(self.model, f"{MODELS_DIR}/meta_model.pkl")
        print(f"  ✅ XGBoost meta-model trained")
        return self

    # ----------------------------------------------------------
    def evaluate(self, X_meta_test, y_test):
        """Run evaluation metrics and return results dict."""
        assert self.model is not None, "Train model first!"

        y_pred = self.model.predict(X_meta_test)

        acc = accuracy_score(y_test, y_pred)
        p   = precision_score(y_test, y_pred, average="macro", zero_division=0)
        r   = recall_score   (y_test, y_pred, average="macro", zero_division=0)
        f1  = f1_score       (y_test, y_pred, average="macro", zero_division=0)

        self.metrics = {
            "accuracy" : acc,
            "precision": p,
            "recall"   : r,
            "f1"       : f1,
            "y_pred"   : y_pred,
        }

        print(f"\n{'='*55}")
        print(f"  EnsembleGuard META-MODEL RESULTS")
        print(f"{'='*55}")
        print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
        print(f"  Precision : {p:.4f}")
        print(f"  Recall    : {r:.4f}")
        print(f"  F1-Score  : {f1:.4f}")
        print(f"\n  Per-class breakdown:")
        print(classification_report(y_test, y_pred,
                                    target_names=self.class_names,
                                    zero_division=0))
        return self.metrics

    # ----------------------------------------------------------
    def predict_proba(self, X_meta):
        """Return probability vector for SHAP / drift input."""
        return self.model.predict_proba(X_meta)

    # ----------------------------------------------------------
    def predict(self, X_meta):
        """Return hard class labels."""
        return self.model.predict(X_meta)

    # ----------------------------------------------------------
    def print_decision_rules(self, max_depth: int = 3):
        """Print human-readable decision rules (tree text)."""
        print(f"\n{'='*55}")
        print(f"  DECISION TREE RULES (depth ≤ {max_depth})")
        print(f"{'='*55}")
        # Build feature names for stacked inputs
        n_models     = self.model.n_features_in_
        feature_names = [f"Model_{i//len(self.class_names)}_Class_{i%len(self.class_names)}"
                         for i in range(n_models)]
        rules = export_text(self.model,
                            feature_names=feature_names,
                            max_depth=max_depth)
        print(rules)

    # ----------------------------------------------------------
    def plot_confusion_matrix(self, y_test, save: bool = True):
        """Plot and optionally save a confusion matrix heatmap."""
        y_pred = self.metrics.get("y_pred")
        if y_pred is None:
            print("  Run evaluate() first.")
            return

        cm = confusion_matrix(y_test, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        # Raw counts
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=self.class_names,
                    yticklabels=self.class_names, ax=axes[0])
        axes[0].set_title("Confusion Matrix — Raw Counts", fontsize=14, fontweight="bold")
        axes[0].set_ylabel("True Label", fontsize=12)
        axes[0].set_xlabel("Predicted Label", fontsize=12)

        # Normalised
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                    xticklabels=self.class_names,
                    yticklabels=self.class_names, ax=axes[1])
        axes[1].set_title("Confusion Matrix — Normalised", fontsize=14, fontweight="bold")
        axes[1].set_ylabel("True Label", fontsize=12)
        axes[1].set_xlabel("Predicted Label", fontsize=12)

        plt.tight_layout()
        if save:
            path = f"{PLOTS_DIR}/confusion_matrix.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  📊 Confusion matrix saved → {path}")
        plt.show()

    # ----------------------------------------------------------
    def plot_model_comparison(self, all_results: list, save: bool = True):
        """Bar chart comparing all models side by side."""
        names  = [r["model"]    for r in all_results]
        accs   = [r["accuracy"] for r in all_results]
        f1s    = [r["f1"]       for r in all_results]

        x  = np.arange(len(names))
        w  = 0.35

        fig, ax = plt.subplots(figsize=(14, 6))
        bars1 = ax.bar(x - w/2, accs, w, label="Accuracy",  color="#2E75B6", alpha=0.85)
        bars2 = ax.bar(x + w/2, f1s,  w, label="F1-Score",  color="#ED7D31", alpha=0.85)

        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Model Performance Comparison",
                     fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=10)
        ax.set_ylim(0.85, 1.01)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        if save:
            path = f"{PLOTS_DIR}/model_comparison.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  📊 Comparison chart saved → {path}")
        plt.show()


# =============================================================
# QUICK TEST
# =============================================================
if __name__ == "__main__":
    X = np.random.rand(1000, 28)   # 7 models × 4 classes
    y = np.random.randint(0, 4, 1000)
    class_names = ["Normal", "DoS", "Probe", "R2L"]

    meta = MetaModel(class_names)
    meta.train_simple(X[:800], y[:800], max_depth=6)
    metrics = meta.evaluate(X[800:], y[800:])
    print("\nMeta-model test passed ✅")