"""
=============================================================
STEP 4 — SHAP EXPLAINABILITY MODULE   ← NOVELTY 1
xEnsembleGuard: SHAP + Concept Drift Extension
=============================================================
Uses TreeSHAP for:
  • Global feature importance (across whole dataset)
  • Per-prediction explanation (why THIS sample = attack)
  • Attack-class attribution  (which features drive each class)
=============================================================
"""

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

PLOTS_DIR   = "plots"
RESULTS_DIR = "results"
os.makedirs(PLOTS_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================
# CLASS: SHAPExplainer
# =============================================================
class SHAPExplainer:
    """
    Wraps TreeSHAP to provide global and local explanations
    for the EnsembleGuard Decision Tree meta-model.

    Two levels of attribution:
      Level 1 → SHAP values on stacked meta-model inputs
                (= which base MODEL's output matters most)
      Level 2 → Back-project to original traffic features
                (= which TRAFFIC FEATURE matters most)
    """

    def __init__(self, meta_model, class_names: list, feature_names: list,
                 n_base_models: int = 4):
        """
        Args:
            meta_model    : Trained meta-model
            class_names   : List of class names (2 for binary, 10 for multiclass)
            feature_names : Original network traffic feature names
            n_base_models : Number of base models in stacking layer (default 4)
        """
        self.meta_model    = meta_model
        self.class_names   = class_names
        self.feature_names = feature_names
        self.n_base_models = n_base_models
        self.explainer     = None
        self.shap_values   = None
        self.X_background  = None

        # Real base model names (only the ones actually trained)
        self.base_model_names = [
            "LightGBM", "XGBoost", "CatBoost", "RandomForest"
        ][:n_base_models]

    # ----------------------------------------------------------
    def fit(self, X_meta_background: np.ndarray):
        """
        Initialise TreeSHAP explainer using a background dataset.
        X_meta_background: stacked prediction matrix (N, n_models * n_classes)
        """
        print(f"\n{'='*55}")
        print(f"  INITIALISING TREESHAP EXPLAINER")
        print(f"{'='*55}")
        self.X_background = X_meta_background
        self.explainer    = shap.TreeExplainer(
            self.meta_model,
            data              = shap.sample(X_meta_background, 100),
            feature_perturbation = "interventional",
        )
        print(f"  ✅ TreeSHAP explainer ready")
        print(f"     Background samples : 100")
        print(f"     Meta-input features: {X_meta_background.shape[1]}")
        return self

    # ----------------------------------------------------------
    def compute_global_shap(self, X_meta_test: np.ndarray, max_samples: int = 500):
        """
        Compute SHAP values for up to max_samples test instances.
        SHAP values shape: (n_classes, N, n_features)
        """
        print(f"\n  Computing global SHAP values ...")
        subset = X_meta_test[:max_samples]
        self.shap_values = self.explainer.shap_values(subset)
        # If binary, wrap in list for uniform handling
        if not isinstance(self.shap_values, list):
            self.shap_values = [self.shap_values]
        print(f"  ✅ SHAP values computed for {len(subset)} samples")
        print(f"     Shape per class: {self.shap_values[0].shape}")
        return self.shap_values

    # ----------------------------------------------------------
    def plot_global_importance(self, X_meta_test: np.ndarray,
                               top_n: int = 15, save: bool = True):
        """
        Bar plot: Top-N most important meta-features globally.
        """
        if self.shap_values is None:
            self.compute_global_shap(X_meta_test)

        sv = np.array(self.shap_values)

        print("Raw SHAP shape:", sv.shape)

        if sv.ndim == 4:
            # (1, samples, features, classes)
            mean_shap = np.abs(sv).mean(axis=(0, 1, 3))

        elif sv.ndim == 3:
            # (samples, features, classes)
            mean_shap = np.abs(sv).mean(axis=(0, 2))

        else:
            raise ValueError(f"Unexpected SHAP shape: {sv.shape}")

        print("mean_shap shape:", mean_shap.shape)

        # Build meta-feature names: ModelName_ClassName
        n_models = len(mean_shap) // len(self.class_names)
        model_names  = ["LightGBM", "XGBoost", "CatBoost", "RandomForest"][:n_models]
        meta_feat_names = []
        for m in model_names:
            for c in self.class_names:
                meta_feat_names.append(f"{m}→{c}")

        # Pad if needed
        while len(meta_feat_names) < len(mean_shap):
            meta_feat_names.append(f"Feature_{len(meta_feat_names)}")

        top_idx   = np.argsort(mean_shap)[-top_n:][::-1]
        print("top_idx:", top_idx)
        top_names = [meta_feat_names[i] for i in top_idx]
        top_vals  = mean_shap[top_idx]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.9, top_n))
        ax.barh(range(top_n), top_vals[::-1], color=colors[::-1])
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_names[::-1], fontsize=9)
        ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
        ax.set_title(f"Top {top_n} Most Important Meta-Features (Global)",
                     fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        if save:
            path = f"{PLOTS_DIR}/shap_global_importance.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  📊 Global SHAP importance saved → {path}")
        plt.close()

    # ----------------------------------------------------------
    def plot_shap_summary(self, X_meta_test: np.ndarray,
                          class_idx: int = 0, save: bool = True):
        """
        SHAP summary (beeswarm) plot for a specific attack class.
        """
        if self.shap_values is None:
            self.compute_global_shap(X_meta_test)

        subset = X_meta_test[:500]
        print("SHAP shape:", np.array(self.shap_values).shape)
        print("Class idx:", class_idx)
        sv_all = np.array(self.shap_values)

        # New SHAP format
        if sv_all.ndim == 4:
            sv = sv_all[0][:500, :, class_idx]

        # Old SHAP format
        else:
            sv = self.shap_values[class_idx][:500]

        # Build simple feature labels
        meta_labels = [f"F{i}" for i in range(sv.shape[1])]

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            sv, subset,
            feature_names = meta_labels,
            class_names   = self.class_names,
            show          = False,
            max_display   = 20,
        )
        plt.title(f"SHAP Summary — Class: {self.class_names[class_idx]}",
                  fontsize=13, fontweight="bold")
        plt.tight_layout()

        if save:
            path = f"{PLOTS_DIR}/shap_summary_class{class_idx}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  📊 SHAP summary saved → {path}")
        plt.close()

    # ----------------------------------------------------------
    def explain_single_prediction(self, x_single: np.ndarray,
                                  true_label: str = None, save: bool = True):
        """
        Explain ONE prediction: shows which base model outputs
        pushed the decision towards a particular attack class.

        Args:
            x_single : 1D array of shape (n_meta_features,)
            true_label: optional string for plot title
        """
        x = x_single.reshape(1, -1)
        sv = self.explainer.shap_values(x)
        if not isinstance(sv, list):
            sv = [sv]

        predicted_class_idx = int(self.meta_model.predict(x)[0])
        predicted_class     = self.class_names[predicted_class_idx]

        print(f"\n  {'─'*50}")
        print(f"  SINGLE PREDICTION EXPLANATION")
        print(f"  {'─'*50}")
        print(f"  Predicted class : {predicted_class}")
        if true_label:
            print(f"  True class      : {true_label}")

        # Handle both XGBoost 4D format and old list-per-class format
        sv_arr = np.array(sv)
        if sv_arr.ndim == 4:
            # XGBoost: (1, n_samples, n_features, n_classes)
            class_sv = sv_arr[0, 0, :, predicted_class_idx]
        elif sv_arr.ndim == 3 and sv_arr.shape[0] == len(self.class_names):
            # Old per-class list: (n_classes, n_samples, n_features)
            class_sv = sv_arr[predicted_class_idx, 0, :]
        else:
            # Fallback: take mean across classes
            class_sv = np.abs(sv_arr).mean(axis=-1).flatten()[:sv_arr.shape[-2] if sv_arr.ndim > 1 else len(sv_arr)]
        n_models = len(class_sv) // len(self.class_names)
        model_names = ["LightGBM", "XGBoost", "CatBoost",
                       "GBM", "Bagging", "LSTM", "GRU"][:n_models]

        meta_labels = []
        for m in model_names:
            for c in self.class_names:
                meta_labels.append(f"{m}→{c}")

        # Group SHAP by model
        model_contributions = {}
        for i, label in enumerate(meta_labels[:len(class_sv)]):
            model_name = label.split("→")[0]
            model_contributions[model_name] = \
                model_contributions.get(model_name, 0) + class_sv[i]

        print(f"\n  Model contributions to '{predicted_class}':")
        for model, contrib in sorted(model_contributions.items(),
                                     key=lambda x: abs(x[1]), reverse=True):
            bar = "█" * int(abs(contrib) * 30)
            sign = "+" if contrib > 0 else "-"
            print(f"    {model:12s} {sign}{abs(contrib):.4f}  {bar}")

        # Save waterfall-style bar chart
        fig, ax = plt.subplots(figsize=(9, 5))
        models  = list(model_contributions.keys())
        vals    = [model_contributions[m] for m in models]
        colors  = ["#2ecc71" if v > 0 else "#e74c3c" for v in vals]
        ax.barh(models, vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP Contribution", fontsize=11)
        ax.set_title(f"Prediction Explanation: '{predicted_class}'\n"
                     f"(True: {true_label or 'N/A'})",
                     fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        if save:
            path = f"{PLOTS_DIR}/shap_single_explanation.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  📊 Single explanation saved → {path}")
        plt.close()

        return model_contributions

    # ----------------------------------------------------------
    def compute_attack_feature_table(self, X_meta_test: np.ndarray):
        """
        Produce a summary DataFrame: for each class,
        which base model output contributes most.
        Goes directly into your paper as a table.
        """
        if self.shap_values is None:
            self.compute_global_shap(X_meta_test)

        sv_all   = np.array(self.shap_values)
        n_cls    = len(self.class_names)
        is_binary = (n_cls == 2)

        # Stacked features = n_base_models * n_cls columns (first block)
        # Original features = remaining columns
        feat_per_model = n_cls                          # proba columns per model
        n_stacked      = self.n_base_models * feat_per_model
        sv_axis0       = sv_all.shape[0]                # 1 for binary, n_cls for multiclass

        rows = []
        for class_idx, class_name in enumerate(self.class_names):
            safe_idx = min(class_idx, sv_axis0 - 1)

            # Extract raw SHAP array for this class
            if sv_all.ndim == 4:
                sv_raw = sv_all[0, :, :, min(class_idx, sv_all.shape[3] - 1)]
            elif sv_all.ndim == 3:
                sv_raw = sv_all[safe_idx]               # (n_samples, n_features)
            else:
                sv_raw = sv_all

            # ── Directional SHAP: binary needs separate logic ────
            if is_binary:
                if class_idx == 0:   # Normal class
                    # Features that push prediction TOWARD Normal
                    # = features with NEGATIVE SHAP (in attack direction)
                    mean_sv = np.where(sv_raw < 0, np.abs(sv_raw), 0).mean(axis=0)
                else:                # Attack class
                    # Features that push prediction TOWARD Attack
                    # = features with POSITIVE SHAP
                    mean_sv = np.where(sv_raw > 0, sv_raw, 0).mean(axis=0)
            else:
                # Multiclass: standard absolute mean
                mean_sv = np.abs(sv_raw).mean(axis=0)

            # ── Group by base model (stacked prediction block) ───
            model_importance = {}
            for m_idx, model in enumerate(self.base_model_names):
                start = m_idx * feat_per_model
                end   = start + feat_per_model
                if end <= len(mean_sv):
                    model_importance[model] = float(mean_sv[start:end].sum())
                else:
                    model_importance[model] = 0.0

            # ── Original traffic features block ──────────────────
            if n_stacked < len(mean_sv):
                orig_importance = float(mean_sv[n_stacked:].sum())
                model_importance["OriginalFeatures"] = orig_importance

            if not model_importance:
                continue

            top_model = max(model_importance, key=model_importance.get)
            top_val   = model_importance[top_model]

            rows.append({
                "Class"         : class_name,
                "Top Contributor": top_model,
                "SHAP Value"    : round(top_val, 4),
                **{m: round(v, 4) for m, v in model_importance.items()}
            })

        df = pd.DataFrame(rows)
        path = f"{RESULTS_DIR}/shap_attack_table.csv"
        df.to_csv(path, index=False)
        print(f"\n  ✅ SHAP attribution table:")
        print(df.to_string(index=False))
        print(f"\n  📄 Table saved → {path}")
        return df


# =============================================================
# QUICK TEST
# =============================================================
if __name__ == "__main__":
    from sklearn.tree import DecisionTreeClassifier

    np.random.seed(42)
    X = np.random.rand(500, 28)
    y = np.random.randint(0, 4, 500)
    class_names = ["Normal", "DoS", "Probe", "R2L"]

    dt = DecisionTreeClassifier(max_depth=6, random_state=42)
    dt.fit(X[:400], y[:400])

    explainer = SHAPExplainer(dt, class_names, feature_names=[])
    explainer.fit(X[:400])
    explainer.compute_global_shap(X[400:])
    explainer.plot_global_importance(X[400:])
    explainer.explain_single_prediction(X[400], true_label="DoS")
    explainer.compute_attack_feature_table(X[400:])
    print("\nSHAP module test passed ✅")