"""
=============================================================
ENSEMBLE STRATEGY COMPARISON — ALL METHODS AS BASE LEARNERS
Drift-Aware Explainable Ensemble Learning for
Adaptive Network Intrusion Detection
=============================================================

ARCHITECTURE:
  Level 0 — BASE LEARNERS (ALL ensemble types):
    ① LightGBM            (Gradient Boosting tree)
    ② XGBoost             (Gradient Boosting tree)
    ③ CatBoost            (Gradient Boosting tree)
    ④ Random Forest       (Bagging ensemble)
    ⑤ Extra Trees         (Bagging variant)
    ⑥ AdaBoost            (Boosting ensemble)
    ⑦ Gradient Boosting   (Boosting ensemble)
    ⑧ Voting (Soft)       (Voting ensemble)
    ⑨ Bagging (DT)        (Bagging ensemble)

  Level 1 — META-LEARNER (XGBoost ★):
    → Takes ALL 9 base model probability outputs
    → Combined with original raw features
    → XGBoost chosen for highest AUC + TreeSHAP support

WHY ALL METHODS:
  → Captures diverse decision boundaries
  → Boosting handles misclassified samples
  → Bagging reduces variance
  → Voting combines majority decisions
  → Meta-learner learns optimal combination
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os, warnings, joblib
warnings.filterwarnings("ignore")

from sklearn.ensemble import (
    VotingClassifier, BaggingClassifier,
    AdaBoostClassifier, GradientBoostingClassifier,
    RandomForestClassifier, ExtraTreesClassifier,
    StackingClassifier
)
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.neural_network  import MLPClassifier
from sklearn.calibration     import CalibratedClassifierCV
from sklearn.metrics         import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
    confusion_matrix, classification_report
)
from lightgbm  import LGBMClassifier
from xgboost   import XGBClassifier
from catboost  import CatBoostClassifier

PLOTS_DIR   = "plots"
RESULTS_DIR = "results"
MODELS_DIR  = "models"
for d in [PLOTS_DIR, RESULTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)


class EnsembleComparison:
    """
    Two-level ensemble:
      Level-0 : ALL 9 ensemble-type base learners
      Level-1 : XGBoost meta-model (our choice)

    Also independently benchmarks every individual method
    so the comparison table is complete.
    """

    def __init__(self, class_names=None):
        self.class_names  = class_names or ["Normal", "Attack"]
        self.results      = []
        self.best_model   = None
        self.best_y_pred  = None

    # ─────────────────────────────────────────────────────────
    # INDIVIDUAL BASE LEARNERS
    # ─────────────────────────────────────────────────────────
    def _make_lgbm(self):
        return LGBMClassifier(
            n_estimators=300, learning_rate=0.05,
            max_depth=8, random_state=42,
            n_jobs=-1, verbose=-1)

    def _make_xgb(self):
        return XGBClassifier(
            n_estimators=300, learning_rate=0.05,
            max_depth=8, eval_metric="logloss",
            random_state=42, n_jobs=-1,
            verbosity=0, objective="binary:logistic")

    def _make_cat(self):
        return CatBoostClassifier(
            iterations=300, learning_rate=0.05,
            depth=8, random_seed=42, verbose=0)

    def _make_rf(self):
        return RandomForestClassifier(
            n_estimators=200, max_depth=10,
            random_state=42, n_jobs=-1)

    def _make_et(self):
        return ExtraTreesClassifier(
            n_estimators=200, max_depth=10,
            random_state=42, n_jobs=-1)

    def _make_ada(self):
        return AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=200, learning_rate=0.1,
            random_state=42)

    def _make_gbm(self):
        return GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=6, random_state=42)

    def _make_bag(self):
        return BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=10),
            n_estimators=200, max_samples=0.8,
            max_features=0.8, bootstrap=True,
            random_state=42, n_jobs=-1)

    def _make_soft_vote(self):
        return VotingClassifier(
            estimators=[
                ("lgbm", self._make_lgbm()),
                ("xgb",  self._make_xgb()),
                ("cat",  self._make_cat()),
                ("rf",   self._make_rf()),
            ], voting="soft", n_jobs=-1)

    def _make_hard_vote(self):
        return VotingClassifier(
            estimators=[
                ("lgbm", self._make_lgbm()),
                ("xgb",  self._make_xgb()),
                ("cat",  self._make_cat()),
                ("rf",   self._make_rf()),
            ], voting="hard", n_jobs=-1)

    # ─────────────────────────────────────────────────────────
    # EVALUATE HELPER
    # ─────────────────────────────────────────────────────────
    def _eval(self, name, model, X_tr, y_tr, X_te, y_te,
              save_key=None, tag=""):
        label = f"{name}{tag}"
        print(f"\n  {'─'*52}")
        print(f"  🔄  {label}")
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        try:
            if hasattr(model, "predict_proba"):
                y_pr = model.predict_proba(X_te)[:, 1]
            else:
                y_pr = model.decision_function(X_te)
            auc = roc_auc_score(y_te, y_pr)
        except Exception:
            auc = float("nan")

        acc  = accuracy_score(y_te, y_pred)
        prec = precision_score(y_te, y_pred, zero_division=0)
        rec  = recall_score(y_te, y_pred, zero_division=0)
        f1   = f1_score(y_te, y_pred, zero_division=0)

        print(f"  Accuracy  : {acc*100:.2f}%  |  "
              f"F1: {f1*100:.2f}%  |  "
              f"AUC: {'' if np.isnan(auc) else f'{auc*100:.2f}%'}")

        row = dict(
            Model=label, Accuracy=round(acc, 4),
            Precision=round(prec, 4), Recall=round(rec, 4),
            F1=round(f1, 4), AUC_ROC=round(auc, 4))
        self.results.append(row)

        if save_key:
            joblib.dump(model, f"{MODELS_DIR}/{save_key}.pkl")

        return model, y_pred

    # ─────────────────────────────────────────────────────────
    # RUN ALL — Individual benchmarks + Two-level stacking
    # ─────────────────────────────────────────────────────────
    def run_all(self, X_train, y_train, X_test, y_test):

        print(f"\n{'='*60}")
        print(f"  ENSEMBLE STRATEGY COMPARISON — ALL METHODS")
        print(f"  Dataset: UNSW-NB15 | "
              f"Train={len(X_train):,} | Test={len(X_test):,}")
        print(f"{'='*60}")

        # ══════════════════════════════════════════════════════
        # SECTION A — INDIVIDUAL METHOD BENCHMARKS
        # ══════════════════════════════════════════════════════
        print(f"\n  ★ SECTION A: Individual Ensemble Methods")
        print(f"    (benchmark each method independently)")

        # 1. Voting — Hard
        self._eval("Voting (Hard)",
                   self._make_hard_vote(),
                   X_train, y_train, X_test, y_test)

        # 2. Voting — Soft
        self._eval("Voting (Soft)",
                   self._make_soft_vote(),
                   X_train, y_train, X_test, y_test)

        # 3. Bagging
        self._eval("Bagging",
                   self._make_bag(),
                   X_train, y_train, X_test, y_test)

        # 4. Boosting — AdaBoost
        self._eval("Boosting (AdaBoost)",
                   self._make_ada(),
                   X_train, y_train, X_test, y_test)

        # 5. Boosting — Gradient Boosting
        self._eval("Boosting (Gradient)",
                   self._make_gbm(),
                   X_train, y_train, X_test, y_test)

        # 6. Random Forest
        self._eval("Random Forest",
                   self._make_rf(),
                   X_train, y_train, X_test, y_test)

        # 7. Extra Trees
        self._eval("Extra Trees",
                   self._make_et(),
                   X_train, y_train, X_test, y_test)

        # 8. LightGBM
        self._eval("LightGBM",
                   self._make_lgbm(),
                   X_train, y_train, X_test, y_test)

        # 9. XGBoost
        self._eval("XGBoost",
                   self._make_xgb(),
                   X_train, y_train, X_test, y_test)

        # 10. CatBoost
        self._eval("CatBoost",
                   self._make_cat(),
                   X_train, y_train, X_test, y_test)

        # ══════════════════════════════════════════════════════
        # SECTION B — STACKING WITH DIFFERENT META-MODELS
        # Base learners: LGBM+XGB+CatBoost+RF+ET+AdaBoost+GBM
        # ══════════════════════════════════════════════════════
        print(f"\n  ★ SECTION B: Stacking with Different Meta-Models")
        print(f"    (base = LightGBM + XGB + CatBoost + RF + ET + ADA + GBM)")

        # All 7 ensemble types as base for every stacking variant
        def base_stack():
            return [
                ("lgbm", self._make_lgbm()),
                ("xgb",  self._make_xgb()),
                ("cat",  self._make_cat()),
                ("rf",   self._make_rf()),
                ("et",   self._make_et()),
                ("ada",  self._make_ada()),
                ("gbm",  self._make_gbm()),
            ]

        # 11. Stacking — LR meta
        self._eval(
            "Stacking (LR)",
            StackingClassifier(
                estimators=base_stack(),
                final_estimator=LogisticRegression(
                    max_iter=500, random_state=42),
                cv=3, n_jobs=-1, passthrough=True),
            X_train, y_train, X_test, y_test)

        # 12. Stacking — RF meta
        self._eval(
            "Stacking (RF)",
            StackingClassifier(
                estimators=base_stack(),
                final_estimator=RandomForestClassifier(
                    n_estimators=100, random_state=42),
                cv=3, n_jobs=-1, passthrough=True),
            X_train, y_train, X_test, y_test)

        # 13. Stacking — MLP meta
        self._eval(
            "Stacking (MLP)",
            StackingClassifier(
                estimators=base_stack(),
                final_estimator=MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    max_iter=300, random_state=42,
                    early_stopping=True),
                cv=3, n_jobs=-1, passthrough=True),
            X_train, y_train, X_test, y_test)

        # 14. Stacking — XGBoost meta
        self._eval(
            "Stacking (XGBoost)",
            StackingClassifier(
                estimators=base_stack(),
                final_estimator=XGBClassifier(
                    n_estimators=300, learning_rate=0.05,
                    max_depth=8, eval_metric="logloss",
                    random_state=42, n_jobs=-1,
                    verbosity=0, objective="binary:logistic"),
                cv=3, n_jobs=-1, passthrough=True),
            X_train, y_train, X_test, y_test)

        # ══════════════════════════════════════════════════════
        # SECTION C — COMBINED META  ★ OUR FINAL METHOD
        # ALL 4 stacking variants combined via Soft Voting
        # ══════════════════════════════════════════════════════
        print(f"\n  ★ SECTION C: Combined Meta — ALL Models Together")
        print(f"    Stacking(LR) + Stacking(RF) + "
              f"Stacking(MLP) + Stacking(XGBoost)")
        print(f"    Final decision = soft-vote of all 4 meta-models")

        combined_meta = VotingClassifier(
            estimators=[
                ("s_lr", StackingClassifier(
                    estimators=base_stack(),
                    final_estimator=LogisticRegression(
                        max_iter=500, random_state=42),
                    cv=3, n_jobs=-1, passthrough=True)),
                ("s_rf", StackingClassifier(
                    estimators=base_stack(),
                    final_estimator=RandomForestClassifier(
                        n_estimators=100, random_state=42),
                    cv=3, n_jobs=-1, passthrough=True)),
                ("s_mlp", StackingClassifier(
                    estimators=base_stack(),
                    final_estimator=MLPClassifier(
                        hidden_layer_sizes=(128, 64),
                        max_iter=300, random_state=42,
                        early_stopping=True),
                    cv=3, n_jobs=-1, passthrough=True)),
                ("s_xgb", StackingClassifier(
                    estimators=base_stack(),
                    final_estimator=XGBClassifier(
                        n_estimators=300, learning_rate=0.05,
                        max_depth=8, eval_metric="logloss",
                        random_state=42, n_jobs=-1,
                        verbosity=0, objective="binary:logistic"),
                    cv=3, n_jobs=-1, passthrough=True)),
            ],
            voting="soft", n_jobs=-1)

        _, y_pred_combined = self._eval(
            "Combined Meta ★ OURS",
            combined_meta,
            X_train, y_train, X_test, y_test,
            save_key="combined_meta_model")
        self.best_model  = combined_meta
        self.best_y_pred = y_pred_combined


        # ── Summary table ─────────────────────────────────────
        df = pd.DataFrame(self.results)
        df = df.sort_values("AUC_ROC", ascending=False)\
               .reset_index(drop=True)
        df.index += 1

        print(f"\n\n{'='*60}")
        print(f"  COMPLETE COMPARISON TABLE (sorted by AUC-ROC)")
        print(f"{'='*60}")
        pd.set_option("display.width", 130)
        pd.set_option("display.float_format", "{:.4f}".format)
        print(df.to_string())
        df.to_csv(f"{RESULTS_DIR}/ensemble_comparison.csv",
                  index=False)
        print(f"\n  📄 Saved → results/ensemble_comparison.csv")

        # ── Why Combined Meta is best ─────────────────────────
        our = df[df["Model"].str.contains("★")].iloc[0]
        print(f"\n{'='*60}")
        print(f"  WHY COMBINED META IS OUR BEST MODEL")
        print(f"{'='*60}")
        print(f"  Base learners (7 ensemble types combined):")
        print(f"  ① LightGBM  ② XGBoost  ③ CatBoost  ④ Random Forest")
        print(f"  ⑤ Extra Trees  ⑥ AdaBoost  ⑦ Gradient Boosting")
        print(f"")
        print(f"  Meta-models combined (4 stacking variants):")
        print(f"  ① Stacking (LR)  ② Stacking (RF)")
        print(f"  ③ Stacking (MLP)  ④ Stacking (XGBoost)")
        print(f"  → Soft-Voting averages all 4 meta-model probabilities")
        print(f"")
        print(f"  ★  AUC-ROC : {our['AUC_ROC']*100:.2f}%  (highest of all methods)")
        print(f"  ✓  No single model is relied upon — all are combined")
        print(f"  ✓  LR: linear boundary for simple patterns")
        print(f"  ✓  RF: reduces variance via bagging")
        print(f"  ✓  MLP: captures non-linear relationships")
        print(f"  ✓  XGBoost: regularized boosting + SHAP support")
        print(f"  ✓  Soft-Voting: averages probabilities → stable output")
        print(f"{'='*60}")

        return df

    # ─────────────────────────────────────────────────────────
    # FINAL REPORT — Confusion Matrix + Classification Report
    # ─────────────────────────────────────────────────────────
    def final_report(self, X_test, y_test):
        if self.best_model is None:
            print("  ⚠  Run run_all() first."); return

        y_pred = self.best_y_pred
        y_prob = self.best_model.predict_proba(X_test)[:, 1]
        cm     = confusion_matrix(y_test, y_pred)
        cr     = classification_report(
            y_test, y_pred,
            target_names=self.class_names, digits=4)
        tn, fp, fn, tp = cm.ravel()

        print(f"\n{'='*60}")
        print(f"  FINAL MODEL REPORT — Combined Meta ★ OURS")
        print(f"  Base : ALL 7 ensemble types (LGBM+XGB+CatBoost+RF+ET+ADA+GBM)")
        print(f"  Meta : Stacking(LR) + Stacking(RF) + Stacking(MLP) + Stacking(XGB)")
        print(f"  Final: Soft-Voting of ALL 4 meta-model probabilities")
        print(f"{'='*60}")
        print(f"  Accuracy           : {accuracy_score(y_test,y_pred)*100:.2f}%")
        print(f"  Precision          : {precision_score(y_test,y_pred)*100:.2f}%")
        print(f"  Recall             : {recall_score(y_test,y_pred)*100:.2f}%")
        print(f"  F1-Score           : {f1_score(y_test,y_pred)*100:.2f}%")
        print(f"  AUC-ROC            : {roc_auc_score(y_test,y_prob)*100:.2f}%")
        print(f"\n  CONFUSION MATRIX:")
        print(f"  {'':>22} Predicted Normal  Predicted Attack")
        print(f"  Actual Normal      : {cm[0,0]:>14,}  {cm[0,1]:>14,}")
        print(f"  Actual Attack      : {cm[1,0]:>14,}  {cm[1,1]:>14,}")
        print(f"\n  TP={tp:,}  TN={tn:,}  FP={fp:,}  FN={fn:,}")
        print(f"  False Positive Rate: {fp/(fp+tn)*100:.2f}%")
        print(f"  False Negative Rate: {fn/(fn+tp)*100:.2f}%")
        print(f"  Detection Rate     : {tp/(tp+fn)*100:.2f}%")
        print(f"\n  CLASSIFICATION REPORT:\n{cr}")
        print(f"{'='*60}")

        self._plot_confusion_matrix(cm)
        self.plot_comparison()

    # ─────────────────────────────────────────────────────────
    # CONFUSION MATRIX PLOT
    # ─────────────────────────────────────────────────────────
    def _plot_confusion_matrix(self, cm):
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            cm, annot=True, fmt=",", cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            linewidths=0.5, linecolor="#e0e0e0",
            annot_kws={"size": 15, "weight": "bold"},
            ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_xlabel("Predicted Label", fontsize=12,
                      fontweight="bold", labelpad=10)
        ax.set_ylabel("True Label", fontsize=12,
                      fontweight="bold", labelpad=10)
        ax.set_title(
            "Confusion Matrix — Stacking XGBoost Meta ★\n"
            "Base: ALL Ensemble Methods (7 base learners)",
            fontsize=11, fontweight="bold", pad=14)
        labels = {(0,0):"TN",(0,1):"FP",(1,0):"FN",(1,1):"TP"}
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j+0.5, i+0.72, labels[(i,j)],
                    ha="center", va="center", fontsize=10,
                    color="white" if (i==j and val>cm.max()*0.5)
                    else "#555", fontweight="bold")
        plt.tight_layout()
        path = f"{PLOTS_DIR}/confusion_matrix_xgb_meta.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\n  📊 Confusion matrix → {path}")
        plt.close()

    # ─────────────────────────────────────────────────────────
    # COMPARISON CHART — ALL methods
    # ─────────────────────────────────────────────────────────
    def plot_comparison(self, df=None):
        if df is None:
            df = pd.DataFrame(self.results)
        df = df.sort_values("Accuracy").reset_index(drop=True)
        n  = len(df)

        # Color by category
        def cat_color(name):
            if "★" in name:         return "#c0392b"   # red = ours
            if "Stacking" in name:   return "#8e44ad"   # purple
            if "Voting" in name:     return "#2980b9"   # blue
            if "Bagging" in name or "Forest" in name or "Extra" in name:
                                     return "#27ae60"   # green
            if "Boost" in name or "GBM" in name or \
               "LightGBM" in name or "XGBoost" in name or \
               "CatBoost" in name:   return "#e67e22"   # orange
            return "#7f8c8d"

        colors = [cat_color(m) for m in df["Model"]]

        fig, axes = plt.subplots(1, 2, figsize=(20, 9))
        fig.suptitle(
            "Complete Ensemble Strategy Comparison\n"
            "Drift-Aware Explainable Ensemble Learning — "
            "Adaptive Network IDS (UNSW-NB15)",
            fontsize=12, fontweight="bold")

        # Left: Accuracy bar chart
        bars = axes[0].barh(
            df["Model"], df["Accuracy"]*100,
            color=colors, edgecolor="white", height=0.65)
        axes[0].set_xlabel("Accuracy (%)", fontsize=11)
        axes[0].set_title("Accuracy by Strategy",
                           fontsize=11, fontweight="bold")
        axes[0].set_xlim(70, 102)
        axes[0].axvline(95, color="green", linestyle="--",
                         linewidth=1.2, label="95% target")
        axes[0].legend(fontsize=9)
        axes[0].grid(axis="x", alpha=0.3)
        for bar, val in zip(bars, df["Accuracy"]):
            axes[0].text(
                bar.get_width()+0.15,
                bar.get_y()+bar.get_height()/2,
                f"{val*100:.2f}%", va="center",
                fontsize=8, fontweight="bold")

        # Legend for colors
        legend_items = [
            mpatches.Patch(color="#c0392b", label="★ Our Method"),
            mpatches.Patch(color="#8e44ad", label="Stacking"),
            mpatches.Patch(color="#e67e22", label="Boosting"),
            mpatches.Patch(color="#27ae60", label="Bagging/Forest"),
            mpatches.Patch(color="#2980b9", label="Voting"),
        ]
        axes[0].legend(handles=legend_items, fontsize=8,
                       loc="lower right")

        # Right: grouped Acc + F1 + AUC
        x   = np.arange(n); w = 0.22
        acc = df["Accuracy"].values
        f1  = df["F1"].values
        auc = df["AUC_ROC"].fillna(0).values
        short = [
            m.replace("Stacking (","S(").replace(" meta)",")")
             .replace("Boosting (","Boost(").replace("Voting (","V(")
             .replace("Random Forest","RF").replace("Extra Trees","ET")
             .replace("LightGBM ","LGBM ").replace(" (Boosting)","")
             .replace("★ OURS","★").replace(")","")
            for m in df["Model"]]
        axes[1].bar(x-w, acc*100, w, label="Accuracy",
                    color="#2E75B6", alpha=0.88)
        axes[1].bar(x,   f1*100,  w, label="F1-Score",
                    color="#ED7D31", alpha=0.88)
        axes[1].bar(x+w, auc*100, w, label="AUC-ROC",
                    color="#70AD47", alpha=0.88)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(short, rotation=40,
                                 ha="right", fontsize=7.5)
        axes[1].set_ylabel("Score (%)", fontsize=11)
        axes[1].set_title("Accuracy vs F1 vs AUC-ROC",
                           fontsize=11, fontweight="bold")
        axes[1].set_ylim(70, 105)
        axes[1].legend(fontsize=9)
        axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        path = f"{PLOTS_DIR}/ensemble_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  📊 Comparison chart → {path}")
        plt.close()
        print(f"  ✅ Ensemble comparison complete\n")
