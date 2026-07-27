"""
xEnsembleGuard — Quick Results Viewer
Run: python show_results.py
Shows: Confusion Matrix + Accuracy + ROC-AUC in terminal + opens plots
"""

import numpy as np
import pandas as pd
import joblib, os, warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, accuracy_score,
                             precision_score, recall_score, f1_score)

# ── Load ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("   xEnsembleGuard — RESULTS VIEWER")
print("="*60)

from data_preprocessing import DataPreprocessor
data    = DataPreprocessor("UNSW_NB15").load()
X_test  = data["X_test"]
y_test  = data["y_test_bin"]

# Load all models
base_names = ["LightGBM","XGBoost","CatBoost","RandomForest"]
models = {}
for m in base_names + ["meta_model"]:
    path = f"models/{m}.pkl"
    if os.path.exists(path):
        models[m] = joblib.load(path)

# Get predictions
probas = [models[m].predict_proba(X_test) for m in base_names if m in models]
stacked   = np.hstack(probas)
meta_in   = np.hstack([stacked, X_test])
y_pred    = models["meta_model"].predict(meta_in)
y_proba   = models["meta_model"].predict_proba(meta_in)

# ── 1. ACCURACY METRICS ───────────────────────────────────────
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_proba[:, 1])

print("\n📊 ACCURACY METRICS")
print("─"*40)
print(f"  Accuracy  : {acc*100:.2f}%")
print(f"  Precision : {prec*100:.2f}%")
print(f"  Recall    : {rec*100:.2f}%")
print(f"  F1-Score  : {f1*100:.2f}%")
print(f"  AUC-ROC   : {auc*100:.2f}%")

# ── 2. ALL MODELS ACCURACY ────────────────────────────────────
print("\n🌲 BASE MODEL ACCURACY")
print("─"*40)
results = {}
for m in base_names:
    if m in models:
        p = models[m].predict(X_test)
        a = accuracy_score(y_test, p)
        results[m] = a
        bar = "█" * int(a * 40)
        print(f"  {m:<14} {a*100:.2f}%  {bar}")
bar = "█" * int(acc * 40)
print(f"  {'MetaModel':<14} {acc*100:.2f}%  {bar}  ← xEnsembleGuard")

# ── 3. CONFUSION MATRIX (ASCII) ───────────────────────────────
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print("\n🔲 CONFUSION MATRIX")
print("─"*40)
print("                 Predicted")
print("              Normal   Attack")
print(f"  Actual Normal  {tn:>6}   {fp:>6}  ← True Normal  = {tn+fp}")
print(f"  Actual Attack  {fn:>6}   {tp:>6}  ← True Attack  = {fn+tp}")
print()
print(f"  True Positives  (TP): {tp:>6}  — Attacks correctly caught")
print(f"  True Negatives  (TN): {tn:>6}  — Normal correctly passed")
print(f"  False Positives (FP): {fp:>6}  — Normal wrongly flagged")
print(f"  False Negatives (FN): {fn:>6}  — Attacks missed ⚠️")

# ── 4. PER CLASS REPORT ───────────────────────────────────────
print("\n📋 CLASSIFICATION REPORT")
print("─"*40)
print(classification_report(y_test, y_pred,
      target_names=["Normal","Attack"]))

# ── 5. AUC-ROC SUMMARY ───────────────────────────────────────
print("📈 AUC-ROC SUMMARY")
print("─"*40)
print(f"  xEnsembleGuard Meta  : {auc*100:.2f}%  ← Best 🏆")
for m in base_names:
    if m in models:
        p2 = models[m].predict_proba(X_test)
        a2 = roc_auc_score(y_test, p2[:,1])
        print(f"  {m:<20} : {a2*100:.2f}%")

print("\n" + "="*60)
print("  ✅ All results shown above!")
print("="*60)

# ── 6. OPEN PLOT FILES ────────────────────────────────────────
plots = [
    "plots/confusion_matrix.png",
    "plots/roc_curves_UNSW_NB15.png",
    "plots/shap_global_importance.png",
    "plots/concept_drift_analysis.png",
    "plots/zeroday_detection.png",
]
print("\n📂 Opening plots...")
for p in plots:
    if os.path.exists(p):
        os.system(f"open '{p}'")
        print(f"  ✅ Opened → {p}")
    else:
        print(f"  ⚠️  Not found → {p}")

print("\n✅ Done! Check your screen for the opened plots.\n")
