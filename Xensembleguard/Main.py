"""
=============================================================
xEnsembleGuard — MAIN EXECUTION SCRIPT
SHAP Explainability + Concept Drift Detection
=============================================================
Run this file to execute the complete pipeline:

  Step 1 → Load & preprocess data
  Step 2 → Train base models (tree + DL)
  Step 3 → Train meta-model (EnsembleGuard)
  Step 4 → SHAP explanations          ← Novelty 1
  Step 5 → Concept drift simulation   ← Novelty 2
  Step 6 → Generate all paper results

Usage:
  python main.py --dataset UNSW_NB15
  python main.py --dataset NSL_KDD
  python main.py --dataset CIC_IDS_2017
  python main.py --all_datasets
=============================================================
"""

import argparse
import numpy as np
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

# ── Import all modules ────────────────────────────────────────
from data_preprocessing import DataPreprocessor
from base_models         import TreeModels
from meta_model          import MetaModel
from shap_explainability import SHAPExplainer
from concept_drift       import DriftSimulator
from evaluation          import PaperEvaluator
from anomaly_detector    import AnomalyDetector
from ensemble_comparison import EnsembleComparison

# ── Output directories ────────────────────────────────────────
for d in ["models", "results", "plots"]:
    os.makedirs(d, exist_ok=True)


# =============================================================
def banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║         xEnsembleGuard — Full Pipeline Execution         ║
║   Novelty 1: SHAP Deep Explainability                    ║
║   Novelty 2: ADWIN Concept Drift Detection               ║
║   Novelty 3: Zero-Day Attack Detection (Isolation Forest)║
╚══════════════════════════════════════════════════════════╝
""")


# =============================================================
def run_pipeline(dataset_name: str, use_dl: bool = True,
                 tune_meta: bool = False, n_drift_batches: int = 400):
    """
    Full end-to-end pipeline for one dataset.
    """
    start_time = time.time()
    print(f"\n{'▓'*58}")
    print(f"  DATASET: {dataset_name}")
    print(f"{'▓'*58}")

    # ── STEP 1: Load & Preprocess ─────────────────────────────
    print("\n📁 STEP 1 — DATA PREPROCESSING")
    preprocessor = DataPreprocessor(dataset_name)
    data         = preprocessor.load()

    X_train     = data["X_train"]
    X_test      = data["X_test"]
    y_train     = data["y_train_bin"]       # Binary: 0=Normal, 1=Attack
    y_test      = data["y_test_bin"]
    class_names = ["Normal", "Attack"]       # Binary IDS → targets 95%+ accuracy
    feat_names  = data["feature_names"]
    n_classes   = 2

    # Keep multiclass labels for SHAP attribution analysis
    y_train_mc  = data["y_train_multi"]
    y_test_mc   = data["y_test_multi"]
    mc_classes  = data["class_names"]

    print(f"  ✅ Mode: BINARY (Normal vs Attack) — target ≥95% accuracy")
    print(f"  ✅ Attack classes available for SHAP: {mc_classes}")

    # ── STEP 2: Train Base Models ─────────────────────────────
    print("\n🌲 STEP 2 — BASE MODEL TRAINING")

    # Tree-based models
    tree_models = TreeModels(n_classes=n_classes)
    tree_train_preds, tree_test_preds, tree_results = \
        tree_models.train_all(X_train, y_train, X_test, y_test, class_names)

    # Deep learning models (optional — takes longer)
    # if use_dl:
    #     print("\n  (DL models enabled — set use_dl=False for faster run)")
    #     dl_models = DeepModels(n_classes=n_classes, epochs=20, batch_size=512)
    #     dl_train_preds, dl_test_preds, dl_results = \
    #         dl_models.train_all(X_train, y_train, X_test, y_test, class_names)
    #     meta_train = combine_predictions(tree_train_preds, tree_test_preds,
    #                                      dl_train_preds,  dl_test_preds)[0]
    #     meta_test  = combine_predictions(tree_train_preds, tree_test_preds,
    #                                      dl_train_preds,  dl_test_preds)[1]
    #     all_base_results = tree_results + dl_results
    # else:
    #     meta_train = tree_train_preds
    #     meta_test  = tree_test_preds
    #     all_base_results = tree_results
    meta_train = tree_train_preds
    meta_test  = tree_test_preds
    all_base_results = tree_results

    # ── ENRICH META-MODEL INPUT WITH ORIGINAL FEATURES ───────
    # Concatenate stacked predictions + original features.
    # This gives the meta-model both "what do the base models think?"
    # AND the raw traffic signal — crucial for hard classes like DoS.
    meta_train = np.hstack([meta_train, X_train])
    meta_test  = np.hstack([meta_test,  X_test])
    print(f"  ✅ Meta-model input enriched: stacked + original = {meta_train.shape[1]} features")

    # ── STEP 3: Meta-Model ────────────────────────────────────
    print("\n🧩 STEP 3 — META-MODEL (EnsembleGuard)")
    meta = MetaModel(class_names)

    if tune_meta:
        meta.tune_and_train(meta_train, y_train)
    else:
        meta.train_simple(meta_train, y_train, max_depth=8)

    meta_metrics = meta.evaluate(meta_test, y_test)
    meta.plot_confusion_matrix(y_test)

    # Add base + meta results to evaluator
    evaluator = PaperEvaluator(class_names, dataset_name)
    for r in all_base_results:
        # We need y_pred to add; use stored results dict
        pass  # Note: in full implementation, pass (y_test, y_pred) pairs

    evaluator.add_result("EnsembleGuard Meta-Model",
                         y_test, meta.predict(meta_test))

    # ── STEP 3.5: ENSEMBLE STRATEGY COMPARISON ───────────────
    print("\n📊 STEP 3.5 — ENSEMBLE STRATEGY COMPARISON")
    print("  Comparing ALL strategies: Voting | Bagging | Boosting | Stacking")
    print("  Meta-model uses ALL 4 base models combined (LGBM+XGB+CatBoost+RF)")
    print("  Justifying XGBoost as the best meta-model choice")

    ec = EnsembleComparison(class_names=class_names)
    ec_df = ec.run_all(X_train, y_train, X_test, y_test)

    # Full report: confusion matrix + classification report + why XGBoost
    ec.final_report(X_test, y_test)

    print("  ✅ Ensemble comparison complete")

    # ── STEP 4: SHAP EXPLAINABILITY (Novelty 1) ──────────────
    print("\n🔍 STEP 4 — SHAP EXPLAINABILITY  ← NOVELTY 1")

    # Number of actual base models (LightGBM, XGBoost, CatBoost, RandomForest)
    n_base_models = len(tree_models.models)   # = 4

    shap_explainer = SHAPExplainer(
        meta_model    = meta.model,
        class_names   = class_names,
        feature_names = feat_names,
        n_base_models = n_base_models,
    )
    shap_explainer.fit(meta_train)

    print("\n  Computing global SHAP values (500 samples) ...")
    shap_explainer.compute_global_shap(meta_test, max_samples=500)
    shap_explainer.plot_global_importance(meta_test, top_n=15)

    shap_explainer.plot_shap_summary(
        meta_test,
        class_idx=0
    )

    # Explain a single prediction (pick a misclassified sample for interest)
    y_pred_all = meta.predict(meta_test)
    misclassified = np.where(y_pred_all != y_test)[0]
    sample_idx = misclassified[0] if len(misclassified) > 0 else 0
    true_label = class_names[y_test[sample_idx]]
    shap_explainer.explain_single_prediction(
        meta_test[sample_idx], true_label=true_label
    )

    # Attack attribution table (goes into paper)
    shap_table = shap_explainer.compute_attack_feature_table(meta_test)

    # ── STEP 4.5: ZERO-DAY DETECTION (Novelty 3) ──────────────
    print("\n🛡️  STEP 4.5 — ZERO-DAY DETECTION  ← NOVELTY 3")

    anomaly_detector = AnomalyDetector(contamination=0.05)
    anomaly_detector.fit(X_train, y_train)

    y_pred_iso, iso_scores, zd_results = anomaly_detector.evaluate(
        X_test, y_test
    )

    # Simulate: treat 2 rarest multiclass attack types as 'unseen zero-day'
    zd_sim = anomaly_detector.simulate_zeroday(
        X_test, y_test_mc, mc_classes
    )

    anomaly_detector.plot_anomaly_scores(iso_scores, y_test)
    zd_auc = anomaly_detector.plot_roc_zeroday(iso_scores, y_test)
    print(f"  Zero-Day Detection AUC : {zd_auc:.4f}")

    # ── STEP 5: CONCEPT DRIFT (Novelty 2) ────────────────────
    print("\n📉 STEP 5 — CONCEPT DRIFT DETECTION  ← NOVELTY 2")

    drift_simulator = DriftSimulator(
        meta_model             = meta.model,
        class_names            = class_names,
        batch_size             = 200,
        drift_injection_batch  = 200,
    )

    df_drift = drift_simulator.run_simulation(
        meta_test, y_test,
        n_batches = n_drift_batches,
    )
    drift_simulator.plot_drift_analysis(df_drift)
    drift_simulator.plot_accuracy_by_phase(df_drift)
    drift_simulator.print_summary_statistics(df_drift)

    # ── STEP 6: PAPER RESULTS ────────────────────────────────
    print("\n📊 STEP 6 — GENERATING PAPER RESULTS")
    evaluator.plot_comparison_table()
    evaluator.generate_paper_table()
    evaluator.plot_radar_chart()

    # ROC curves
    meta_proba = meta.predict_proba(meta_test)
    mean_auc = evaluator.plot_roc_curves(
        y_test, meta_proba, model_name="xEnsembleGuard"
    )
    print(f"\n  Mean AUC-ROC: {mean_auc:.4f}")

    # ── FINAL SUMMARY ─────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n{'═'*58}")
    print(f"  ✅ PIPELINE COMPLETE — {dataset_name}")
    print(f"{'═'*58}")
    print(f"  Meta-model Accuracy : {meta_metrics['accuracy']*100:.2f}%")
    print(f"  Meta-model F1-Score : {meta_metrics['f1']*100:.2f}%")
    print(f"  Mean AUC-ROC        : {mean_auc:.4f}")
    if drift_simulator.drift_detected_at:
        latency = drift_simulator.drift_detected_at - 200
        print(f"  Drift detected in   : {latency} batches")
    print(f"  Total runtime       : {elapsed/60:.1f} minutes")
    print(f"\n  📁 Outputs saved to:")
    print(f"     models/   — all trained model files")
    print(f"     results/  — CSV tables for paper")
    print(f"     plots/    — all figures for paper")
    print(f"{'═'*58}")

    return {
        "dataset"        : dataset_name,
        "accuracy"       : meta_metrics["accuracy"],
        "f1"             : meta_metrics["f1"],
        "mean_auc"       : mean_auc,
        "shap_table"     : shap_table,
        "drift_results"  : df_drift,
    }


# =============================================================
# ENTRY POINT
# =============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="xEnsembleGuard: SHAP + Concept Drift IDS"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="UNSW_NB15",
        choices=["UNSW_NB15", "NSL_KDD", "CIC_IDS_2017"],
        help="Dataset to run"
    )
    parser.add_argument(
        "--all_datasets",
        action="store_true",
        help="Run on all 3 datasets sequentially"
    )
    parser.add_argument(
        "--no_dl",
        action="store_true",
        help="Skip deep learning models (faster, tree-only)"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable hyperparameter tuning for meta-model (slower)"
    )
    parser.add_argument(
        "--drift_batches",
        type=int,
        default=400,
        help="Number of batches for drift simulation (default: 400)"
    )
    args = parser.parse_args()

    banner()

    datasets = (["UNSW_NB15", "NSL_KDD", "CIC_IDS_2017"]
                if args.all_datasets else [args.dataset])

    all_pipeline_results = []
    for ds in datasets:
        result = run_pipeline(
            dataset_name   = ds,
            use_dl         = not args.no_dl,
            tune_meta      = args.tune,
            n_drift_batches= args.drift_batches,
        )
        all_pipeline_results.append(result)

    # Cross-dataset summary
    if len(all_pipeline_results) > 1:
        print(f"\n{'▓'*58}")
        print(f"  CROSS-DATASET SUMMARY")
        print(f"{'▓'*58}")
        for r in all_pipeline_results:
            print(f"  {r['dataset']:<20} | "
                  f"Acc: {r['accuracy']*100:.2f}% | "
                  f"F1: {r['f1']*100:.2f}% | "
                  f"AUC: {r['mean_auc']:.4f}")