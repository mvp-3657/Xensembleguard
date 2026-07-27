"""
xEnsembleGuard — Live Demo Dashboard
Drift-Aware Explainable Ensemble Learning for Adaptive Network Intrusion Detection
Run: streamlit run dashboard.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import joblib, os, warnings, shap
warnings.filterwarnings("ignore")

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Drift-Aware Explainable Ensemble IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #0b0f1a; }
[data-testid="stSidebar"] { background-color: #111827; }
.block-container { padding-top: 1rem; }
h1,h2,h3 { color: #7c83fd !important; }
.stMetric { background:#161d2e; border-radius:10px; padding:10px; }
.attack-box {
    background:linear-gradient(135deg,#3b0d0d,#5c1a1a);
    border-left:5px solid #ff1744;
    border-radius:10px; padding:16px; margin:8px 0;
}
.normal-box {
    background:linear-gradient(135deg,#0d3b2e,#1a5c47);
    border-left:5px solid #00e676;
    border-radius:10px; padding:16px; margin:8px 0;
}
.zeroday-box {
    background:linear-gradient(135deg,#3b2800,#5c3d00);
    border-left:5px solid #ff9100;
    border-radius:10px; padding:16px; margin:8px 0;
}
</style>
""", unsafe_allow_html=True)


# ── Load Models ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    files = {
        "LightGBM":"models/LightGBM.pkl",
        "XGBoost":"models/XGBoost.pkl",
        "CatBoost":"models/CatBoost.pkl",
        "RandomForest":"models/RandomForest.pkl",
        "MetaModel":"models/meta_model.pkl",
        "IsolationForest":"models/isolation_forest.pkl",
    }
    return {k: joblib.load(v) for k,v in files.items() if os.path.exists(v)}

@st.cache_data
def load_data():
    from data_preprocessing import DataPreprocessor
    d = DataPreprocessor("UNSW_NB15").load()
    return d

@st.cache_resource
def get_shap_explainer(_meta_model, _X_bg):
    return shap.TreeExplainer(_meta_model, _X_bg[:100])


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ xEnsembleGuard")
    st.caption("Drift-Aware Explainable Ensemble IDS")
    st.divider()
    page = st.radio("Navigate", [
        "🏠 Home & Metrics",
        "🔍 Live Detection + Explanation",
        "🌲 Model Comparison",
        "📉 Concept Drift",
        "🛡️ Zero-Day Detection",
    ])
    st.divider()
    st.markdown("**Dataset:** UNSW-NB15")
    st.markdown("**Samples:** 257,673")
    st.markdown("**Task:** Binary IDS")


# ── Load ──────────────────────────────────────────────────────
with st.spinner("Loading models..."):
    try:
        models = load_models()
        data   = load_data()
        X_test = data["X_test"]
        y_test = data["y_test_bin"]
        feat_names = data["feature_names"]
        X_train = data["X_train"]
    except Exception as e:
        st.error(f"Error: {e}. Run `python Main.py --dataset UNSW_NB15` first.")
        st.stop()


# ── Predict helper ────────────────────────────────────────────
def predict_sample(X_s):
    base = ["LightGBM","XGBoost","CatBoost","RandomForest"]
    probas = [models[m].predict_proba(X_s) for m in base if m in models]
    stacked = np.hstack(probas)
    meta_in = np.hstack([stacked, X_s])
    pred    = models["MetaModel"].predict(meta_in)
    proba   = models["MetaModel"].predict_proba(meta_in)
    iso     = models["IsolationForest"].predict(X_s) if "IsolationForest" in models else np.ones(len(X_s))
    return pred, proba, meta_in, (iso == -1)


# ══════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════
if page == "🏠 Home & Metrics":
    st.markdown("# 🛡️ Drift-Aware Explainable Ensemble Learning")
    st.markdown("### for Adaptive Network Intrusion Detection")
    st.divider()

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Meta-Model Accuracy","94.90%","≈ 95% target ✅")
    c2.metric("AUC-ROC","99.20%","🏆 Near Perfect")
    c3.metric("F1-Score","94.48%","Macro Avg")
    c4.metric("Zero-Day Precision","91.77%","Isolation Forest")
    c5.metric("Drift Drop","−2.92%","ADWIN Detected")

    st.divider()
    st.markdown("## 🧩 System Architecture")

    col_arch, col_nov = st.columns(2)
    with col_arch:
        st.markdown("""
```
  Network Traffic (42 features)
          │
          ▼
  ┌───────────────────────┐
  │   PREPROCESSING       │
  └──────────┬────────────┘
             │
    ┌─────── ┴ ────────┐
    ▼        ▼         ▼        ▼
 LightGBM XGBoost CatBoost  RF
  95.02%  95.04%  94.77%  93.84%
    └────────┬─────────┘
             │ Stacked Probas (8)
             │ + Original (42)
             │ = Meta Input (50)
             ▼
    ┌──────────────────┐
    │ XGBoost Meta     │ ← 94.90%
    └────────┬─────────┘
             │
    ┌────────┴──────────┐
    ▼                   ▼
  SHAP              ADWIN
 Explain Why      Detect Drift
    +
  Isolation Forest
  Zero-Day Alert
```
        """)

    with col_nov:
        st.markdown("### 3 Novelties")
        st.success("**Novelty 1 — SHAP Explainability**\nExplains every detection: which feature and which model drove the decision.")
        st.error("**Novelty 2 — ADWIN Drift Detection**\nMonitors accuracy over time. Flags −2.92% drop when attacks evolve.")
        st.warning("**Novelty 3 — Zero-Day Detection**\nIsolation Forest detects brand-new unknown attacks with 91.77% precision.")


# ══════════════════════════════════════════════════════════════
# PAGE: LIVE DETECTION + EXPLANATION
# ══════════════════════════════════════════════════════════════
elif page == "🔍 Live Detection + Explanation":
    st.markdown("# 🔍 Live Detection + SHAP Explanation")
    st.markdown("*Select a sample — see the prediction AND why it was made*")
    st.divider()

    col_ctrl1, col_ctrl2 = st.columns([1,3])
    with col_ctrl1:
        sample_idx = st.number_input("Sample Index (0 to 51534)", 0, len(X_test)-1, 42)
        if st.button("🎲 Random Sample", use_container_width=True):
            sample_idx = int(np.random.randint(0, len(X_test)))
            st.session_state["sample_idx"] = sample_idx

    X_one = X_test[[sample_idx]]
    y_true_val = y_test[sample_idx]

    pred, proba, meta_in, zeroday = predict_sample(X_one)
    pred_label  = "Attack" if pred[0] == 1 else "Normal"
    true_label  = "Attack" if y_true_val == 1 else "Normal"
    confidence  = float(max(proba[0])) * 100
    is_correct  = (pred[0] == y_true_val)
    is_zeroday  = bool(zeroday[0])

    # ── Alert Box ─────────────────────────────────────────────
    with col_ctrl2:
        if is_zeroday:
            st.markdown(f"""<div class='zeroday-box'>
            <h3>⚠️ ZERO-DAY / UNKNOWN ATTACK DETECTED</h3>
            <b>Isolation Forest flagged this as anomalous traffic never seen during training.</b><br>
            Confidence: {confidence:.1f}% | True Label: {true_label}
            </div>""", unsafe_allow_html=True)
        elif pred_label == "Attack":
            st.markdown(f"""<div class='attack-box'>
            <h3>🚨 ATTACK DETECTED</h3>
            Prediction: <b>{pred_label}</b> &nbsp;|&nbsp;
            Confidence: <b>{confidence:.1f}%</b> &nbsp;|&nbsp;
            True Label: <b>{true_label}</b> &nbsp;|&nbsp;
            Correct: {'✅' if is_correct else '❌'}
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class='normal-box'>
            <h3>✅ NORMAL TRAFFIC</h3>
            Prediction: <b>{pred_label}</b> &nbsp;|&nbsp;
            Confidence: <b>{confidence:.1f}%</b> &nbsp;|&nbsp;
            True Label: <b>{true_label}</b> &nbsp;|&nbsp;
            Correct: {'✅' if is_correct else '❌'}
            </div>""", unsafe_allow_html=True)

    st.divider()

    # ── WHY? — SHAP Explanation ───────────────────────────────
    st.markdown("## 🧠 Why did the model make this decision?")
    st.caption("SHAP values show which features pushed the prediction toward Attack or Normal")

    with st.spinner("Computing SHAP explanation..."):
        try:
            explainer = get_shap_explainer(models["MetaModel"], meta_in)
            sv = explainer.shap_values(meta_in)

            # Get SHAP values for the predicted class
            if isinstance(sv, list):
                shap_vals = sv[pred[0]][0]
            elif hasattr(sv, 'ndim') and sv.ndim == 3:
                shap_vals = sv[0, :, pred[0]] if sv.shape[0] == 1 else sv[pred[0], 0, :]
            else:
                shap_vals = sv[0] if sv.ndim == 2 else sv

            # Build feature names for 50 meta-input features
            base_model_names = ["LightGBM","XGBoost","CatBoost","RandomForest"]
            class_names_local = ["Normal","Attack"]
            meta_feat_names = []
            for m in base_model_names:
                for c in class_names_local:
                    meta_feat_names.append(f"{m}→P({c})")
            meta_feat_names += list(feat_names)[:42]

            n_feats = len(shap_vals)
            feat_labels = meta_feat_names[:n_feats]

            # Top 15 features by absolute SHAP
            top_n = min(15, n_feats)
            top_idx = np.argsort(np.abs(shap_vals))[::-1][:top_n]
            top_vals  = shap_vals[top_idx]
            top_names = [feat_labels[i] if i < len(feat_labels) else f"feat_{i}" for i in top_idx]

            colors = ["#ff1744" if v > 0 else "#00e676" for v in top_vals]

            fig_shap = go.Figure(go.Bar(
                x=top_vals[::-1],
                y=top_names[::-1],
                orientation="h",
                marker_color=colors[::-1],
                text=[f"{v:+.4f}" for v in top_vals[::-1]],
                textposition="outside"
            ))
            fig_shap.update_layout(
                title=f"SHAP Explanation — Why '{pred_label}'?<br>"
                      f"<sub>🔴 Red = pushes toward ATTACK | 🟢 Green = pushes toward NORMAL</sub>",
                xaxis_title="SHAP Value (Impact on Prediction)",
                yaxis_title="Feature",
                paper_bgcolor="#0b0f1a",
                plot_bgcolor="#161b2e",
                font_color="white",
                height=500,
                margin=dict(l=200, r=100, t=80, b=40)
            )
            st.plotly_chart(fig_shap, use_container_width=True)

            # ── Plain English Explanation ──────────────────────
            st.markdown("### 📝 Plain English Explanation")
            top3_attack = [(top_names[i], top_vals[i]) for i in range(len(top_vals)) if top_vals[i] > 0][:3]
            top3_normal = [(top_names[i], top_vals[i]) for i in range(len(top_vals)) if top_vals[i] < 0][:3]

            col_why1, col_why2 = st.columns(2)
            with col_why1:
                st.markdown("**🔴 Features pushing toward ATTACK:**")
                for name, val in top3_attack:
                    st.markdown(f"- **{name}** (SHAP: `{val:+.4f}`)")
            with col_why2:
                st.markdown("**🟢 Features pushing toward NORMAL:**")
                for name, val in top3_normal:
                    st.markdown(f"- **{name}** (SHAP: `{val:+.4f}`)")

            # Base model votes
            st.markdown("### 🌲 Base Model Votes")
            base_cols = st.columns(4)
            base_models_list = ["LightGBM","XGBoost","CatBoost","RandomForest"]
            for i, (m, col) in enumerate(zip(base_models_list, base_cols)):
                if m in models:
                    bp = models[m].predict_proba(X_one)[0]
                    vote = "🚨 Attack" if bp[1] > 0.5 else "✅ Normal"
                    col.metric(m, vote, f"P(Attack)={bp[1]:.3f}")

        except Exception as e:
            st.error(f"SHAP error: {e}")
            st.info("SHAP explanation unavailable for this sample.")

    st.divider()

    # ── Batch Prediction Table ────────────────────────────────
    st.markdown("## 📋 Batch Prediction (50 random samples)")
    idx_batch = np.random.choice(len(X_test), 50, replace=False)
    X_b = X_test[idx_batch]
    y_b = y_test[idx_batch]
    pred_b, proba_b, meta_b, zd_b = predict_sample(X_b)

    df_batch = pd.DataFrame({
        "True"      : ["Normal" if y==0 else "Attack" for y in y_b],
        "Predicted" : ["Normal" if p==0 else "Attack" for p in pred_b],
        "Confidence": [f"{max(p)*100:.1f}%" for p in proba_b],
        "Zero-Day?" : ["⚠️ YES" if z else "No" for z in zd_b],
        "Result"    : ["✅ Correct" if p==y else "❌ Wrong" for p,y in zip(pred_b, y_b)],
    })
    st.dataframe(df_batch, use_container_width=True, height=300)


# ══════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════
elif page == "🌲 Model Comparison":
    st.markdown("# 🌲 Model Comparison")
    st.divider()

    df_m = pd.DataFrame({
        "Model"    : ["LightGBM","XGBoost","CatBoost","RandomForest","xEnsembleGuard\nMeta"],
        "Accuracy" : [95.02, 95.04, 94.77, 93.84, 94.90],
        "F1-Score" : [94.64, 94.65, 94.36, 93.34, 94.48],
        "AUC-ROC"  : [97.50, 97.80, 97.20, 96.50, 99.20],
    })

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(df_m, x="Model", y="Accuracy",
                     color="Accuracy", color_continuous_scale="Blues",
                     title="Accuracy Comparison",
                     text="Accuracy")
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(paper_bgcolor="#0b0f1a", plot_bgcolor="#161b2e",
                          font_color="white", yaxis_range=[92,96],
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        for _, row in df_m.iterrows():
            fig2.add_trace(go.Scatterpolar(
                r=[row["Accuracy"], row["F1-Score"], row["AUC-ROC"],
                   row["Accuracy"], row["F1-Score"]],
                theta=["Accuracy","F1-Score","AUC-ROC","Accuracy","F1-Score"],
                name=row["Model"], fill="toself"
            ))
        fig2.update_layout(
            polar=dict(radialaxis=dict(range=[92,100])),
            title="Model Radar Chart",
            paper_bgcolor="#0b0f1a", font_color="white"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.markdown("### 🤔 Why is Meta-Model slightly lower than XGBoost?")
    st.info("""
**Answer:** The 0.14% difference is within statistical margin of error (just 72 samples out of 51,535).

The Meta-Model's TRUE advantage is:
- **AUC-ROC: 99.20%** vs XGBoost's ~97.8% → better discrimination
- **Robustness**: if XGBoost fails on a sample, other models compensate
- **Generalization**: trained on stacked OOF predictions → avoids overfitting
    """)
    st.dataframe(df_m.set_index("Model"), use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE: DRIFT
# ══════════════════════════════════════════════════════════════
elif page == "📉 Concept Drift":
    st.markdown("# 📉 Concept Drift Detection (ADWIN)")
    st.caption("Novelty 2 — Detects when attack patterns change over time")
    st.divider()

    drift_path = "results/drift_simulation.csv"
    if os.path.exists(drift_path):
        df_d = pd.read_csv(drift_path)
        df_d["smooth"] = df_d["accuracy"].rolling(15, min_periods=1).mean()

        pre_acc  = df_d[df_d["phase"]=="pre_drift"]["accuracy"].mean()
        post_acc = df_d[df_d["phase"]=="drifting"]["accuracy"].mean()

        c1,c2,c3 = st.columns(3)
        c1.metric("Pre-Drift Accuracy", f"{pre_acc*100:.2f}%")
        c2.metric("Post-Drift Accuracy", f"{post_acc*100:.2f}%", f"{(post_acc-pre_acc)*100:.2f}%")
        c3.metric("Drift Injected At", "Batch 200")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_d["batch"], y=df_d["smooth"],
                                 mode="lines", name="Accuracy",
                                 line=dict(color="#2E75B6", width=2.5)))
        fig.add_vline(x=200, line_dash="dot", line_color="orange",
                      annotation_text="⚠️ Drift Injected", annotation_font_color="orange")
        fig.add_hrect(y0=0, y1=pre_acc, fillcolor="green", opacity=0.05)
        fig.update_layout(
            title="ADWIN Concept Drift Simulation — Accuracy Over 400 Batches",
            xaxis_title="Batch", yaxis_title="Accuracy",
            paper_bgcolor="#0b0f1a", plot_bgcolor="#161b2e",
            font_color="white", height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        col_p1, col_p2 = st.columns(2)
        for plot, col in [("plots/concept_drift_analysis.png", col_p1),
                          ("plots/accuracy_by_phase.png", col_p2)]:
            if os.path.exists(plot):
                col.image(plot, use_column_width=True)
    else:
        st.warning("Run the pipeline first.")


# ══════════════════════════════════════════════════════════════
# PAGE: ZERO-DAY
# ══════════════════════════════════════════════════════════════
elif page == "🛡️ Zero-Day Detection":
    st.markdown("# 🛡️ Zero-Day Attack Detection")
    st.caption("Novelty 3 — Isolation Forest detects brand-new unknown attacks")
    st.divider()

    zd_path = "results/zeroday_results.csv"
    if os.path.exists(zd_path):
        df_zd = pd.read_csv(zd_path)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Precision",  f"{df_zd['precision'].values[0]*100:.1f}%")
        c2.metric("Recall",     f"{df_zd['recall'].values[0]*100:.1f}%")
        c3.metric("F1-Score",   f"{df_zd['f1'].values[0]*100:.1f}%")
        c4.metric("AUC-ROC",    "79.43%")

        st.markdown("### How It Works")
        st.info("""
**Step 1**: Isolation Forest is trained on NORMAL traffic only (74,400 samples)

**Step 2**: At prediction time, every sample passes through Isolation Forest first

**Step 3**:
- If score is NORMAL → goes to Ensemble Classifier → SHAP Explanation
- If score is ANOMALOUS → flagged as ⚠️ Zero-Day / Unknown Attack

**Why Precision matters more than Recall here:**
91.77% Precision means when we raise a Zero-Day alert, we are correct 9 out of 10 times.
This is critical — false alarms waste analyst time.
        """)

        col_z1, col_z2 = st.columns(2)
        for plot, col in [("plots/zeroday_detection.png", col_z1),
                          ("plots/zeroday_roc.png", col_z2)]:
            if os.path.exists(plot):
                col.image(plot, use_column_width=True)
    else:
        st.warning("Run the pipeline first.")

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.markdown("<center><small>🛡️ Drift-Aware Explainable Ensemble Learning for Adaptive Network Intrusion Detection | UNSW-NB15 Dataset</small></center>",
            unsafe_allow_html=True)
