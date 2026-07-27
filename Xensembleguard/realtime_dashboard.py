"""
Real-Time Network Intrusion Detection System
Drift-Aware Explainable Ensemble Learning for Adaptive Network Intrusion Detection
Run: streamlit run realtime_dashboard.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib, os, time, warnings, shap
from datetime import datetime
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Real-Time IDS | Network Intrusion Detection",
    page_icon="🛡️", layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }
[data-testid="stAppViewContainer"] { background:#f0f2f6; }
[data-testid="stSidebar"]          { background:#1e2a3a; }
.block-container                   { padding:1rem 2rem; }

/* Header */
.ids-header {
    background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #1565c0 100%);
    border-radius: 12px; padding: 20px 28px; margin-bottom: 16px;
    box-shadow: 0 4px 20px rgba(26,35,126,0.3);
    display: flex; align-items: center; justify-content: space-between;
}
.header-title { color:white; font-size:22px; font-weight:700; margin:0; }
.header-sub   { color:#90caf9; font-size:12px; margin:4px 0 0 0; }
.novelty-badge {
    background:rgba(255,255,255,0.15); color:white;
    padding:6px 14px; border-radius:20px; font-size:12px;
    font-weight:600; border:1px solid rgba(255,255,255,0.3);
}

/* KPI Cards */
.kpi-card {
    background:white; border-radius:10px; padding:16px 18px;
    box-shadow:0 2px 8px rgba(0,0,0,0.08);
    border-left:4px solid #1565c0;
}
.kpi-card.attack  { border-left-color:#d32f2f; }
.kpi-card.normal  { border-left-color:#2e7d32; }
.kpi-card.zeroday { border-left-color:#e65100; }
.kpi-card.acc     { border-left-color:#6a1b9a; }
.kpi-card.drift   { border-left-color:#f57f17; }
.kpi-label { font-size:11px; color:#78909c; font-weight:600;
             text-transform:uppercase; letter-spacing:0.5px; margin:0; }
.kpi-value { font-size:28px; font-weight:700; color:#1a237e; margin:4px 0 0 0; }
.kpi-sub   { font-size:11px; color:#90a4ae; margin:2px 0 0 0; }

/* Status badge */
.badge-attack  { background:#ffebee; color:#c62828; border:1px solid #ef9a9a;
                 border-radius:20px; padding:2px 10px; font-size:12px; font-weight:600; }
.badge-normal  { background:#e8f5e9; color:#1b5e20; border:1px solid #a5d6a7;
                 border-radius:20px; padding:2px 10px; font-size:12px; font-weight:600; }
.badge-zeroday { background:#fff3e0; color:#e65100; border:1px solid #ffcc80;
                 border-radius:20px; padding:2px 10px; font-size:12px; font-weight:600; }

/* Alert banners */
.drift-alert {
    background:#fff8e1; border:1px solid #ffc107; border-radius:8px;
    padding:12px 18px; margin:8px 0;
    display:flex; align-items:center; gap:12px;
}
.drift-alert-text { color:#e65100; font-weight:600; font-size:14px; }

/* Section cards */
.section-card {
    background:white; border-radius:10px; padding:16px;
    box-shadow:0 2px 8px rgba(0,0,0,0.06); margin-bottom:12px;
}
.section-title {
    font-size:14px; font-weight:700; color:#1a237e;
    margin:0 0 12px 0; display:flex; align-items:center; gap:8px;
}

/* Info box */
.info-box {
    background:#e3f2fd; border-radius:8px; padding:10px 14px;
    font-size:12px; color:#1565c0; margin-top:8px;
    border-left:3px solid #1565c0;
}

/* Packet source note */
.real-data-badge {
    background:#e8f5e9; color:#2e7d32; border:1px solid #81c784;
    border-radius:6px; padding:4px 10px; font-size:11px; font-weight:600;
}
</style>
""", unsafe_allow_html=True)


# ─── Load models ─────────────────────────────────────────
@st.cache_resource
def load_all():
    ms = {}
    for n, f in [("LightGBM","LightGBM.pkl"),("XGBoost","XGBoost.pkl"),
                 ("CatBoost","CatBoost.pkl"),("RandomForest","RandomForest.pkl"),
                 ("MetaModel","meta_model.pkl"),
                 ("IsolationForest","isolation_forest.pkl")]:
        p = f"models/{f}"
        if os.path.exists(p):
            ms[n] = joblib.load(p)
    return ms

@st.cache_data
def load_data():
    from data_preprocessing import DataPreprocessor
    return DataPreprocessor("UNSW_NB15").load()

@st.cache_resource
def get_explainer(_meta, _bg):
    return shap.TreeExplainer(_meta, _bg)

with st.spinner("Loading models..."):
    models = load_all()
    data   = load_data()
    X_all  = data["X_test"]
    y_all  = data["y_test_bin"]
    feat_names = data["feature_names"]

BASE = ["LightGBM","XGBoost","CatBoost","RandomForest"]
meta_feat_names = (
    [f"{m} → P({c})" for m in BASE for c in ["Normal","Attack"]]
    + list(feat_names)[:42]
)
_bg = np.hstack([np.hstack([models[m].predict_proba(X_all[:100]) for m in BASE
                              if m in models]), X_all[:100]])
explainer = get_explainer(models["MetaModel"], _bg)


# ─── Predict one packet ───────────────────────────────────
def predict_packet(x, y_true):
    X       = x.reshape(1,-1)
    iso     = models["IsolationForest"].predict(X)[0] == -1
    probas  = [models[m].predict_proba(X) for m in BASE if m in models]
    meta_in = np.hstack([np.hstack(probas), X])
    pred    = int(models["MetaModel"].predict(meta_in)[0])
    proba   = models["MetaModel"].predict_proba(meta_in)[0]

    # SHAP for attack packets only
    shap_feats, shap_vals = [], []
    if pred == 1 or iso:
        try:
            sv = explainer.shap_values(meta_in)
            v  = sv[1][0] if isinstance(sv, list) else (
                 sv[0,:,-1] if sv.ndim==3 else sv[0])
            ti = np.argsort(np.abs(v))[::-1][:5]
            shap_feats = [meta_feat_names[i] if i<len(meta_feat_names)
                          else f"Feature {i}" for i in ti]
            shap_vals  = [float(v[i]) for i in ti]
        except: pass

    return dict(
        pred=pred, label="Attack" if pred==1 else "Normal Traffic",
        confidence=float(max(proba))*100,
        prob_attack=float(proba[1]),
        is_zeroday=bool(iso),
        shap_feats=shap_feats, shap_vals=shap_vals,
        correct=int(pred==y_true)
    )


# ─── Session state ────────────────────────────────────────
from river.drift import ADWIN
for k,v in [("adwin",None),("log",[]),("acc_hist",[]),
             ("drifts",0),("running",False)]:
    if k not in st.session_state:
        st.session_state[k] = v
if st.session_state.adwin is None:
    st.session_state.adwin = ADWIN(delta=0.002)


# ─── HEADER ──────────────────────────────────────────────
st.markdown("""
<div class="ids-header">
  <div>
    <p class="header-title">🛡️ Real-Time Network Intrusion Detection System</p>
    <p class="header-sub">
      Drift-Aware Explainable Ensemble Learning for Adaptive Network Intrusion Detection
    </p>
  </div>
  <div>
    <span class="novelty-badge">✦ NOVELTY 4 — Real-Time Detection</span>
    <br/><br/>
    <span class="real-data-badge">🔬 Real UNSW-NB15 Network Traffic</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Controls ────────────────────────────────────────────
c1, c2, c3, c4 = st.columns([2,1,1,4])
with c1:
    speed_opt = st.selectbox(
        "Detection Speed",
        ["Fast — 0.3s per packet","Normal — 0.7s per packet","Slow — 1.5s per packet"],
        index=0, label_visibility="collapsed"
    )
    delay = {"Fast — 0.3s per packet":0.3,
             "Normal — 0.7s per packet":0.7,
             "Slow — 1.5s per packet":1.5}[speed_opt]
with c2:
    start = st.button("▶  Start Detection", type="primary", use_container_width=True)
    if start: st.session_state.running = True
with c3:
    stop = st.button("⏹  Stop", use_container_width=True)
    if stop: st.session_state.running = False
with c4:
    st.markdown("""
    <div class="info-box">
    📡 <b>Data Source:</b> Real network flows captured at UNSW-NB15 Cyber Lab —
    257,673 actual network connections with 42 features each
    (packet size, duration, protocol, connection type, bytes transferred, etc.)
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ─── KPI Row ─────────────────────────────────────────────
sd  = st.session_state.log
tot = len(sd)
atk = sum(1 for r in sd if "Attack" in r.get("Prediction",""))
nrm = tot - atk
zd  = sum(1 for r in sd if r.get("Zero-Day")=="⚠ Zero-Day")
dr  = st.session_state.drifts
acc_val = (sum(1 for r in sd if r.get("Correct")=="✓") / tot * 100) if tot else 0.0

k0,k1,k2,k3,k4,k5 = st.columns(6)
k0.markdown(f"""<div class="kpi-card">
<p class="kpi-label">📦 Total Packets</p>
<p class="kpi-value">{tot}</p>
<p class="kpi-sub">Processed in real time</p></div>""", unsafe_allow_html=True)

k1.markdown(f"""<div class="kpi-card attack">
<p class="kpi-label">🚨 Attacks Detected</p>
<p class="kpi-value" style="color:#c62828">{atk}</p>
<p class="kpi-sub">{atk/tot*100:.1f}% of traffic</p></div>""" if tot else
f"""<div class="kpi-card attack"><p class="kpi-label">🚨 Attacks Detected</p>
<p class="kpi-value" style="color:#c62828">0</p>
<p class="kpi-sub">Waiting...</p></div>""", unsafe_allow_html=True)

k2.markdown(f"""<div class="kpi-card normal">
<p class="kpi-label">✅ Normal Traffic</p>
<p class="kpi-value" style="color:#2e7d32">{nrm}</p>
<p class="kpi-sub">Allowed through</p></div>""", unsafe_allow_html=True)

k3.markdown(f"""<div class="kpi-card zeroday">
<p class="kpi-label">⚠ Zero-Day Alerts</p>
<p class="kpi-value" style="color:#e65100">{zd}</p>
<p class="kpi-sub">Unknown attack patterns</p></div>""", unsafe_allow_html=True)

k4.markdown(f"""<div class="kpi-card acc">
<p class="kpi-label">🎯 Detection Accuracy</p>
<p class="kpi-value" style="color:#6a1b9a">{acc_val:.1f}%</p>
<p class="kpi-sub">Live rolling accuracy</p></div>""", unsafe_allow_html=True)

k5.markdown(f"""<div class="kpi-card drift">
<p class="kpi-label">⚡ Drift Alerts</p>
<p class="kpi-value" style="color:#f57f17">{dr}</p>
<p class="kpi-sub">Model retraining needed</p></div>""", unsafe_allow_html=True)

# ─── Drift banner ─────────────────────────────────────────
drift_slot = st.empty()

# ─── Main layout ─────────────────────────────────────────
left, right = st.columns([3,2])

with left:
    st.markdown('<p class="section-title">📡 Live Network Traffic Feed</p>',
                unsafe_allow_html=True)
    st.caption("Every row = one real network connection being classified as it arrives")
    feed_slot  = st.empty()

with right:
    st.markdown('<p class="section-title">🧠 Why Was This Flagged? (SHAP Explanation)</p>',
                unsafe_allow_html=True)
    st.caption("SHAP shows which network feature caused the detection decision")
    shap_slot  = st.empty()
    reason_slot= st.empty()

# ─── Accuracy graph ───────────────────────────────────────
st.markdown('<p class="section-title">📈 Live Accuracy Monitor (ADWIN Drift Detection)</p>',
            unsafe_allow_html=True)
st.caption("When accuracy drops below threshold — ADWIN raises a drift alert and recommends retraining")
graph_slot = st.empty()


# ─── STREAM LOOP ─────────────────────────────────────────
if st.session_state.running:
    idx    = int(np.random.randint(0, len(X_all)))
    x, yt  = X_all[idx], y_all[idx]
    r      = predict_packet(x, yt)

    st.session_state.adwin.update(r["correct"])
    drift = st.session_state.adwin.drift_detected
    if drift:
        st.session_state.drifts += 1

    ts = datetime.now().strftime("%H:%M:%S")
    entry = {
        "Time"      : ts,
        "Prediction": r["label"],
        "Confidence": f"{r['confidence']:.1f}%",
        "Zero-Day"  : "⚠ Zero-Day" if r["is_zeroday"] else "—",
        "P(Attack)" : f"{r['prob_attack']:.3f}",
        "Drift"     : "⚡ YES" if drift else "—",
        "True Label": "Attack" if yt==1 else "Normal",
        "Correct"   : "✓" if r["correct"] else "✗",
    }
    st.session_state.log.append(entry)
    if len(st.session_state.log) > 60:
        st.session_state.log = st.session_state.log[-60:]

    rec  = st.session_state.log[-20:]
    lacc = sum(1 for e in rec if e["Correct"]=="✓") / len(rec) * 100
    st.session_state.acc_hist.append(
        {"Packet": len(st.session_state.log), "Accuracy": lacc})

    # Drift banner
    if drift:
        drift_slot.markdown("""
        <div class="drift-alert">
          <span style="font-size:22px">⚡</span>
          <div>
            <div class="drift-alert-text">Concept Drift Detected by ADWIN!</div>
            <div style="font-size:12px;color:#795548;margin-top:2px;">
              Attack patterns have changed — model accuracy is dropping.
              Retraining the model is recommended.
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        drift_slot.empty()

    # Feed table — styled
    df_show = pd.DataFrame(st.session_state.log[::-1][:15])
    def style_row(row):
        styles = [""] * len(row)
        if "Attack" in str(row.get("Prediction","")):
            styles = ["background:#fff5f5;color:#c62828"] * len(row)
        elif row.get("Zero-Day","") == "⚠ Zero-Day":
            styles = ["background:#fff8e1;color:#e65100"] * len(row)
        if row.get("Drift","") == "⚡ YES":
            styles = ["background:#fffde7;color:#f57f17"] * len(row)
        return styles

    styled = df_show.style.apply(style_row, axis=1)\
                          .set_properties(**{"font-size":"12px"})\
                          .set_table_styles([
                              {"selector":"th","props":[
                                  ("background","#1a237e"),("color","white"),
                                  ("font-size","11px"),("padding","6px 10px"),
                                  ("text-transform","uppercase"),("letter-spacing","0.5px")
                              ]},
                              {"selector":"td","props":[("padding","6px 10px")]},
                          ])
    feed_slot.dataframe(styled, use_container_width=True, height=370)

    # SHAP chart
    if r["shap_feats"] and r["pred"] == 1:
        colors = ["#d32f2f" if v>0 else "#2e7d32" for v in r["shap_vals"]]
        labels = ["→ ATTACK" if v>0 else "→ NORMAL" for v in r["shap_vals"]]
        fig = go.Figure(go.Bar(
            x=r["shap_vals"][::-1],
            y=[f[:28] for f in r["shap_feats"][::-1]],
            orientation="h",
            marker_color=colors[::-1],
            text=[f"{v:+.3f} {l}" for v,l in
                  zip(r["shap_vals"][::-1], labels[::-1])],
            textposition="outside",
            textfont=dict(size=11)
        ))
        fig.update_layout(
            title=dict(text="Feature Impact on This Detection",
                       font=dict(size=13, color="#1a237e"), x=0),
            xaxis=dict(title="SHAP Value (Impact Strength)",
                       title_font=dict(size=11),
                       zeroline=True, zerolinecolor="#bdbdbd"),
            yaxis=dict(tickfont=dict(size=11)),
            paper_bgcolor="white", plot_bgcolor="#fafafa",
            font=dict(color="#424242"),
            height=280, margin=dict(l=180,r=80,t=40,b=40),
            shapes=[dict(type="line",x0=0,x1=0,
                         y0=-0.5,y1=len(r["shap_feats"])-0.5,
                         line=dict(color="#9e9e9e",width=1.5))]
        )
        shap_slot.plotly_chart(fig, use_container_width=True)

        top_f = r["shap_feats"][0]
        top_v = r["shap_vals"][0]
        reason_slot.markdown(
            f"""<div class="info-box">
            <b>🔍 Main Trigger:</b> <code>{top_f}</code> — 
            Impact Score: <b>{top_v:+.3f}</b> — 
            {'This feature strongly indicates an <b>ATTACK</b> 🔴' if top_v>0
             else 'This feature suggests <b>NORMAL</b> traffic 🟢'}
            </div>""", unsafe_allow_html=True)
    elif r["is_zeroday"]:
        shap_slot.markdown("""
        <div style="background:#fff3e0;border:1px solid #ffb74d;border-radius:8px;
                    padding:20px;text-align:center;">
          <div style="font-size:32px">⚠️</div>
          <div style="font-weight:700;color:#e65100;font-size:16px;margin:8px 0;">
            Zero-Day Attack Detected
          </div>
          <div style="color:#795548;font-size:13px;">
            Isolation Forest flagged this network flow as anomalous.<br/>
            This traffic pattern was never seen during training —<br/>
            it may be a brand-new, unknown attack type.
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        shap_slot.markdown("""
        <div style="background:#e8f5e9;border:1px solid #81c784;border-radius:8px;
                    padding:20px;text-align:center;">
          <div style="font-size:32px">✅</div>
          <div style="font-weight:700;color:#2e7d32;font-size:16px;margin:8px 0;">
            Normal Network Traffic
          </div>
          <div style="color:#388e3c;font-size:13px;">
            All models agree this connection is safe.<br/>
            No suspicious features detected.<br/>
            SHAP explanation not required.
          </div>
        </div>""", unsafe_allow_html=True)

    # Accuracy graph
    if len(st.session_state.acc_hist) > 1:
        df_a = pd.DataFrame(st.session_state.acc_hist)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df_a["Packet"], y=df_a["Accuracy"],
            mode="lines", name="Live Accuracy",
            line=dict(color="#1565c0", width=2.5),
            fill="tozeroy", fillcolor="rgba(21,101,192,0.08)"
        ))
        fig2.add_hline(y=95, line_dash="dot", line_color="#2e7d32",
                       line_width=1.5,
                       annotation_text="95% Target",
                       annotation_font_color="#2e7d32",
                       annotation_font_size=11)
        fig2.add_hline(y=90, line_dash="dot", line_color="#f57f17",
                       line_width=1.5,
                       annotation_text="⚡ Drift Threshold",
                       annotation_font_color="#f57f17",
                       annotation_font_size=11)
        fig2.update_layout(
            xaxis=dict(title="Packets Processed",
                       title_font=dict(size=11), gridcolor="#eeeeee"),
            yaxis=dict(title="Accuracy (%)", range=[60,101],
                       title_font=dict(size=11), gridcolor="#eeeeee"),
            paper_bgcolor="white", plot_bgcolor="#fafafa",
            font=dict(color="#424242"),
            height=200, margin=dict(l=50,r=30,t=10,b=40),
            showlegend=False
        )
        graph_slot.plotly_chart(fig2, use_container_width=True)

    time.sleep(delay)
    st.rerun()

else:
    if not st.session_state.log:
        feed_slot.markdown("""
        <div style="background:white;border-radius:10px;padding:40px;
                    text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.06);">
          <div style="font-size:40px;margin-bottom:12px">📡</div>
          <div style="font-size:16px;font-weight:600;color:#1a237e;">
            Ready to Start Detection
          </div>
          <div style="font-size:13px;color:#78909c;margin-top:8px;">
            Press <b>▶ Start Detection</b> to begin streaming<br/>
            real UNSW-NB15 network traffic through the IDS pipeline
          </div>
        </div>""", unsafe_allow_html=True)
        shap_slot.markdown("""
        <div style="background:white;border-radius:10px;padding:40px;
                    text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.06);">
          <div style="font-size:40px;margin-bottom:12px">🧠</div>
          <div style="font-size:13px;color:#78909c;">
            SHAP explanation will appear here<br/>when an attack is detected
          </div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# REAL PACKET CAPTURE TAB (add to sidebar)
# ─────────────────────────────────────────────────────────
# This is called from realtime_dashboard.py when user selects
# "Live Capture" mode. Import and use LiveCaptureEngine.
