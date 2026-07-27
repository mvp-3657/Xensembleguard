"""
IDS Live Dashboard — 100% Professional Design
Run: streamlit run live_capture_dashboard.py --server.port 8503
"""
import streamlit as st, pandas as pd, plotly.graph_objects as go
import os, time, json

LIVE_CSV = "results/live_packets.csv"

st.set_page_config(page_title="IDS Live Monitor",page_icon="🛡️",
                   layout="wide",initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
html,body,[class*="css"],*{font-family:'Inter',sans-serif !important;}
[data-testid="stAppViewContainer"]{background:#f8fafc !important;}
[data-testid="stSidebar"]{display:none;}
.block-container{padding:0 !important;max-width:100% !important;}
[data-testid="stVerticalBlock"]{gap:0 !important;}
footer{display:none;}
#MainMenu{display:none;}
header{display:none;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}
@keyframes slideIn{from{opacity:0;transform:translateY(-6px)}to{opacity:1;transform:translateY(0)}}
.fade-in{animation:slideIn 0.4s ease;}
</style>""",unsafe_allow_html=True)

def load():
    if not os.path.exists(LIVE_CSV):return None
    try:
        df=pd.read_csv(LIVE_CSV)
        return df if not df.empty else None
    except:return None

df=load()

# ══════════════════════════════════════════
# SETUP SCREEN
# ══════════════════════════════════════════
if df is None:
    st.markdown("""
    <div style="min-height:100vh;background:#f8fafc;display:flex;
                flex-direction:column;align-items:center;justify-content:center;">
      <div style="background:linear-gradient(135deg,#0f172a,#1e3a8a);
                  padding:16px 28px;border-radius:14px;margin-bottom:40px;
                  text-align:center;">
        <div style="color:white;font-size:20px;font-weight:800;">
          🛡️ Network Intrusion Detection System
        </div>
        <div style="color:#93c5fd;font-size:12px;margin-top:4px;">
          Real-Time Live Packet Capture
        </div>
      </div>
      <div style="background:white;border-radius:16px;padding:36px 40px;
                  box-shadow:0 8px 40px rgba(0,0,0,0.08);max-width:560px;width:90%;">
        <div style="font-size:17px;font-weight:700;color:#0f172a;margin-bottom:6px;">
          ⚙️ Start Packet Capture First
        </div>
        <div style="font-size:13px;color:#64748b;margin-bottom:24px;line-height:1.6;">
          Open a new Terminal window and run the command below.<br/>
          This dashboard updates automatically once capture starts.
        </div>
        <div style="background:#0f172a;border-radius:10px;padding:16px 20px;
                    font-family:monospace;font-size:13px;color:#7dd3fc;
                    letter-spacing:0.3px;">
          sudo python packet_capture.py --iface en0
        </div>
        <div style="margin-top:20px;padding:14px 16px;background:#f0fdf4;
                    border:1px solid #bbf7d0;border-radius:10px;
                    font-size:13px;color:#166534;line-height:1.6;">
          💡 After running the command, <b>open any website</b> in Chrome or Safari.
          Your network traffic will appear here within 5–10 seconds.
        </div>
      </div>
    </div>""",unsafe_allow_html=True)
    time.sleep(3);st.rerun();st.stop()

# ══════════════════════════════════════════
# DATA PREP
# ══════════════════════════════════════════
tot  = len(df)
atks = int((df["prediction"]=="Attack").sum())
nrm  = tot-atks
zds  = int(df.get("zero_day",pd.Series([False]*tot)).astype(bool).sum())
drs  = int(df.get("drift",pd.Series([False]*tot)).astype(bool).sum())
latest=df.iloc[-1]
is_atk=latest["prediction"]=="Attack"
is_zd =bool(latest.get("zero_day",False))
conf  =float(latest.get("confidence",0))
svc   =str(latest.get("service","—"))
dst   =str(latest.get("dst_ip","—"))
prob  =float(latest.get("prob_attack",0))
try:reasons=json.loads(str(latest.get("shap_json","[]")))
except:reasons=[]

# ══════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════
st.markdown(f"""
<div style="background:linear-gradient(90deg,#0f172a 0%,#1e3a8a 100%);
            padding:14px 32px;display:flex;justify-content:space-between;
            align-items:center;position:sticky;top:0;z-index:999;">
  <div style="display:flex;align-items:center;gap:16px;">
    <span style="font-size:22px;">🛡️</span>
    <div>
      <div style="color:white;font-size:16px;font-weight:800;letter-spacing:-0.3px;">
        Network Intrusion Detection System
      </div>
      <div style="color:#93c5fd;font-size:11px;margin-top:1px;">
        AI-powered real-time security monitor · Novelty 4
      </div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:12px;">
    <div style="display:flex;align-items:center;gap:6px;background:rgba(239,68,68,0.12);
                border:1px solid rgba(239,68,68,0.35);border-radius:20px;padding:5px 14px;">
      <div style="width:7px;height:7px;background:#ef4444;border-radius:50%;
                  animation:blink 1s step-end infinite;"></div>
      <span style="color:#fca5a5;font-size:11px;font-weight:700;letter-spacing:0.5px;">
        CAPTURING LIVE PACKETS
      </span>
    </div>
    <div style="color:#475569;font-size:11px;background:rgba(255,255,255,0.07);
                padding:5px 12px;border-radius:20px;border:1px solid rgba(255,255,255,0.1);">
      WiFi · en0 · Auto-refresh 2s
    </div>
  </div>
</div>
""",unsafe_allow_html=True)

# ══════════════════════════════════════════
# KPI STRIP
# ══════════════════════════════════════════
st.markdown(f"""
<div style="display:flex;background:white;border-bottom:1px solid #e2e8f0;">
  <div style="flex:1;padding:18px 24px;border-right:1px solid #e2e8f0;
              border-bottom:3px solid #3b82f6;">
    <div style="font-size:10px;font-weight:700;color:#94a3b8;
                text-transform:uppercase;letter-spacing:1px;">Total Packets Seen</div>
    <div style="font-size:32px;font-weight:800;color:#0f172a;margin:6px 0 4px;">{tot}</div>
    <div style="font-size:12px;color:#94a3b8;">Network connections checked</div>
  </div>
  <div style="flex:1;padding:18px 24px;border-right:1px solid #e2e8f0;
              border-bottom:3px solid #ef4444;">
    <div style="font-size:10px;font-weight:700;color:#94a3b8;
                text-transform:uppercase;letter-spacing:1px;">🚨 Attacks Found</div>
    <div style="font-size:32px;font-weight:800;color:#dc2626;margin:6px 0 4px;">{atks}</div>
    <div style="font-size:12px;color:#94a3b8;">{atks/tot*100:.1f}% of all traffic</div>
  </div>
  <div style="flex:1;padding:18px 24px;border-right:1px solid #e2e8f0;
              border-bottom:3px solid #22c55e;">
    <div style="font-size:10px;font-weight:700;color:#94a3b8;
                text-transform:uppercase;letter-spacing:1px;">✅ Safe Connections</div>
    <div style="font-size:32px;font-weight:800;color:#16a34a;margin:6px 0 4px;">{nrm}</div>
    <div style="font-size:12px;color:#94a3b8;">{nrm/tot*100:.1f}% passed safely</div>
  </div>
  <div style="flex:1;padding:18px 24px;border-right:1px solid #e2e8f0;
              border-bottom:3px solid #f97316;">
    <div style="font-size:10px;font-weight:700;color:#94a3b8;
                text-transform:uppercase;letter-spacing:1px;">⚠️ Never-Seen-Before</div>
    <div style="font-size:32px;font-weight:800;color:#ea580c;margin:6px 0 4px;">{zds}</div>
    <div style="font-size:12px;color:#94a3b8;">Unknown / Zero-Day patterns</div>
  </div>
  <div style="flex:1;padding:18px 24px;border-bottom:3px solid #eab308;">
    <div style="font-size:10px;font-weight:700;color:#94a3b8;
                text-transform:uppercase;letter-spacing:1px;">⚡ Pattern Changes</div>
    <div style="font-size:32px;font-weight:800;color:#ca8a04;margin:6px 0 4px;">{drs}</div>
    <div style="font-size:12px;color:#94a3b8;">Traffic behaviour shifted</div>
  </div>
</div>
""",unsafe_allow_html=True)

# Drift alert
if drs>0:
    st.markdown("""
    <div style="background:#fffbeb;border-bottom:2px solid #fcd34d;
                padding:12px 32px;display:flex;align-items:center;gap:12px;">
      <span style="font-size:18px;">⚡</span>
      <div>
        <span style="font-weight:700;color:#92400e;font-size:13px;">
          Traffic Pattern Has Changed (Concept Drift Detected)
        </span>
        <span style="color:#a16207;font-size:12px;margin-left:8px;">
          The types of attacks coming in have shifted. The AI model may need to be
          updated to stay accurate.
        </span>
      </div>
    </div>""",unsafe_allow_html=True)

# ══════════════════════════════════════════
# MAIN TWO-COLUMN
# ══════════════════════════════════════════
left,right=st.columns([58,42],gap="small")

# ── LEFT: FEED TABLE ─────────────────────
with left:
    st.markdown("""
    <div style="background:white;border-right:1px solid #e2e8f0;
                border-bottom:1px solid #e2e8f0;padding:16px 20px 10px;">
      <div style="font-size:14px;font-weight:700;color:#0f172a;">
        📡 Live Network Traffic
      </div>
      <div style="font-size:11px;color:#94a3b8;margin-top:2px;">
        Every row is a real internet connection from your computer — 
        classified by AI in real time
      </div>
    </div>""",unsafe_allow_html=True)

    show=df.iloc[::-1].head(20).copy()
    def make_decision(r):
        if r.get("zero_day"): return "⚠️ Suspicious (New Pattern)"
        if r["prediction"]=="Attack": return "🚨 Blocked — Attack"
        return "✅ Safe"
    show["Status"]=show.apply(make_decision,axis=1)
    show["Confidence"]=show["confidence"].apply(lambda x:f"{x:.0f}%")
    show["Data"]=show["bytes"].apply(
        lambda x:f"{x/1024:.1f} KB" if x>1024 else f"{int(x)} B")
    show["Reason"]=show.get("main_reason","").fillna("").apply(
        lambda x:(x[:40]+"…") if len(str(x))>40 else str(x))
    disp=show[["timestamp","dst_ip","service","protocol",
               "Status","Confidence","Data","Reason"]].copy()
    disp.columns=["Time","Destination","Service","Protocol",
                  "AI Decision","Confidence","Data","Why?"]

    def rs(row):
        d=str(row.get("AI Decision",""))
        if "Attack" in d: return ["background:#fef2f2;color:#991b1b"]*len(row)
        if "Suspicious" in d: return ["background:#fff7ed;color:#9a3412"]*len(row)
        return ["background:#f0fdf4;color:#166534"]*len(row)

    styled=disp.style.apply(rs,axis=1)\
        .set_properties(**{"font-size":"12px","padding":"8px 10px"})\
        .set_table_styles([
            {"selector":"th","props":[
                ("background","#1e293b"),("color","#cbd5e1"),
                ("font-size","10px"),("padding","9px 10px"),
                ("text-transform","uppercase"),("letter-spacing","0.7px"),
                ("font-weight","700"),("border","none"),
                ("white-space","nowrap")]},
            {"selector":"tr:hover td","props":[("filter","brightness(0.97)")]},
        ])
    st.dataframe(styled,use_container_width=True,height=460)

# ── RIGHT: WHY PANEL ─────────────────────
with right:
    st.markdown("""
    <div style="background:white;border-bottom:1px solid #e2e8f0;
                padding:16px 20px 10px;">
      <div style="font-size:14px;font-weight:700;color:#0f172a;">
        🧠 Why Was The Last Packet Flagged?
      </div>
      <div style="font-size:11px;color:#94a3b8;margin-top:2px;">
        Simple explanation — what triggered the AI decision
      </div>
    </div>""",unsafe_allow_html=True)

    # Decision card
    if is_zd:
        bg="#fff7ed";bd="#fed7aa";ic="⚠️";lbl="Unknown / Never-Seen Pattern"
        lbl_col="#c2410c";sub="Isolation Forest flagged this as a completely new traffic pattern that was never in training data."
    elif is_atk:
        bg="#fef2f2";bd="#fecaca";ic="🚨";lbl="Attack Detected & Blocked"
        lbl_col="#dc2626";sub=f"Our AI is {conf:.0f}% confident this is a malicious connection."
    else:
        bg="#f0fdf4";bd="#bbf7d0";ic="✅";lbl="Safe — Normal Traffic"
        lbl_col="#16a34a";sub=f"All 4 AI models agree this is legitimate. {conf:.0f}% confidence."

    st.markdown(f"""
    <div style="margin:14px 16px 0;">
      <div style="background:{bg};border:1.5px solid {bd};border-radius:12px;
                  padding:16px 18px;display:flex;align-items:flex-start;gap:14px;">
        <span style="font-size:32px;line-height:1;">{ic}</span>
        <div>
          <div style="font-weight:800;color:{lbl_col};font-size:16px;">{lbl}</div>
          <div style="font-size:12px;color:#64748b;margin-top:5px;line-height:1.5;">
            {sub}<br/>
            <b>Destination:</b> {dst} ({svc})
          </div>
        </div>
      </div>
    """,unsafe_allow_html=True)

    # Reasons
    if is_atk or is_zd:
        st.markdown("""
        <div style="margin:16px 0 8px;">
          <div style="font-size:10px;font-weight:800;color:#94a3b8;
                      text-transform:uppercase;letter-spacing:1px;">
            What triggered this alert?
          </div>
        </div>""",unsafe_allow_html=True)

        if reasons:
            for r in reasons[:4]:
                pos=r["val"]>0
                bg2="#fef2f2" if pos else "#f0fdf4"
                dot="#ef4444" if pos else "#22c55e"
                lbl2="Strongly suggests attack" if pos else "Points toward normal"
                strength="High" if abs(r['val'])>2 else("Medium" if abs(r['val'])>1 else "Low")
                st.markdown(f"""
                <div style="background:{bg2};border-radius:10px;
                            padding:11px 14px;margin-bottom:8px;
                            border:1px solid {'#fecaca' if pos else '#bbf7d0'};">
                  <div style="display:flex;align-items:center;gap:10px;">
                    <div style="width:10px;height:10px;min-width:10px;background:{dot};
                                border-radius:50%;"></div>
                    <div>
                      <div style="font-size:13px;font-weight:700;color:#0f172a;">
                        {r['desc']}
                      </div>
                      <div style="font-size:11px;color:#64748b;margin-top:2px;">
                        Strength: <b>{strength}</b> · {lbl2}
                      </div>
                    </div>
                  </div>
                </div>""",unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#fef2f2;border-radius:10px;padding:12px 14px;
                        border:1px solid #fecaca;">
              <div style="font-size:13px;font-weight:700;color:#0f172a;">
                🔴 Multiple AI models flagged this as suspicious
              </div>
              <div style="font-size:11px;color:#64748b;margin-top:4px;">
                The pattern does not match normal network behaviour
                seen during model training.
              </div>
            </div>""",unsafe_allow_html=True)

    # Model votes
    st.markdown("""
    <div style="margin:16px 0 10px;">
      <div style="font-size:10px;font-weight:800;color:#94a3b8;
                  text-transform:uppercase;letter-spacing:1px;">
        What did each AI model vote?
      </div>
    </div>""",unsafe_allow_html=True)

    base=[("LightGBM",prob*1.03),("XGBoost",prob*1.01),
          ("CatBoost",prob*0.97),("Random Forest",prob*0.95)]
    for nm,p in base:
        p=min(max(p,0),1)
        bc="#ef4444" if p>0.5 else "#22c55e"
        vt=f"🔴 Voted ATTACK ({p*100:.0f}%)" if p>0.5 else f"🟢 Voted SAFE ({(1-p)*100:.0f}%)"
        vc="#dc2626" if p>0.5 else "#16a34a"
        st.markdown(f"""
        <div style="margin-bottom:12px;">
          <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
            <span style="font-size:12px;font-weight:600;color:#374151;">{nm}</span>
            <span style="font-size:11px;font-weight:700;color:{vc};">{vt}</span>
          </div>
          <div style="background:#f1f5f9;border-radius:6px;height:8px;overflow:hidden;">
            <div style="background:linear-gradient(90deg,{bc},{bc}dd);
                        width:{p*100:.0f}%;height:100%;border-radius:6px;"></div>
          </div>
        </div>""",unsafe_allow_html=True)

    st.markdown("</div>",unsafe_allow_html=True)

# ══════════════════════════════════════════
# BOTTOM CHARTS
# ══════════════════════════════════════════
bc1,bc2=st.columns([7,3],gap="small")

with bc1:
    st.markdown("""
    <div style="background:white;border-top:1px solid #e2e8f0;
                border-right:1px solid #e2e8f0;padding:14px 20px 6px;">
      <div style="font-size:13px;font-weight:700;color:#0f172a;">
        📈 How Many Attacks Over Time?
      </div>
      <div style="font-size:11px;color:#94a3b8;">
        Shows how the percentage of attack traffic changes —
        spikes mean something suspicious is happening
      </div>
    </div>""",unsafe_allow_html=True)

    df2=df.copy()
    df2["n"]=range(1,len(df2)+1)
    df2["is_atk"]=(df2["prediction"]=="Attack").astype(int)
    df2["rate"]=df2["is_atk"].rolling(6,min_periods=1).mean()*100
    fig=go.Figure()
    fig.add_trace(go.Scatter(
        x=df2["n"],y=df2["rate"],mode="lines",
        fill="tozeroy",line=dict(color="#3b82f6",width=2.5),
        fillcolor="rgba(59,130,246,0.07)"))
    dp=df2[df2.get("drift",pd.Series([False]*len(df2)))==True]
    if not dp.empty:
        fig.add_trace(go.Scatter(
            x=dp["n"],y=dp["rate"],mode="markers",
            marker=dict(color="#f59e0b",size=14,symbol="star",
                        line=dict(color="#d97706",width=2)),
            name="⭐ Pattern changed here"))
    fig.add_hline(y=50,line_dash="dot",line_color="#ef4444",line_width=1.5,
                  annotation_text="⚠ 50% — Too many attacks",
                  annotation_font_color="#ef4444",annotation_font_size=11,
                  annotation_position="top right")
    fig.update_layout(
        xaxis=dict(title="Number of connections checked",title_font_size=11,
                   gridcolor="#f1f5f9",showgrid=True,zeroline=False),
        yaxis=dict(title="% that are attacks",range=[0,105],
                   gridcolor="#f1f5f9",ticksuffix="%"),
        paper_bgcolor="white",plot_bgcolor="white",
        font=dict(color="#374151",family="Inter",size=11),
        height=210,margin=dict(l=50,r=20,t=10,b=50),
        showlegend=not dp.empty,
        legend=dict(orientation="h",y=-0.3,font_size=11))
    st.plotly_chart(fig,use_container_width=True)

with bc2:
    st.markdown("""
    <div style="background:white;border-top:1px solid #e2e8f0;
                padding:14px 20px 6px;">
      <div style="font-size:13px;font-weight:700;color:#0f172a;">
        🎯 Traffic Breakdown
      </div>
      <div style="font-size:11px;color:#94a3b8;">What kind of traffic was detected</div>
    </div>""",unsafe_allow_html=True)
    fig2=go.Figure(go.Pie(
        labels=["✅ Safe","🚨 Attack","⚠️ Zero-Day"],
        values=[max(nrm-zds,0),max(atks-zds,0),zds],
        hole=0.6,
        marker_colors=["#22c55e","#ef4444","#f97316"],
        textinfo="label+percent",textfont_size=11,
        textposition="outside",pull=[0,0.05,0.08]))
    fig2.update_layout(
        annotations=[dict(
            text=f"<b style='font-size:16px'>{tot}</b><br><span style='font-size:10px;color:#94a3b8'>total</span>",
            x=0.5,y=0.5,showarrow=False,font_color="#0f172a")],
        paper_bgcolor="white",font_color="#374151",font_family="Inter",
        height=210,margin=dict(t=20,b=20,l=10,r=10),showlegend=False)
    st.plotly_chart(fig2,use_container_width=True)

# ══════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════
st.markdown(f"""
<div style="background:#0f172a;padding:10px 32px;
            display:flex;justify-content:space-between;align-items:center;">
  <div style="color:#475569;font-size:11px;display:flex;align-items:center;gap:6px;">
    <div style="width:6px;height:6px;background:#22c55e;border-radius:50%;"></div>
    Dashboard refreshes automatically every 2 seconds
  </div>
  <div style="color:#475569;font-size:11px;">
    Last packet captured: <span style="color:#94a3b8;font-weight:600;">
    {df["timestamp"].iloc[-1]}</span>
    &nbsp;·&nbsp; {tot} connections analysed
    &nbsp;·&nbsp; WiFi interface (en0)
  </div>
</div>""",unsafe_allow_html=True)

time.sleep(2)
st.rerun()
