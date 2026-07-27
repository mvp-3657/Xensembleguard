"""
Real-Time IDS using psutil (NO sudo required)
Monitors live network connections from your Mac
and classifies them using the trained ensemble model.
Run: python live_ids_nosudo.py
"""
import psutil, numpy as np, pandas as pd
import joblib, os, time, warnings
from datetime import datetime
from river.drift import ADWIN
warnings.filterwarnings("ignore")

MODELS_DIR = "models"

print("\n" + "="*60)
print("  🛡️  REAL-TIME IDS — Live Network Monitor")
print("  Drift-Aware Explainable Ensemble Learning")
print("  NO SUDO REQUIRED — Uses psutil")
print("="*60)

# Load models
models = {}
for n,f in [("LightGBM","LightGBM.pkl"),("XGBoost","XGBoost.pkl"),
            ("CatBoost","CatBoost.pkl"),("RandomForest","RandomForest.pkl"),
            ("MetaModel","meta_model.pkl"),
            ("IsolationForest","isolation_forest.pkl"),
            ("Scaler","scaler.pkl")]:
    p = f"{MODELS_DIR}/{f}"
    if os.path.exists(p):
        models[n] = joblib.load(p)
        print(f"  ✅ Loaded {n}")

BASE     = ["LightGBM","XGBoost","CatBoost","RandomForest"]
adwin    = ADWIN(delta=0.002)
seen     = set()
results  = []
stats    = dict(total=0, attacks=0, normal=0, zeroday=0, drifts=0)

COMMON_PORTS = {80:"HTTP",443:"HTTPS",22:"SSH",21:"FTP",
                25:"SMTP",53:"DNS",3306:"MySQL",
                3389:"RDP",8080:"HTTP-Alt"}

def classify_connection(conn):
    """Extract features from a live connection and classify it."""
    try:
        laddr = conn.laddr
        raddr = conn.raddr if conn.raddr else None
        if not raddr: return None

        sport = laddr.port
        dport = raddr.port
        proto_num = 6 if conn.type.name == "SOCK_STREAM" else 17

        # Service from port
        service = 0
        for p,s in [(80,1),(443,2),(22,3),(21,4),(25,5),(53,6)]:
            if dport==p or sport==p: service=p; break

        # Get process info
        try:
            proc = psutil.Process(conn.pid) if conn.pid else None
            cpu  = proc.cpu_percent() if proc else 0
            mem  = proc.memory_percent() if proc else 0
        except: cpu = mem = 0

        # Build 42-feature vector (approx UNSW-NB15 mapping)
        state_map = {"ESTABLISHED":2,"SYN_SENT":3,"CLOSE_WAIT":0,
                     "TIME_WAIT":0,"LISTEN":1,"NONE":1}
        state_num = state_map.get(str(conn.status),1)

        feats = np.zeros(42, dtype=np.float32)
        feats[0]  = 1.0                    # dur
        feats[1]  = proto_num              # proto
        feats[2]  = float(service)        # service
        feats[3]  = float(state_num)      # state
        feats[4]  = 1.0                   # spkts
        feats[5]  = 1.0                   # dpkts
        feats[6]  = float(min(dport,65535))# sbytes proxy
        feats[7]  = float(min(sport,65535))# dbytes proxy
        feats[8]  = float(dport)/65535    # rate proxy
        feats[9]  = 64.0                  # sttl (typical)
        feats[10] = 64.0                  # dttl
        feats[11] = float(dport)*8        # sload
        feats[12] = float(sport)*8        # dload
        feats[26] = float(dport)          # smean
        feats[27] = float(sport)          # dmean
        feats[30] = float(len([c for c in psutil.net_connections()
                               if c.raddr and c.raddr.port==dport]))
        feats[40] = float(len([c for c in psutil.net_connections()
                               if c.raddr and hasattr(c.raddr,'ip') and
                               c.raddr.ip == raddr.ip]))
        feats[41] = 1.0 if sport==dport else 0.0

        # Scale
        if "Scaler" in models:
            feats = models["Scaler"].transform(feats.reshape(1,-1))[0]

        return feats, raddr, sport, dport, proto_num

    except Exception as e:
        return None

print(f"\n  {'Time':>10} {'Decision':>10} {'Src Port':>9} "
      f"{'Remote IP':<18} {'Port':<6} {'Service':<8} {'Conf':>6}")
print(f"  {'-'*70}")

try:
    while True:
        conns = psutil.net_connections(kind="inet")
        for conn in conns:
            if not conn.raddr: continue
            key = (conn.laddr.port,
                   conn.raddr.ip if conn.raddr else "",
                   conn.raddr.port if conn.raddr else 0)
            if key in seen: continue
            seen.add(key)

            res = classify_connection(conn)
            if res is None: continue
            feats, raddr, sport, dport, proto_num = res

            X = feats.reshape(1,-1)

            # Zero-Day
            is_zd = False
            if "IsolationForest" in models:
                is_zd = models["IsolationForest"].predict(X)[0] == -1

            # Ensemble
            probas  = [models[m].predict_proba(X)
                       for m in BASE if m in models]
            if not probas: continue
            stacked = np.hstack(probas)
            meta_in = np.hstack([stacked, X])
            pred    = int(models["MetaModel"].predict(meta_in)[0])
            proba   = models["MetaModel"].predict_proba(meta_in)[0]

            # ADWIN
            adwin.update(1-pred)
            drift = adwin.drift_detected

            ts      = datetime.now().strftime("%H:%M:%S")
            label   = "Attack" if pred==1 else "Normal"
            conf    = float(max(proba))*100
            service = COMMON_PORTS.get(dport, f"Port-{dport}")
            remote  = raddr.ip if raddr else "?"

            stats["total"] += 1
            if pred==1:      stats["attacks"] += 1
            else:            stats["normal"]  += 1
            if is_zd:        stats["zeroday"] += 1
            if drift:        stats["drifts"]  += 1

            icon = ("⚠️ " if is_zd else
                    "🚨" if pred==1 else "✅")
            drift_f = " ⚡DRIFT!" if drift else ""
            print(f"  [{ts}] {icon} {label:<8} "
                  f":{sport:<7} → {remote:<18} :{dport:<5} "
                  f"{service:<10} {conf:.0f}%{drift_f}")

            results.append({
                "time":ts,"prediction":label,
                "remote_ip":remote,"dport":dport,
                "service":service,"confidence":conf,
                "zero_day":is_zd,"drift":drift
            })

        # Print stats every 30 seconds
        if stats["total"] > 0 and stats["total"] % 20 == 0:
            print(f"\n  ── Stats: {stats} ──\n")

        time.sleep(2)  # check every 2 seconds

except KeyboardInterrupt:
    print(f"\n\n{'='*60}")
    print(f"  ✅ STOPPED — Final Results")
    print(f"{'='*60}")
    for k,v in stats.items():
        print(f"  {k:<12} : {v}")
    if results:
        pd.DataFrame(results).to_csv(
            "results/live_monitor_log.csv", index=False)
        print(f"  📄 Saved → results/live_monitor_log.csv")
