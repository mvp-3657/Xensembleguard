"""
Real-Time Packet Capture Engine
Run: sudo python packet_capture.py --iface en0
"""
import numpy as np, pandas as pd, joblib, os, time, warnings, threading, json
from datetime import datetime
from river.drift import ADWIN
import shap
warnings.filterwarnings("ignore")

from scapy.all import sniff, IP, TCP, UDP, ICMP

MODELS_DIR = "models"
RESULTS_DIR= "results"
LIVE_CSV   = f"{RESULTS_DIR}/live_packets.csv"
os.makedirs(RESULTS_DIR, exist_ok=True)
if os.path.exists(LIVE_CSV): os.remove(LIVE_CSV)

PROTO_MAP   = {"tcp":6,"udp":17,"icmp":1,"other":0}
SERVICE_MAP = {80:"HTTP Web",443:"HTTPS (Secure Web)",22:"SSH (Remote Login)",
               21:"FTP (File Transfer)",25:"SMTP (Email)",53:"DNS (Domain Lookup)",
               3306:"MySQL Database",8080:"HTTP Alternative",3389:"Remote Desktop",
               110:"POP3 Email",143:"IMAP Email",0:"Unknown"}

FEATURE_DESC = {
    "dur":          "Connection duration",
    "sbytes":       "Data sent (bytes)",
    "dbytes":       "Data received (bytes)",
    "rate":         "Packet rate",
    "sttl":         "Source TTL value",
    "dttl":         "Destination TTL value",
    "sload":        "Source bandwidth load",
    "dload":        "Destination bandwidth load",
    "spkts":        "Source packet count",
    "dpkts":        "Destination packet count",
    "sinpkt":       "Source inter-packet time",
    "dinpkt":       "Destination inter-packet time",
    "swin":         "Source TCP window size",
    "dwin":         "Destination TCP window size",
    "smean":        "Avg source packet size",
    "dmean":        "Avg destination packet size",
    "tcprtt":       "TCP round-trip time",
    "LightGBM":     "LightGBM model vote",
    "XGBoost":      "XGBoost model vote",
    "CatBoost":     "CatBoost model vote",
    "RandomForest": "Random Forest model vote",
}

class Flow:
    def __init__(self, key, pkt):
        self.key=key; self.start=float(pkt.time); self.last=float(pkt.time)
        self.spkts=0; self.dpkts=0; self.sbytes=0; self.dbytes=0
        self.sttl=0; self.dttl=0; self.swin=0; self.dwin=0
        self.s_times=[]; self.d_times=[]; self.s_sizes=[]; self.d_sizes=[]
        self.syn_ts=None; self.synack_ts=None; self.ack_ts=None; self.state="INT"
        src,dst,sp,dp,proto=key
        self.src_ip=src; self.dst_ip=dst; self.sport=sp; self.dport=dp; self.proto=proto

    def add(self, pkt, direction="src"):
        ts=float(pkt.time); sz=len(pkt); self.last=ts
        if direction=="src":
            self.spkts+=1; self.sbytes+=sz; self.s_times.append(ts); self.s_sizes.append(sz)
            if IP in pkt: self.sttl=pkt[IP].ttl
            if TCP in pkt:
                self.swin=pkt[TCP].window; f=str(pkt[TCP].flags)
                if "S" in f and "A" not in f: self.syn_ts=ts; self.state="REQ"
                elif "F" in f: self.state="FIN"
                elif "R" in f: self.state="RST"
                elif "A" in f and self.synack_ts: self.ack_ts=ts; self.state="CON"
        else:
            self.dpkts+=1; self.dbytes+=sz; self.d_times.append(ts); self.d_sizes.append(sz)
            if IP in pkt: self.dttl=pkt[IP].ttl
            if TCP in pkt:
                self.dwin=pkt[TCP].window; f=str(pkt[TCP].flags)
                if "S" in f and "A" in f: self.synack_ts=ts

    def features(self):
        dur=max(self.last-self.start,1e-6)
        def ipt(t): return float(np.mean(np.diff(sorted(t)))) if len(t)>1 else 0
        def jit(t): return float(np.std(np.diff(sorted(t)))) if len(t)>2 else 0
        syn=ack=rtt=0
        if self.syn_ts and self.synack_ts: syn=self.synack_ts-self.syn_ts
        if self.synack_ts and self.ack_ts: ack=self.ack_ts-self.synack_ts
        if syn and ack: rtt=syn+ack
        svc=0
        for p,s in [(80,1),(443,2),(22,3),(21,4),(25,5),(53,6),(3306,7),(8080,8)]:
            if self.dport==p or self.sport==p: svc=p; break
        f=np.zeros(42,dtype=np.float32)
        f[0]=dur; f[1]=PROTO_MAP.get(self.proto,0); f[2]=float(svc)
        f[3]={"INT":1,"CON":2,"REQ":3,"FIN":0,"RST":4}.get(self.state,1)
        f[4]=self.spkts; f[5]=self.dpkts; f[6]=self.sbytes; f[7]=self.dbytes
        f[8]=(self.spkts+self.dpkts)/dur; f[9]=float(self.sttl); f[10]=float(self.dttl)
        f[11]=self.sbytes*8/dur; f[12]=self.dbytes*8/dur
        f[15]=ipt(self.s_times); f[16]=ipt(self.d_times)
        f[17]=jit(self.s_times); f[18]=jit(self.d_times)
        f[19]=float(self.swin); f[22]=float(self.dwin)
        f[23]=rtt; f[24]=syn; f[25]=ack
        f[26]=float(np.mean(self.s_sizes)) if self.s_sizes else 0
        f[27]=float(np.mean(self.d_sizes)) if self.d_sizes else 0
        f[41]=1.0 if self.sport==self.dport else 0.0
        return f

class CaptureEngine:
    def __init__(self, iface="en0", timeout=5.0):
        self.iface=iface; self.timeout=timeout
        self.flows={}; self.lock=threading.Lock()
        self.running=False; self.adwin=ADWIN(delta=0.002)
        self.stats=dict(total=0,attacks=0,normal=0,zeroday=0,drifts=0)

        print(f"\n{'='*55}\n  🔄 LOADING MODELS\n{'='*55}")
        self.models={}
        for n,f in [("LightGBM","LightGBM.pkl"),("XGBoost","XGBoost.pkl"),
                    ("CatBoost","CatBoost.pkl"),("RandomForest","RandomForest.pkl"),
                    ("MetaModel","meta_model.pkl"),
                    ("IsolationForest","isolation_forest.pkl"),("Scaler","scaler.pkl")]:
            p=f"{MODELS_DIR}/{f}"
            if os.path.exists(p): self.models[n]=joblib.load(p); print(f"  ✅ {n}")
        self.BASE=["LightGBM","XGBoost","CatBoost","RandomForest"]

        # Setup SHAP explainer
        print("  🔄 Setting up SHAP explainer...")
        try:
            from data_preprocessing import DataPreprocessor
            d=DataPreprocessor("UNSW_NB15").load()
            X_bg=d["X_test"][:100]
            _p=[self.models[m].predict_proba(X_bg) for m in self.BASE if m in self.models]
            _bg=np.hstack([np.hstack(_p),X_bg])
            self.explainer=shap.TreeExplainer(self.models["MetaModel"],_bg)
            self.feat_names=(
                [f"{m}→P(Attack)" for m in self.BASE]+
                [f"feat_{i}" for i in range(42)]
            )
            print("  ✅ SHAP explainer ready")
        except Exception as e:
            self.explainer=None; print(f"  ⚠️  SHAP unavailable: {e}")

    def _pkt(self, pkt):
        if IP not in pkt: return
        ip=pkt[IP]; proto="other"; sp=dp=0
        if TCP in pkt:   proto="tcp"; sp,dp=pkt[TCP].sport,pkt[TCP].dport
        elif UDP in pkt: proto="udp"; sp,dp=pkt[UDP].sport,pkt[UDP].dport
        elif ICMP in pkt: proto="icmp"
        fk=(ip.src,ip.dst,sp,dp,proto); rk=(ip.dst,ip.src,dp,sp,proto)
        with self.lock:
            if fk in self.flows:   self.flows[fk].add(pkt,"src")
            elif rk in self.flows: self.flows[rk].add(pkt,"dst")
            else:
                fl=Flow(fk,pkt); fl.add(pkt,"src"); self.flows[fk]=fl

    def _export(self):
        while self.running:
            time.sleep(1); now=time.time(); to_exp=[]
            with self.lock:
                exp=[k for k,f in self.flows.items()
                     if now-f.last>self.timeout and f.spkts>=1]
                for k in exp: to_exp.append(self.flows.pop(k))
            for fl in to_exp: self._predict(fl)

    def _predict(self, fl):
        try:
            feats=fl.features()
            if "Scaler" in self.models:
                feats=self.models["Scaler"].transform(feats.reshape(1,-1))[0]
            X=feats.reshape(1,-1)
            iso=(self.models["IsolationForest"].predict(X)[0]==-1
                 if "IsolationForest" in self.models else False)
            probas=[self.models[m].predict_proba(X) for m in self.BASE if m in self.models]
            if not probas: return
            meta_in=np.hstack([np.hstack(probas),X])
            pred=int(self.models["MetaModel"].predict(meta_in)[0])
            proba=self.models["MetaModel"].predict_proba(meta_in)[0]
            self.adwin.update(1-pred)
            drift=self.adwin.drift_detected
            if drift: self.stats["drifts"]+=1
            self.stats["total"]+=1
            if pred==1: self.stats["attacks"]+=1
            else:       self.stats["normal"]+=1
            if iso:     self.stats["zeroday"]+=1

            svc=SERVICE_MAP.get(fl.dport,SERVICE_MAP.get(fl.sport,f"Port {fl.dport}"))
            label="Attack" if pred==1 else "Normal"
            ts=datetime.now().strftime("%H:%M:%S")
            conf=float(max(proba))*100
            icon="⚠️" if iso else "🚨" if pred==1 else "✅"

            # SHAP explanation
            shap_reason=""; shap_json="[]"
            if pred==1 and self.explainer:
                try:
                    sv=self.explainer.shap_values(meta_in)
                    v=sv[1][0] if isinstance(sv,list) else (sv[0,:,-1] if sv.ndim==3 else sv[0])
                    top_idx=np.argsort(np.abs(v))[::-1][:4]
                    reasons=[]
                    for i in top_idx:
                        fname=self.feat_names[i] if i<len(self.feat_names) else f"feat_{i}"
                        val=float(v[i])
                        # Map to plain English
                        if "LightGBM" in fname:   desc="LightGBM model flagged as attack"
                        elif "XGBoost" in fname:  desc="XGBoost model flagged as attack"
                        elif "CatBoost" in fname: desc="CatBoost model flagged as attack"
                        elif "RandomForest" in fname: desc="Random Forest flagged as attack"
                        elif i==0: desc="Unusual connection duration"
                        elif i==6: desc="Abnormal data sent"
                        elif i==7: desc="Abnormal data received"
                        elif i==8: desc="Suspicious packet rate"
                        elif i==11: desc="High network load from source"
                        elif i==4: desc="Too many source packets"
                        else: desc=f"Feature {i} anomaly"
                        direction="→ ATTACK" if val>0 else "→ NORMAL"
                        reasons.append({"desc":desc,"val":round(val,3),"dir":direction})
                    shap_reason=reasons[0]["desc"] if reasons else ""
                    shap_json=json.dumps(reasons)
                except: pass

            print(f"[{ts}] {icon} {label:<8} {fl.src_ip:>15}→{fl.dst_ip:<16} "
                  f"{svc:<20} {conf:.0f}% {'⚡DRIFT' if drift else ''}")

            row=pd.DataFrame([{
                "timestamp":ts,"src_ip":fl.src_ip,"dst_ip":fl.dst_ip,
                "sport":fl.sport,"dport":fl.dport,"protocol":fl.proto.upper(),
                "service":svc,"prediction":label,"confidence":round(conf,1),
                "prob_attack":round(float(proba[1]),3),
                "zero_day":iso,"drift":drift,
                "duration":round(fl.last-fl.start,3),
                "packets":fl.spkts+fl.dpkts,"bytes":fl.sbytes+fl.dbytes,
                "main_reason":shap_reason,"shap_json":shap_json
            }])
            row.to_csv(LIVE_CSV,mode="a",
                       header=not os.path.exists(LIVE_CSV) or os.path.getsize(LIVE_CSV)==0,
                       index=False)
        except Exception as e:
            pass

    def start(self):
        self.running=True
        threading.Thread(target=self._export,daemon=True).start()
        print(f"\n{'='*55}")
        print(f"  🚀 LIVE CAPTURE STARTED — Interface: {self.iface}")
        print(f"  Saving to: {LIVE_CSV}")
        print(f"  Dashboard: http://localhost:8503")
        print(f"  Browse any website to generate traffic!")
        print(f"  Ctrl+C to stop")
        print(f"{'='*55}\n")
        try:
            sniff(iface=self.iface,prn=self._pkt,store=False,filter="ip")
        except KeyboardInterrupt:
            self.running=False
            print(f"\n\n✅ Stopped. Stats: {self.stats}")

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--iface",default="en0")
    ap.add_argument("--timeout",type=float,default=4.0)
    args=ap.parse_args()
    CaptureEngine(iface=args.iface,timeout=args.timeout).start()
