"""
=============================================================
STEP 1 — DATA PREPROCESSING
xEnsembleGuard: SHAP + Concept Drift Extension
=============================================================
Handles: UNSW-NB15 | NSL-KDD | CIC-IDS-2017
Author : [Your Name]
=============================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import warnings
import os
warnings.filterwarnings("ignore")


# =============================================================
# CONFIGURATION — Change paths to your dataset locations
# =============================================================
DATA_CONFIG = {
    "UNSW_NB15": {
        "train": "data/UNSW_NB15_training-set.csv",
        "test":  "data/UNSW_NB15_testing-set.csv",
        "label_col": "label",
        "attack_col": "attack_cat",
    },
    "NSL_KDD": {
        "train": "data/KDDTrain+.txt",
        "test":  "data/KDDTest+.txt",
        "label_col": "label",
        "attack_col": "attack_type",
    },
    "CIC_IDS_2017": {
        "path":  "data/CIC-IDS-2017.csv",
        "label_col": "Label",
        "attack_col": "Label",
    }
}

# =============================================================
# NSL-KDD Column Names (no headers in raw file)
# =============================================================
NSL_KDD_COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"
]

NSL_KDD_ATTACK_MAP = {
    "normal":     "Normal",
    "neptune":    "DoS",    "back":        "DoS",
    "land":       "DoS",    "pod":         "DoS",
    "smurf":      "DoS",    "teardrop":    "DoS",
    "mailbomb":   "DoS",    "apache2":     "DoS",
    "processtable":"DoS",   "udpstorm":    "DoS",
    "ipsweep":    "Probe",  "nmap":        "Probe",
    "portsweep":  "Probe",  "satan":       "Probe",
    "mscan":      "Probe",  "saint":       "Probe",
    "ftp_write":  "R2L",    "guess_passwd":"R2L",
    "imap":       "R2L",    "multihop":    "R2L",
    "phf":        "R2L",    "spy":         "R2L",
    "warezclient":"R2L",    "warezmaster": "R2L",
    "buffer_overflow":"U2R","loadmodule":  "U2R",
    "perl":       "U2R",    "rootkit":     "U2R",
    "ps":         "U2R",    "sqlattack":   "U2R",
    "xterm":      "U2R",    "httptunnel":  "R2L",
    "named":      "R2L",    "sendmail":    "R2L",
    "snmpgetattack":"R2L",  "snmpguess":   "R2L",
    "worm":       "U2R",    "xlock":       "R2L",
    "xsnoop":     "R2L",
}


# =============================================================
# CLASS: DataPreprocessor
# =============================================================
class DataPreprocessor:
    """
    Handles loading, cleaning, encoding, and splitting
    for UNSW-NB15, NSL-KDD, and CIC-IDS-2017 datasets.
    """

    def __init__(self, dataset_name: str):
        assert dataset_name in DATA_CONFIG, \
            f"Dataset must be one of {list(DATA_CONFIG.keys())}"
        self.dataset_name = dataset_name
        self.scaler       = StandardScaler()
        self.imputer      = SimpleImputer(strategy="mean")
        self.le           = LabelEncoder()
        self.feature_names = None

    # ----------------------------------------------------------
    def load(self):
        print(f"\n{'='*55}")
        print(f"  Loading {self.dataset_name}")
        print(f"{'='*55}")

        if self.dataset_name == "UNSW_NB15":
            return self._load_unsw()
        elif self.dataset_name == "NSL_KDD":
            return self._load_nslkdd()
        elif self.dataset_name == "CIC_IDS_2017":
            return self._load_cicids()

    # ----------------------------------------------------------
    def _load_unsw(self):
        cfg = DATA_CONFIG["UNSW_NB15"]
        df_train = pd.read_csv(cfg["train"])
        df_test  = pd.read_csv(cfg["test"])
        df = pd.concat([df_train, df_test], ignore_index=True)
        print(f"  Total records : {len(df):,}")
        print(f"  Columns       : {df.shape[1]}")

        # Remove ID columns
        drop_cols = ["id"] if "id" in df.columns else []
        df.drop(columns=drop_cols, inplace=True, errors="ignore")

        # Extract labels (binary + multiclass)
        y_binary = df[cfg["label_col"]].astype(int)
        y_multi  = df[cfg["attack_col"]].fillna("Normal")
        df.drop(columns=[cfg["label_col"], cfg["attack_col"]], inplace=True, errors="ignore")

        return self._clean_and_split(df, y_binary, y_multi)

    # ----------------------------------------------------------
    def _load_nslkdd(self):
        cfg = DATA_CONFIG["NSL_KDD"]
        df_train = pd.read_csv(cfg["train"], names=NSL_KDD_COLUMNS, header=None)
        df_test  = pd.read_csv(cfg["test"],  names=NSL_KDD_COLUMNS, header=None)
        df = pd.concat([df_train, df_test], ignore_index=True)
        print(f"  Total records : {len(df):,}")

        # Map raw labels → attack categories
        df["attack_cat"] = df["label"].str.strip(".").map(NSL_KDD_ATTACK_MAP).fillna("Other")
        y_binary = (df["label"].str.strip(".") != "normal").astype(int)
        y_multi  = df["attack_cat"]
        df.drop(columns=["label", "difficulty", "attack_cat"], inplace=True, errors="ignore")

        return self._clean_and_split(df, y_binary, y_multi)

    # ----------------------------------------------------------
    def _load_cicids(self):
        cfg = DATA_CONFIG["CIC_IDS_2017"]
        df = pd.read_csv(cfg["path"])
        print(f"  Total records : {len(df):,}")

        # CIC-IDS-2017 has spaces in column names
        df.columns = df.columns.str.strip()
        y_multi  = df[cfg["label_col"]].str.strip()
        y_binary = (y_multi != "BENIGN").astype(int)
        df.drop(columns=[cfg["label_col"]], inplace=True, errors="ignore")

        return self._clean_and_split(df, y_binary, y_multi)

    # ----------------------------------------------------------
    def _clean_and_split(self, df, y_binary, y_multi):
        print(f"\n  [1] Handling missing values ...")
        # Replace inf values with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Separate categorical and numerical columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"       Numerical cols  : {len(num_cols)}")
        print(f"       Categorical cols: {len(cat_cols)}")

        # Encode categorical features
        print(f"  [2] Label encoding categorical features ...")
        for col in cat_cols:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

        # Impute missing values
        print(f"  [3] Imputing missing values ...")
        df_imputed = pd.DataFrame(
            self.imputer.fit_transform(df),
            columns=df.columns
        )

        # Outlier clipping (3 sigma)
        print(f"  [4] Clipping outliers (3-sigma) ...")
        for col in df_imputed.columns:
            mu = df_imputed[col].mean()
            sigma = df_imputed[col].std()
            df_imputed[col] = df_imputed[col].clip(mu - 3*sigma, mu + 3*sigma)

        # Encode multi-class labels
        print(f"  [5] Encoding labels ...")
        y_multi_encoded = self.le.fit_transform(y_multi.astype(str))
        class_names = list(self.le.classes_)
        print(f"       Attack classes: {class_names}")

        # Train / test split (80/20 stratified)
        print(f"  [6] Splitting 80% train / 20% test ...")
        X_train, X_test, y_train_b, y_test_b, y_train_m, y_test_m = \
            train_test_split(
                df_imputed.values, y_binary.values, y_multi_encoded,
                test_size=0.2, random_state=42, stratify=y_multi_encoded
            )

        # Standard scaling
        print(f"  [7] Standard scaling ...")
        X_train = self.scaler.fit_transform(X_train)
        X_test  = self.scaler.transform(X_test)
        self.feature_names = df_imputed.columns.tolist()

        print(f"\n  ✅ Done!")
        print(f"     Train shape : {X_train.shape}")
        print(f"     Test shape  : {X_test.shape}")
        print(f"     Classes     : {len(class_names)}")

        return {
            "X_train":      X_train,
            "X_test":       X_test,
            "y_train_bin":  y_train_b,
            "y_test_bin":   y_test_b,
            "y_train_multi":y_train_m,
            "y_test_multi": y_test_m,
            "class_names":  class_names,
            "feature_names":self.feature_names,
            "scaler":       self.scaler,
            "label_encoder":self.le,
        }

    # ----------------------------------------------------------
    def _clean_and_split_official(self,
                                   df_train, df_test,
                                   y_train_bin, y_test_bin,
                                   y_train_multi, y_test_multi):
        """
        Process official train/test CSV files WITHOUT random re-splitting.
        Fits preprocessors on train only — correct evaluation practice.
        """
        def _preprocess(df):
            df = df.copy()
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            return df

        print(f"\n  [1] Handling missing values ...")
        df_train = _preprocess(df_train)
        df_test  = _preprocess(df_test)

        cat_cols = df_train.select_dtypes(include=["object","category"]).columns.tolist()
        num_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
        print(f"       Numerical cols  : {len(num_cols)}")
        print(f"       Categorical cols: {len(cat_cols)}")

        print(f"  [2] Label encoding categorical features ...")
        for col in cat_cols:
            le_col = LabelEncoder()
            df_train[col] = le_col.fit_transform(df_train[col].astype(str))
            mapping = {c: i for i, c in enumerate(le_col.classes_)}
            df_test[col] = df_test[col].astype(str).map(mapping).fillna(-1).astype(int)

        print(f"  [3] Imputing missing values ...")
        df_train_imp = pd.DataFrame(
            self.imputer.fit_transform(df_train), columns=df_train.columns)
        df_test_imp  = pd.DataFrame(
            self.imputer.transform(df_test),      columns=df_test.columns)

        print(f"  [4] Clipping outliers (3-sigma) ...")
        for col in df_train_imp.columns:
            mu, sigma = df_train_imp[col].mean(), df_train_imp[col].std()
            df_train_imp[col] = df_train_imp[col].clip(mu - 3*sigma, mu + 3*sigma)
            df_test_imp[col]  = df_test_imp[col].clip( mu - 3*sigma, mu + 3*sigma)

        print(f"  [5] Encoding labels ...")
        y_train_m = self.le.fit_transform(y_train_multi.astype(str))
        class_names = list(self.le.classes_)
        mapping_mc = {c: i for i, c in enumerate(self.le.classes_)}
        y_test_m = np.array([mapping_mc.get(str(v), 0) for v in y_test_multi])
        print(f"       Attack classes : {class_names}")

        print(f"  [6] Using official train/test split (no re-split) ...")
        X_train   = df_train_imp.values
        X_test    = df_test_imp.values
        y_train_b = y_train_bin.values
        y_test_b  = y_test_bin.values

        print(f"  [7] Standard scaling ...")
        X_train = self.scaler.fit_transform(X_train)
        X_test  = self.scaler.transform(X_test)
        self.feature_names = df_train_imp.columns.tolist()

        print(f"\n  ✅ Done!")
        print(f"     Train shape : {X_train.shape}")
        print(f"     Test shape  : {X_test.shape}")
        print(f"     Classes     : {len(class_names)}")

        return {
            "X_train":       X_train,
            "X_test":        X_test,
            "y_train_bin":   y_train_b,
            "y_test_bin":    y_test_b,
            "y_train_multi": y_train_m,
            "y_test_multi":  y_test_m,
            "class_names":   class_names,
            "feature_names": self.feature_names,
            "scaler":        self.scaler,
            "label_encoder": self.le,
        }


# =============================================================
# QUICK TEST — run this file directly to verify loading
# =============================================================
if __name__ == "__main__":
    # Test with UNSW-NB15 (change to your dataset)
    prep = DataPreprocessor("UNSW_NB15")
    data = prep.load()
    print("\nFeature names sample:", data["feature_names"][:5])
    print("Class names:", data["class_names"])