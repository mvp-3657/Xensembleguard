"""
=============================================================
STEP 2 — BASE MODELS TRAINING
xEnsembleGuard: SHAP + Concept Drift Extension
=============================================================
Models: LightGBM | XGBoost | CatBoost | GBM | Bagging
        LSTM | GRU
=============================================================
"""

import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# Tree-based
from sklearn.ensemble    import GradientBoostingClassifier, BaggingClassifier
from sklearn.tree        import DecisionTreeClassifier
from lightgbm            import LGBMClassifier
from xgboost             import XGBClassifier
from catboost            import CatBoostClassifier
from sklearn.ensemble    import RandomForestClassifier

# Deep Learning
# from tensorflow.keras.models  import Sequential, load_model
# from tensorflow.keras.layers  import LSTM, GRU, Dense, Dropout, Reshape
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from tensorflow.keras.utils   import to_categorical

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


# =============================================================
# HELPER — Print results neatly
# =============================================================
def print_results(name, y_true, y_pred, class_names):
    acc = accuracy_score(y_true, y_pred)
    p   = precision_score(y_true, y_pred, average="macro", zero_division=0)
    r   = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"\n  {'─'*50}")
    print(f"  {name}")
    print(f"  {'─'*50}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {p:.4f}")
    print(f"  Recall    : {r:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    return {"model": name, "accuracy": acc, "precision": p,
            "recall": r, "f1": f1}


# =============================================================
# TREE-BASED MODELS
# =============================================================
class TreeModels:
    """
    Trains all 5 tree-based models and returns
    their prediction probability matrices for meta-model input.
    """

    def __init__(self, n_classes: int, random_state: int = 42):
        self.n_classes    = n_classes
        self.random_state = random_state
        self.models       = {}
        self.results      = []

    # ----------------------------------------------------------
    def build_models(self):
        """Define all tree-based classifiers."""
        self.models = {

            "LightGBM": LGBMClassifier(
                n_estimators    = 500,
                learning_rate   = 0.05,
                max_depth       = 8,
                num_leaves      = 63,
                colsample_bytree= 0.8,
                subsample       = 0.8,
                min_child_samples= 20,
                random_state    = self.random_state,
                verbose         = -1,
                n_jobs          = -1,
            ),

            "XGBoost": XGBClassifier(
                n_estimators   = 500,
                learning_rate  = 0.05,
                max_depth      = 8,
                subsample      = 0.8,
                colsample_bytree= 0.8,
                min_child_weight= 3,
                use_label_encoder=False,
                eval_metric    = "mlogloss",
                random_state   = self.random_state,
                n_jobs         = -1,
                verbosity      = 0,
            ),

            "CatBoost": CatBoostClassifier(
                iterations     = 500,
                learning_rate  = 0.05,
                depth          = 8,
                random_seed    = self.random_state,
                verbose        = 0,
            ),

            "RandomForest": RandomForestClassifier(
                n_estimators   = 300,
                max_depth      = 12,
                min_samples_leaf = 2,
                random_state   = self.random_state,
                n_jobs         = -1,
            ),
        }
        print(f"  ✅ {len(self.models)} tree-based models defined")

    # ----------------------------------------------------------
    def train_all(self, X_train, y_train, X_test, y_test, class_names):
        """Train all tree models and collect metrics + prediction probs."""
        self.build_models()
        pred_probs_train = []   # Stacked predictions for meta-model
        pred_probs_test  = []

        print(f"\n{'='*55}")
        print(f"  TRAINING TREE-BASED MODELS")
        print(f"{'='*55}")

        for name, model in self.models.items():
            print(f"\n  🌲 Training {name} ...")
            model.fit(X_train, y_train)

            # Test predictions
            y_pred = model.predict(X_test)

            # Probability predictions for meta-model stacking
            if hasattr(model, "predict_proba"):
                p_train = model.predict_proba(X_train)
                p_test  = model.predict_proba(X_test)
            else:
                # Fallback: one-hot of prediction
                p_train = np.eye(self.n_classes)[model.predict(X_train)]
                p_test  = np.eye(self.n_classes)[y_pred]

            pred_probs_train.append(p_train)
            pred_probs_test.append(p_test)

            # Save model
            joblib.dump(model, f"{MODELS_DIR}/{name}.pkl")

            # Record results
            r = print_results(name, y_test, y_pred, class_names)
            self.results.append(r)

        # Stack into matrices: shape (N, n_models * n_classes)
        stacked_train = np.hstack(pred_probs_train)
        stacked_test  = np.hstack(pred_probs_test)
        print(f"\n  ✅ Stacked prediction matrix:")
        print(f"     Train: {stacked_train.shape}")
        print(f"     Test : {stacked_test.shape}")

        return stacked_train, stacked_test, self.results


# =============================================================
# DEEP LEARNING MODELS
# =============================================================
class DeepModels:
    """
    Trains LSTM and GRU models.
    Returns prediction probability matrices for meta-model input.
    """

    def __init__(self, n_classes: int, epochs: int = 30, batch_size: int = 512):
        self.n_classes  = n_classes
        self.epochs     = epochs
        self.batch_size = batch_size
        self.models     = {}
        self.results    = []

    # ----------------------------------------------------------
    def _build_lstm(self, input_dim):
        model = Sequential([
            Reshape((1, input_dim), input_shape=(input_dim,)),
            LSTM(128, return_sequences=False),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(self.n_classes, activation="softmax"),
        ])
        model.compile(
            optimizer = "adam",
            loss      = "categorical_crossentropy",
            metrics   = ["accuracy"]
        )
        return model

    # ----------------------------------------------------------
    def _build_gru(self, input_dim):
        model = Sequential([
            Reshape((1, input_dim), input_shape=(input_dim,)),
            GRU(128, return_sequences=False),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(self.n_classes, activation="softmax"),
        ])
        model.compile(
            optimizer = "adam",
            loss      = "categorical_crossentropy",
            metrics   = ["accuracy"]
        )
        return model

    # ----------------------------------------------------------
    def train_all(self, X_train, y_train, X_test, y_test, class_names):
        """Train LSTM and GRU, return stacked prediction matrices."""
        input_dim = X_train.shape[1]
        y_train_cat = to_categorical(y_train, self.n_classes)
        y_test_cat  = to_categorical(y_test,  self.n_classes)

        builders = {
            "LSTM": self._build_lstm,
            "GRU":  self._build_gru,
        }
        pred_probs_train = []
        pred_probs_test  = []

        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(patience=3, factor=0.5, verbose=0),
        ]

        print(f"\n{'='*55}")
        print(f"  TRAINING DEEP LEARNING MODELS")
        print(f"{'='*55}")

        for name, build_fn in builders.items():
            print(f"\n  🧠 Training {name} ...")
            model = build_fn(input_dim)
            model.fit(
                X_train, y_train_cat,
                validation_split = 0.1,
                epochs           = self.epochs,
                batch_size       = self.batch_size,
                callbacks        = callbacks,
                verbose          = 1,
            )

            # Predictions
            p_train = model.predict(X_train, verbose=0)
            p_test  = model.predict(X_test,  verbose=0)
            y_pred  = np.argmax(p_test, axis=1)

            pred_probs_train.append(p_train)
            pred_probs_test.append(p_test)

            # Save model
            model.save(f"{MODELS_DIR}/{name}.h5")

            r = print_results(name, y_test, y_pred, class_names)
            self.results.append(r)

        stacked_train = np.hstack(pred_probs_train)
        stacked_test  = np.hstack(pred_probs_test)
        print(f"\n  ✅ DL Stacked prediction matrix:")
        print(f"     Train: {stacked_train.shape}")
        print(f"     Test : {stacked_test.shape}")

        return stacked_train, stacked_test, self.results


# =============================================================
# COMBINE ALL BASE MODEL PREDICTIONS
# =============================================================
def combine_predictions(tree_train, tree_test, dl_train, dl_test):
    """
    Concatenate tree + DL prediction matrices.
    This becomes the input to the meta-model.
    """
    meta_train = np.hstack([tree_train, dl_train])
    meta_test  = np.hstack([tree_test,  dl_test])
    print(f"\n  ✅ FINAL META-MODEL INPUT:")
    print(f"     Train: {meta_train.shape}  "
          f"({tree_train.shape[1]} tree + {dl_train.shape[1]} DL features)")
    print(f"     Test : {meta_test.shape}")
    return meta_train, meta_test


# =============================================================
# QUICK TEST
# =============================================================
if __name__ == "__main__":
    # Smoke test with dummy data
    X = np.random.randn(1000, 50)
    y = np.random.randint(0, 4, 1000)
    class_names = ["Normal", "DoS", "Probe", "R2L"]

    print("Testing tree models ...")
    tree = TreeModels(n_classes=4)
    tree_tr, tree_te, _ = tree.train_all(X[:800], y[:800], X[800:], y[800:], class_names)

    print("\nTesting deep models ...")
    # deep = DeepModels(n_classes=4, epochs=2)
    # dl_tr, dl_te, _ = deep.train_all(X[:800], y[:800], X[800:], y[800:], class_names)

    # meta_tr, meta_te = combine_predictions(tree_tr, tree_te, dl_tr, dl_te)
    print("\n Tree base models working correctly ✅")