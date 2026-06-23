"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                                                                                  ║
║   PneumoFusion XAI — ONE COMPLETE FILE                                          ║
║   XGBoost (Vitals) + ResNet50 (X-Ray) + Fusion + Grad-CAM + SHAP              ║
║                                                                                  ║
║   Run Commands:                                                                  ║
║   ─────────────────────────────────────────────────────────────────             ║
║   STEP 1 — Train both models + fusion:                                          ║
║   python pneumofusion_xai.py --mode train                                       ║
║       --data_dir "chest-xray-pneumonia/chest_xray"                              ║
║       --vitals_csv human_vital_signs_dataset_2024.csv                           ║
║                                                                                  ║
║   STEP 2 — Predict one patient:                                                 ║
║   python pneumofusion_xai.py --mode predict                                     ║
║       --image path/to/xray.jpeg                                                 ║
║       --heart_rate 102 --body_temperature 38.5                                  ║
║       --oxygen_saturation 94 --respiratory_rate 22                              ║
║       --systolic_bp 125 --diastolic_bp 82                                       ║
║       --age 45 --gender Male --hrv 28.5 --map 96.3 --bmi 24.5                 ║
║                                                                                  ║
║   STEP 3 — Normal vs Pneumonia Grad-CAM comparison:                             ║
║   python pneumofusion_xai.py --mode compare                                     ║
║       --normal_img  data/test/NORMAL/img.jpeg                                   ║
║       --pneumonia_img data/test/PNEUMONIA/img.jpeg                              ║
║                                                                                  ║
║   STEP 4 — SHAP analysis on vital signs:                                        ║
║   python pneumofusion_xai.py --mode shap                                        ║
║       --vitals_csv human_vital_signs_dataset_2024.csv                           ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════════════════════
#  IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import os, sys, json, time, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_score, recall_score,
    f1_score, precision_recall_curve
)
from sklearn.utils.class_weight import compute_class_weight

# XGBoost
from xgboost import XGBClassifier

# TensorFlow
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model, Model
from pathlib import Path

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False
    print("[info] pip install shap  →  SHAP disabled")

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
SEED        = 42
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS_P1   = 20        # CNN frozen phase
EPOCHS_P2   = 10        # CNN fine-tune phase
LR_P1       = 1e-4
LR_P2       = 1e-5
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

np.random.seed(SEED)
tf.random.set_seed(SEED)

# Exact features from your vital.py
VITAL_FEATURES = [
    "Heart Rate", "Respiratory Rate", "Body Temperature",
    "Oxygen Saturation", "Systolic Blood Pressure",
    "Diastolic Blood Pressure", "Age", "Gender",
    "Derived_HRV", "Derived_MAP", "Derived_BMI",
]

# Output folders
OUT    = Path("pneumofusion_outputs")
MDIR   = OUT / "models"
PDIR   = OUT / "plots"
RDIR   = OUT / "results"
for d in [OUT, MDIR, PDIR, RDIR]:
    d.mkdir(parents=True, exist_ok=True)

# Colormaps
CAM_MAP  = LinearSegmentedColormap.from_list(
    "cam",    ["#000000","#0d2137","#f39c12","#e74c3c","#ffffff"])
NORM_MAP = LinearSegmentedColormap.from_list(
    "normal", ["#000000","#001a33","#00d4ff","#00ff9d","#ffffff"])
PNEU_MAP = LinearSegmentedColormap.from_list(
    "pneumo", ["#000000","#1a0a00","#f39c12","#e74c3c","#ffffff"])


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT STYLE
# ══════════════════════════════════════════════════════════════════════════════
def _style():
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor":"#050a12","axes.facecolor":"#0b1220",
        "axes.edgecolor":"#1a2a40","axes.labelcolor":"#c0d0e0",
        "xtick.color":"#6080a0","ytick.color":"#6080a0",
        "text.color":"#c0d0e0","grid.color":"#1a2a40",
        "grid.linestyle":"--","font.family":"monospace",
    })


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 ── VITAL SIGNS  (exact vital.py logic)
# ══════════════════════════════════════════════════════════════════════════════

def train_vitals(csv_path: str) -> dict:
    """
    Reproduces your entire vital.py exactly:
      • Same XGBoost parameters
      • Same features
      • Same encoders / scaler
      • All plots: confusion matrix, ROC, precision-recall, loss, feature importance
      • Cross-validation
      • Saves: advanced_risk_model.pkl, advanced_scaler.pkl,
               advanced_label_encoder.pkl, advanced_gender_encoder.pkl
    """
    print("\n" + "═"*60)
    print("  VITAL SIGNS  —  XGBoost  (vital.py)")
    print("═"*60)

    # ── Load ──────────────────────────────────────────────────────────────
    data = pd.read_csv(csv_path)
    print(f"  Loaded: {csv_path}  shape={data.shape}")
    print(f"  Columns: {list(data.columns)}")

    data = _fix_vital_columns(data)

    # Keep exact columns from vital.py
    data = data[VITAL_FEATURES + ["Risk Category"]].copy()

    # ── Encode (exact from vital.py) ──────────────────────────────────────
    label_encoder  = LabelEncoder()
    gender_encoder = LabelEncoder()
    data["Risk Category"] = label_encoder.fit_transform(data["Risk Category"])
    data["Gender"]        = gender_encoder.fit_transform(data["Gender"])

    X = data.drop("Risk Category", axis=1)
    y = data["Risk Category"]

    scaler   = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # ── XGBoost (exact from vital.py) ────────────────────────────────────
    model = XGBClassifier(
        n_estimators=400, max_depth=7, learning_rate=0.03,
        subsample=0.9, colsample_bytree=0.9, gamma=0.1,
        reg_alpha=0.1, reg_lambda=1,
        objective="binary:logistic", eval_metric="logloss",
        random_state=42,
    )
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    # ── Evaluation (exact from vital.py) ─────────────────────────────────
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    print(f"\n  🔥 FINAL TEST ACCURACY: {accuracy:.4f}")
    print(f"\n  Classification Report:\n")
    print(classification_report(y_test, y_pred))
    print(f"  🔥 AUC Score: {roc_auc:.4f}")

    # ── All plots from vital.py ───────────────────────────────────────────
    _vital_plots(model, X, y_test, y_pred, y_prob, fpr, tpr, roc_auc)

    # ── Cross validation ──────────────────────────────────────────────────
    kfold     = StratifiedKFold(n_splits=5)
    cv_scores = cross_val_score(model, X_scaled, y, cv=kfold)
    print(f"\n  🔥 5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Save (exact from vital.py) ────────────────────────────────────────
    joblib.dump(model,          MDIR / "advanced_risk_model.pkl")
    joblib.dump(scaler,         MDIR / "advanced_scaler.pkl")
    joblib.dump(label_encoder,  MDIR / "advanced_label_encoder.pkl")
    joblib.dump(gender_encoder, MDIR / "advanced_gender_encoder.pkl")
    print("  ✅ Advanced Model Saved Successfully")

    return dict(model=model, scaler=scaler,
                label_encoder=label_encoder, gender_encoder=gender_encoder,
                X_train=X_train, X_test=X_test,
                y_train=y_train, y_test=y_test,
                y_prob_test=y_prob,
                feature_names=list(X.columns),
                X_scaled=X_scaled, y_all=y)


def _fix_vital_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map CSV column names → exact vital.py names. Also derive missing columns."""
    lc = {c.lower().strip(): c for c in df.columns}
    mp = {
        "Heart Rate":               ["heart rate","heartrate","hr","pulse rate"],
        "Respiratory Rate":         ["respiratory rate","resp rate","rr"],
        "Body Temperature":         ["body temperature","temperature","body temp","temp"],
        "Oxygen Saturation":        ["oxygen saturation","spo2","o2 saturation"],
        "Systolic Blood Pressure":  ["systolic blood pressure","systolic bp","sbp"],
        "Diastolic Blood Pressure": ["diastolic blood pressure","diastolic bp","dbp"],
        "Age":                      ["age","patient age"],
        "Gender":                   ["gender","sex"],
        "Derived_HRV":              ["derived_hrv","hrv","heart rate variability"],
        "Derived_MAP":              ["derived_map","map","mean arterial pressure"],
        "Derived_BMI":              ["derived_bmi","bmi","body mass index"],
        "Risk Category":            ["risk category","risk_category","label",
                                     "diagnosis","condition","target","risk"],
    }
    rename = {}
    for std, aliases in mp.items():
        for a in aliases:
            if a in lc and lc[a] != std:
                rename[lc[a]] = std; break
    df = df.rename(columns=rename)

    # Derive missing columns
    if "Derived_HRV" not in df.columns:
        df["Derived_HRV"] = (1000.0 / (df["Heart Rate"]+1e-6)
                             if "Heart Rate" in df.columns else 30.0)
    if "Derived_MAP" not in df.columns:
        if {"Systolic Blood Pressure","Diastolic Blood Pressure"} <= set(df.columns):
            df["Derived_MAP"] = (df["Diastolic Blood Pressure"] +
                                 (df["Systolic Blood Pressure"] -
                                  df["Diastolic Blood Pressure"]) / 3.0)
        else:
            df["Derived_MAP"] = 93.0
    if "Derived_BMI" not in df.columns:
        df["Derived_BMI"] = 22.5
    if "Risk Category" not in df.columns:
        if "Oxygen Saturation" in df.columns:
            df["Risk Category"] = (df["Oxygen Saturation"] < 95).astype(int)
        else:
            df["Risk Category"] = 0
    return df


def _vital_plots(model, X, y_test, y_pred, y_prob, fpr, tpr, roc_auc):
    """Reproduce all vital.py plots with dark styling."""
    _style()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6,5))
    fig.suptitle("Confusion Matrix — XGBoost Vitals",
                 color="#00d4ff", fontsize=12, fontweight="bold")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                linewidths=0.5, ax=ax, annot_kws={"size":14,"weight":"bold"})
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    fig.savefig(PDIR/"vitals_confusion_matrix.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    # ROC
    fig, ax = plt.subplots(figsize=(7,6))
    fig.suptitle("ROC Curve — XGBoost Vitals",
                 color="#00d4ff", fontsize=12, fontweight="bold")
    ax.plot(fpr, tpr, color="#00d4ff", lw=2.5, label=f"AUC = {roc_auc:.4f}")
    ax.fill_between(fpr, tpr, alpha=0.10, color="#00d4ff")
    ax.plot([0,1],[0,1], color="#4a6080", ls="--")
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=11, facecolor="#0b1220")
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(PDIR/"vitals_roc_curve.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    # Precision-Recall
    precision_arr, recall_arr, _ = precision_recall_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(7,6))
    fig.suptitle("Precision-Recall Curve",
                 color="#00d4ff", fontsize=12, fontweight="bold")
    ax.plot(recall_arr, precision_arr, color="#ff6b35", lw=2.5)
    ax.fill_between(recall_arr, precision_arr, alpha=0.10, color="#ff6b35")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(PDIR/"vitals_precision_recall.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    # Loss curve
    res = model.evals_result()
    fig, ax = plt.subplots(figsize=(8,5))
    fig.suptitle("XGBoost Loss Curve",
                 color="#00d4ff", fontsize=12, fontweight="bold")
    ax.plot(res["validation_0"]["logloss"], color="#00d4ff", lw=2,
            label="Train Loss")
    ax.plot(res["validation_1"]["logloss"], color="#ff3d5a", lw=2,
            ls="--", label="Val Loss")
    ax.fill_between(range(len(res["validation_0"]["logloss"])),
                    res["validation_0"]["logloss"], alpha=0.07, color="#00d4ff")
    ax.set_xlabel("Boosting Round"); ax.set_ylabel("Log Loss")
    ax.legend(fontsize=10, facecolor="#0b1220")
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(PDIR/"vitals_loss_curve.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    # Feature importance
    imp = model.feature_importances_
    fdf = pd.DataFrame({"Feature": list(X.columns), "Importance": imp}
                       ).sort_values("Importance", ascending=True)
    fig, ax = plt.subplots(figsize=(9,7))
    fig.suptitle("XGBoost Feature Importance",
                 color="#00d4ff", fontsize=12, fontweight="bold")
    cols = ["#ff3d5a" if v > fdf["Importance"].median()
            else "#00d4ff" for v in fdf["Importance"]]
    bars = ax.barh(fdf["Feature"], fdf["Importance"],
                   color=cols, edgecolor="#1a2a40")
    for bar, val in zip(bars, fdf["Importance"]):
        ax.text(val+0.001, bar.get_y()+bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=8, color="white")
    ax.set_xlabel("Importance Score")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2, axis="x")
    plt.tight_layout()
    fig.savefig(PDIR/"vitals_feature_importance.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"  ✅ Vitals plots saved → {PDIR}")
    print(f"\n  🔥 Feature Importance Ranking:")
    print(fdf[::-1].to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 ── CNN  (ResNet50 — exact pneumonia.py logic)
# ══════════════════════════════════════════════════════════════════════════════

def build_cnn(fine_tune=False):
    base = ResNet50(weights="imagenet", include_top=False,
                    input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = fine_tune
    if fine_tune:
        for l in base.layers[:-30]:
            l.trainable = False

    inp = tf.keras.Input(shape=(IMG_SIZE,IMG_SIZE,3), name="image_input")
    x   = base(inp, training=False)
    x   = layers.GlobalAveragePooling2D(name="gap")(x)
    x   = layers.Dense(256, activation="relu",
                       kernel_regularizer=regularizers.l2(1e-4),
                       name="dense_256")(x)
    x   = layers.BatchNormalization(name="bn_256")(x)
    x   = layers.Dropout(0.40, name="drop_256")(x)
    x   = layers.Dense(128, activation="relu", name="dense_128")(x)
    x   = layers.BatchNormalization(name="bn_128")(x)
    x   = layers.Dropout(0.30, name="drop_128")(x)
    out = layers.Dense(1, activation="sigmoid", name="cnn_out")(x)
    return Model(inp, out, name="PneumoCNN")


def _compile_cnn(model, lr):
    model.compile(
        optimizer=optimizers.Adam(lr), loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall"),
                 tf.keras.metrics.AUC(name="auc")])
    return model


def _img_gens(data_dir, batch_size=BATCH_SIZE):
    train_g = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15, width_shift_range=0.10,
        height_shift_range=0.10, zoom_range=0.10,
        horizontal_flip=True, brightness_range=[0.80,1.20],
        fill_mode="nearest")
    eval_g = ImageDataGenerator(preprocessing_function=preprocess_input)

    def _flow(g, split, sh):
        return g.flow_from_directory(
            os.path.join(data_dir, split),
            target_size=(IMG_SIZE,IMG_SIZE), batch_size=batch_size,
            class_mode="binary", classes=CLASS_NAMES,
            shuffle=sh, seed=SEED)

    return _flow(train_g,"train",True), _flow(eval_g,"val",False), \
           _flow(eval_g,"test",False)


def train_cnn(data_dir, epochs_p1=EPOCHS_P1, epochs_p2=EPOCHS_P2,
              batch_size=BATCH_SIZE):
    print("\n" + "═"*60)
    print("  CNN  —  ResNet50  (pneumonia.py)")
    print("═"*60)

    train_gen, val_gen, test_gen = _img_gens(data_dir, batch_size)
    mp  = str(MDIR / "cnn_best.h5")
    cw  = dict(enumerate(compute_class_weight(
        "balanced", classes=np.unique(train_gen.classes),
        y=train_gen.classes)))

    cbs = lambda: [
        callbacks.ModelCheckpoint(mp, monitor="val_auc", mode="max",
                                   save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                     patience=3, min_lr=1e-7, verbose=1),
        callbacks.EarlyStopping(monitor="val_auc", patience=6, mode="max",
                                 restore_best_weights=True, verbose=1),
        callbacks.CSVLogger(str(RDIR/"cnn_log.csv")),
    ]

    print("\n  Phase 1 — Frozen ResNet50")
    m1 = _compile_cnn(build_cnn(False), LR_P1)
    m1.summary(line_length=78)
    h1 = m1.fit(train_gen, validation_data=val_gen, epochs=epochs_p1,
                callbacks=cbs(), class_weight=cw, verbose=1)

    print("\n  Phase 2 — Fine-tune top 30 layers")
    # Load the saved best model directly, then unfreeze top 30 ResNet layers
    m2 = load_model(mp)
    resnet_layer = m2.get_layer("resnet50") if "resnet50" in \
        [l.name for l in m2.layers] else None
    if resnet_layer:
        resnet_layer.trainable = True
        for layer in resnet_layer.layers[:-30]:
            layer.trainable = False
    else:
        for layer in m2.layers:
            layer.trainable = True
    m2 = _compile_cnn(m2, LR_P2)
    h2 = m2.fit(train_gen, validation_data=val_gen, epochs=epochs_p2,
                callbacks=cbs(), class_weight=cw, verbose=1)

    history = {k: h1.history[k]+h2.history.get(k,[]) for k in h1.history}
    _plot_cnn_training(history)

    best = load_model(mp)
    _eval_cnn(best, test_gen)
    return best


def _plot_cnn_training(history):
    _style()
    epochs = range(1, len(history["accuracy"])+1)
    fig, axes = plt.subplots(1, 3, figsize=(19,5))
    fig.suptitle("CNN Training Curves — ResNet50",
                 fontsize=14, color="#00d4ff", fontweight="bold")
    for ax,(tk,vk,title,c1,c2) in zip(axes,[
        ("accuracy","val_accuracy","Accuracy","#00d4ff","#ff6b35"),
        ("loss","val_loss","Loss","#00ff9d","#ff3d5a"),
        ("auc","val_auc","AUC","#ffd166","#c77dff"),
    ]):
        if tk in history:
            ax.plot(epochs, history[tk], color=c1, lw=2.5, label="Train")
            ax.fill_between(epochs, history[tk], alpha=0.07, color=c1)
        if vk in history:
            ax.plot(epochs, history[vk], color=c2, lw=2, ls="--",
                    label="Val", alpha=0.85)
        ax.set_title(title, color="#00d4ff", fontsize=11)
        ax.set_xlabel("Epoch"); ax.set_ylabel(title)
        ax.legend(fontsize=9, facecolor="#0b1220", edgecolor="#1a2a40")
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(PDIR/"cnn_training_curves.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [plot] CNN training curves → {PDIR/'cnn_training_curves.png'}")


def _eval_cnn(model, test_gen):
    test_gen.reset()
    y_prob = model.predict(test_gen, verbose=1).ravel()
    y_true = test_gen.classes[:len(y_prob)]
    y_pred = (y_prob >= 0.5).astype(int)
    fpr,tpr,_ = roc_curve(y_true, y_prob)
    rauc = auc(fpr,tpr)
    print(f"\n  CNN Accuracy : {accuracy_score(y_true,y_pred):.4f}")
    print(f"  CNN AUC      : {rauc:.4f}")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    _plot_cm_roc(y_true, y_pred, fpr, tpr, rauc,
                 "CNN — Confusion Matrix & ROC", "cnn_cm_roc.png")


def _plot_cm_roc(y_true, y_pred, fpr, tpr, roc_auc, title, fname):
    _style()
    fig, axes = plt.subplots(1,2,figsize=(14,6))
    fig.suptitle(title, color="#00d4ff", fontsize=13, fontweight="bold")
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, ax=axes[0],
                annot_kws={"size":14,"weight":"bold"})
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            axes[0].text(j+0.5,i+0.72,f"({cm[i,j]/total*100:.1f}%)",
                         ha="center",fontsize=9,color="#aabbcc")
    axes[0].set_title("Confusion Matrix",color="#00d4ff")
    axes[0].tick_params(colors="#c0d0e0")
    axes[1].plot(fpr,tpr,color="#00d4ff",lw=2.5,
                 label=f"AUC={roc_auc:.4f}")
    axes[1].fill_between(fpr,tpr,alpha=0.10,color="#00d4ff")
    axes[1].plot([0,1],[0,1],color="#4a6080",ls="--")
    axes[1].set_xlim([0,1]); axes[1].set_ylim([0,1.02])
    axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
    axes[1].set_title("ROC Curve",color="#00d4ff")
    axes[1].legend(fontsize=10,facecolor="#0b1220")
    axes[1].grid(True,alpha=0.25)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(PDIR/fname, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 ── FUSION HEAD
# ══════════════════════════════════════════════════════════════════════════════

def build_fusion_head(n_vital_in: int) -> tf.keras.Model:
    """
    Inputs  : img_features (128+1=129) + vital_features (xgb_proba + scaled vitals)
    Output  : P(PNEUMONIA)
    """
    img_in   = tf.keras.Input(shape=(129,),       name="img_feat")
    vital_in = tf.keras.Input(shape=(n_vital_in,), name="vital_feat")

    f = layers.Concatenate(name="concat")([img_in, vital_in])
    f = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4),
                     name="fus_128")(f)
    f = layers.BatchNormalization(name="fus_bn1")(f)
    f = layers.Dropout(0.40, name="fus_d1")(f)
    f = layers.Dense(64, activation="relu", name="fus_64")(f)
    f = layers.BatchNormalization(name="fus_bn2")(f)
    f = layers.Dropout(0.30, name="fus_d2")(f)
    out = layers.Dense(1, activation="sigmoid", name="fusion_out")(f)
    return Model([img_in, vital_in], out, name="FusionHead")


def _extract_img_features(cnn_model, gen):
    """Extract 128-dim penultimate features + probability → 129-dim."""
    feat_model = Model(
        inputs=cnn_model.input,
        outputs=[cnn_model.get_layer("dense_128").output,
                 cnn_model.output])
    gen.reset()
    feats, probs, labels = [], [], []
    for imgs, lbls in gen:
        f, p = feat_model.predict(imgs, verbose=0)
        feats.append(f); probs.append(p.ravel()); labels.append(lbls)
        if sum(len(x) for x in labels) >= gen.n:
            break
    F = np.concatenate(feats)[:gen.n]
    P = np.concatenate(probs)[:gen.n]
    L = np.concatenate(labels)[:gen.n]
    return np.column_stack([F, P]), L


def _sample_vitals(X_vit, y_vit, img_labels):
    """Sample vital rows whose label matches image label for alignment."""
    v0 = X_vit[y_vit==0]; v1 = X_vit[y_vit==1]
    if len(v0)==0: v0=X_vit
    if len(v1)==0: v1=X_vit
    out = []
    for lbl in img_labels:
        pool = v1 if lbl==1 else v0
        out.append(pool[np.random.randint(0,len(pool))])
    return np.array(out, dtype=np.float32)


def train_fusion(data_dir, vital_results, cnn_model):
    print("\n" + "═"*60)
    print("  FUSION HEAD TRAINING")
    print("═"*60)

    train_gen, val_gen, test_gen = _img_gens(data_dir)

    # Image features
    tr_if, tr_lbl = _extract_img_features(cnn_model, train_gen)
    vl_if, vl_lbl = _extract_img_features(cnn_model, val_gen)
    te_if, te_lbl = _extract_img_features(cnn_model, test_gen)

    # Vital features aligned to image labels
    X_vit = vital_results["X_scaled"]
    y_vit = vital_results["y_all"].values
    xgb   = vital_results["model"]

    def _make_vital_feat(img_lbl):
        sampled  = _sample_vitals(X_vit, y_vit, img_lbl)
        xgb_prob = xgb.predict_proba(sampled)
        return np.concatenate([xgb_prob, sampled], axis=1).astype(np.float32)

    tr_vf = _make_vital_feat(tr_lbl)
    vl_vf = _make_vital_feat(vl_lbl)
    te_vf = _make_vital_feat(te_lbl)

    print(f"\n  img_feat shape   : {tr_if.shape}")
    print(f"  vital_feat shape : {tr_vf.shape}")

    n_vit  = tr_vf.shape[1]
    fmodel = build_fusion_head(n_vit)
    fmodel.compile(
        optimizer=optimizers.Adam(1e-3), loss="binary_crossentropy",
        metrics=["accuracy",tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")])
    fmodel.summary(line_length=78)

    fp  = str(MDIR/"fusion_head_best.h5")
    cw  = dict(enumerate(compute_class_weight(
        "balanced", classes=np.unique(tr_lbl.astype(int)),
        y=tr_lbl.astype(int))))

    hf = fmodel.fit(
        [tr_if, tr_vf], tr_lbl,
        validation_data=([vl_if, vl_vf], vl_lbl),
        epochs=50, batch_size=64, class_weight=cw,
        callbacks=[
            callbacks.ModelCheckpoint(fp, monitor="val_auc", mode="max",
                                       save_best_only=True, verbose=1),
            callbacks.ReduceLROnPlateau(patience=5,factor=0.5,verbose=1),
            callbacks.EarlyStopping(monitor="val_auc",patience=10,
                                     mode="max",restore_best_weights=True),
        ], verbose=1)

    _plot_fusion_training(hf.history)

    best = load_model(fp)
    _eval_fusion(best, te_if, te_vf, te_lbl)

    # Save config
    json.dump({"n_vital_in": int(n_vit),
               "feature_names": vital_results["feature_names"]},
              open(MDIR/"fusion_config.json","w"), indent=2)
    return best, n_vit


def _plot_fusion_training(history):
    _style()
    epochs = range(1, len(history["accuracy"])+1)
    fig, axes = plt.subplots(1,3,figsize=(19,5))
    fig.suptitle("Fusion Head Training Curves",
                 fontsize=14, color="#00d4ff", fontweight="bold")
    for ax,(tk,vk,title,c1,c2) in zip(axes,[
        ("accuracy","val_accuracy","Accuracy","#00d4ff","#ff6b35"),
        ("loss","val_loss","Loss","#00ff9d","#ff3d5a"),
        ("auc","val_auc","AUC","#ffd166","#c77dff"),
    ]):
        if tk in history:
            ax.plot(epochs,history[tk],color=c1,lw=2.5,label="Train")
        if vk in history:
            ax.plot(epochs,history[vk],color=c2,lw=2,ls="--",label="Val")
        ax.set_title(title,color="#00d4ff"); ax.set_xlabel("Epoch")
        ax.legend(fontsize=9,facecolor="#0b1220",edgecolor="#1a2a40")
        ax.grid(True,alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(PDIR/"fusion_training_curves.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _eval_fusion(model, te_if, te_vf, y_true):
    y_prob = model.predict([te_if,te_vf],verbose=0).ravel()
    y_pred = (y_prob>=0.5).astype(int)
    acc=accuracy_score(y_true,y_pred)
    prec=precision_score(y_true,y_pred)
    rec=recall_score(y_true,y_pred)
    f1=f1_score(y_true,y_pred)
    fpr,tpr,_=roc_curve(y_true,y_prob)
    rauc=auc(fpr,tpr)
    print(f"\n  ╔══ FUSION RESULTS ══════════════════════════════╗")
    print(f"  ║  Accuracy  : {acc:.4f}                           ║")
    print(f"  ║  Precision : {prec:.4f}                           ║")
    print(f"  ║  Recall    : {rec:.4f}                           ║")
    print(f"  ║  F1 Score  : {f1:.4f}                           ║")
    print(f"  ║  ROC-AUC   : {rauc:.4f}                           ║")
    print(f"  ╚════════════════════════════════════════════════╝")
    print(classification_report(y_true,y_pred,target_names=CLASS_NAMES))
    metrics={k:round(float(v),4) for k,v in zip(
        ["accuracy","precision","recall","f1","auc"],
        [acc,prec,rec,f1,rauc])}
    json.dump(metrics, open(RDIR/"fusion_metrics.json","w"), indent=2)
    _plot_cm_roc(y_true,y_pred,fpr,tpr,rauc,
                 "Fusion — Confusion Matrix & ROC","fusion_cm_roc.png")

    # Big dashboard
    _style()
    fig = plt.figure(figsize=(20,8),facecolor="#050a12")
    fig.suptitle("PneumoFusion — Final Evaluation Dashboard",
                 fontsize=15,color="#00d4ff",fontweight="bold")
    gs = gridspec.GridSpec(1,4,figure=fig,wspace=0.35)
    for i,(name,val,col) in enumerate(zip(
        ["Accuracy","Precision","Recall","F1"],
        [acc,prec,rec,f1],
        ["#00d4ff","#ff6b35","#00ff9d","#ffd166"]
    )):
        ax=fig.add_subplot(gs[i])
        ax.set_facecolor("#0b1220")
        ax.add_patch(plt.Rectangle((0.05,0.08),0.90,0.84,
            transform=ax.transAxes,
            facecolor=f"{col}18",edgecolor=f"{col}99",
            linewidth=2,clip_on=False))
        ax.text(0.5,0.60,f"{val*100:.2f}%",transform=ax.transAxes,
                ha="center",va="center",fontsize=26,
                fontweight="bold",color=col)
        ax.text(0.5,0.26,name,transform=ax.transAxes,
                ha="center",va="center",fontsize=12,color="#c0d0e0")
        ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis("off")
    plt.tight_layout()
    fig.savefig(PDIR/"fusion_dashboard.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [plot] Fusion dashboard → {PDIR/'fusion_dashboard.png'}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 ── GRAD-CAM  +  GRAD-CAM++
# ══════════════════════════════════════════════════════════════════════════════

class GradCAM:
    def __init__(self, cnn, layer="conv5_block3_out"):
        self.gm = Model(cnn.input,
                        [cnn.get_layer(layer).output, cnn.output])

    def heatmap(self, img_arr, cls=1):
        t = tf.cast(img_arr, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(t)
            co, pred = self.gm(t, training=False)
            score = pred[:,0] if cls==1 else (1-pred[:,0])
        g   = tape.gradient(score, co)
        pw  = tf.reduce_mean(g, axis=(0,1,2))
        h   = tf.squeeze(tf.nn.relu(co[0] @ pw[...,tf.newaxis])).numpy()
        if h.max()>0: h/=h.max()
        return cv2.resize(h,(IMG_SIZE,IMG_SIZE)).astype(np.float32)

    def overlay(self, img, h, alpha=0.45, cmap=cv2.COLORMAP_JET):
        c = cv2.cvtColor(cv2.applyColorMap(np.uint8(255*h),cmap),
                         cv2.COLOR_BGR2RGB)
        return ((1-alpha)*img+alpha*c).clip(0,255).astype(np.uint8)


class GradCAMPP:
    def __init__(self, cnn, layer="conv5_block3_out"):
        self.gm = Model(cnn.input,
                        [cnn.get_layer(layer).output, cnn.output])

    def heatmap(self, img_arr, cls=1):
        t = tf.cast(img_arr, tf.float32)
        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
                with tf.GradientTape() as t0:
                    for tape in [t0,t1,t2]: tape.watch(t)
                    co, pred = self.gm(t, training=False)
                    score = pred[:,0] if cls==1 else (1-pred[:,0])
                g1 = t0.gradient(score, co)
            g2 = t1.gradient(g1, t)
        g3 = t2.gradient(g2, t)
        g1n = g1.numpy()
        g2n = g2.numpy() if g2 is not None else np.zeros_like(g1n)
        g3n = g3.numpy() if g3 is not None else np.zeros_like(g1n)
        a_n = g2n[0]
        a_d = 2*g2n[0] + co.numpy()[0]*g3n[0] + 1e-7
        alpha = np.where(a_d!=0, a_n/a_d, 0)
        w = (alpha*np.maximum(g1n[0],0)).sum(axis=(0,1))
        h = np.maximum((co.numpy()[0]*w).sum(axis=-1),0)
        if h.max()>0: h/=h.max()
        return cv2.resize(h,(IMG_SIZE,IMG_SIZE)).astype(np.float32)

    def overlay(self, img, h, alpha=0.45, cmap=cv2.COLORMAP_INFERNO):
        c = cv2.cvtColor(cv2.applyColorMap(np.uint8(255*h),cmap),
                         cv2.COLOR_BGR2RGB)
        return ((1-alpha)*img+alpha*c).clip(0,255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 ── NORMAL vs PNEUMONIA GRAD-CAM COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def compare_gradcam(cnn_model, normal_path, pneumo_path,
                    save_path=None):
    """
    4-row × 2-col figure:
      Row 1: Original X-ray
      Row 2: Lung ROI Mask
      Row 3: Grad-CAM overlay
      Row 4: Grad-CAM++ overlay
    Left column = NORMAL,  Right column = PNEUMONIA
    """
    _style()
    save_path = save_path or str(PDIR/"gradcam_normal_vs_pneumonia.png")

    cam  = GradCAM(cnn_model)
    capp = GradCAMPP(cnn_model)
    res  = {}

    for tag, path in [("normal",normal_path),("pneumonia",pneumo_path)]:
        arr, disp = _load_img(path)
        cls  = 0 if tag=="normal" else 1
        prob = float(cnn_model.predict(arr, verbose=0)[0][0])
        h1   = cam.heatmap(arr, cls)
        h2   = capp.heatmap(arr, cls)
        res[tag] = dict(
            disp=disp, roi=_lung_mask(disp), prob=prob,
            cam=cam.overlay(disp, h1,
                            cmap=cv2.COLORMAP_COOL if tag=="normal"
                            else cv2.COLORMAP_JET),
            campp=capp.overlay(disp, h2,
                               cmap=cv2.COLORMAP_OCEAN if tag=="normal"
                               else cv2.COLORMAP_INFERNO),
            h_cam=h1, h_campp=h2,
        )

    col_c = {"normal":"#00ff9d", "pneumonia":"#ff3d5a"}
    rows  = ["① Original X-ray", "② Lung ROI Mask",
             "③ Grad-CAM", "④ Grad-CAM++"]
    keys  = ["disp","roi","cam","campp"]

    fig = plt.figure(figsize=(14,23), facecolor="#050a12")
    fig.suptitle(
        "Normal vs Pneumonia — Grad-CAM & Grad-CAM++ Comparison",
        fontsize=14, color="#00d4ff", fontweight="bold", y=0.993)
    gs = gridspec.GridSpec(4,2, figure=fig, hspace=0.05, wspace=0.05)

    for col, tag in enumerate(["normal","pneumonia"]):
        d = res[tag]; bc = col_c[tag]
        for row,(key,rlbl) in enumerate(zip(keys,rows)):
            ax = fig.add_subplot(gs[row,col])
            ax.imshow(d[key])
            if row==0:
                ax.set_title(
                    f"{'NORMAL' if tag=='normal' else 'PNEUMONIA'}\n"
                    f"P(Pneumonia)={d['prob']*100:.1f}%",
                    color=bc, fontsize=12, fontweight="bold", pad=6)
            if col==0:
                ax.set_ylabel(rlbl, color="#c0d0e0",
                              fontsize=9, labelpad=4)
                ax.yaxis.set_label_coords(-0.03, 0.5)
            ax.axis("off")
            for sp in ax.spines.values():
                sp.set_edgecolor(bc); sp.set_linewidth(2)

    handles = [
        mpatches.Patch(color="#00ff9d", label="Normal — cool tones"),
        mpatches.Patch(color="#ff3d5a", label="Pneumonia — hot tones"),
        mpatches.Patch(color="#00d4ff", label="Low activation"),
        mpatches.Patch(color="#f39c12", label="High activation"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               facecolor="#0b1220", edgecolor="#1a2a40",
               bbox_to_anchor=(0.5,0.002))
    plt.tight_layout(rect=[0,0.03,1,0.98])
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  [XAI] Grad-CAM comparison → {save_path}")
    return save_path


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 ── SHAP  (XGBoost TreeExplainer — 4 panel)
# ══════════════════════════════════════════════════════════════════════════════

def run_shap(vital_results, n_explain=100):
    """
    4-panel SHAP on XGBoost vitals model:
      A — Mean |SHAP| bar chart
      B — Beeswarm scatter
      C — Waterfall for highest-risk patient
      D — Dependence plot (top 2 features)
    """
    if not SHAP_OK:
        print("  [skip] pip install shap"); return

    _style()
    model      = vital_results["model"]
    X_test     = vital_results["X_test"]
    feat_names = vital_results["feature_names"]

    print("\n  Running SHAP TreeExplainer…")
    explainer = shap.TreeExplainer(model)
    X_ex      = X_test[:n_explain]
    sv_raw    = explainer.shap_values(X_ex)
    sv = sv_raw[1] if isinstance(sv_raw,list) else sv_raw

    # Predictions for picking top patient
    preds = model.predict_proba(X_ex)[:,1]

    fig = plt.figure(figsize=(22,14), facecolor="#050a12")
    fig.suptitle("SHAP Analysis — XGBoost Vital Signs",
                 fontsize=16, color="#00d4ff", fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2,2, figure=fig, hspace=0.42, wspace=0.35)

    # ── A: Mean |SHAP| bar ────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0,0]); ax_a.set_facecolor("#0b1220")
    msv  = np.abs(sv).mean(axis=0)
    si   = np.argsort(msv)[::-1]
    bc   = ["#ff3d5a" if msv[i]>msv.mean() else "#00d4ff" for i in si]
    bars = ax_a.bar([feat_names[i] for i in si], msv[si],
                    color=bc, edgecolor="#1a2a40")
    for b,v in zip(bars,msv[si]):
        ax_a.text(b.get_x()+b.get_width()/2, b.get_height()+msv.max()*0.012,
                  f"{v:.4f}", ha="center", fontsize=8, color="white")
    ax_a.set_title("A. Mean |SHAP| per Feature",
                   color="#00d4ff", fontsize=12, pad=8)
    ax_a.set_ylabel("|SHAP value|")
    ax_a.tick_params(axis="x", rotation=35, labelsize=8)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.grid(True, alpha=0.20, axis="y")

    # ── B: Beeswarm ───────────────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0,1]); ax_b.set_facecolor("#0b1220")
    for idx in range(len(feat_names)):
        jitter = np.random.normal(idx, 0.08, size=len(sv))
        sc = ax_b.scatter(sv[:,idx], jitter,
                          c=X_ex[:,idx], cmap="RdYlBu_r",
                          alpha=0.65, s=20, edgecolors="none")
    ax_b.set_yticks(range(len(feat_names)))
    ax_b.set_yticklabels(feat_names, fontsize=9)
    ax_b.axvline(0, color="#4a6080", lw=1.5, ls="--")
    ax_b.set_title("B. Beeswarm — SHAP Distribution",
                   color="#00d4ff", fontsize=12, pad=8)
    ax_b.set_xlabel("SHAP value  (→ High Risk  |  ← Low Risk)")
    plt.colorbar(sc, ax=ax_b, label="Feature value (scaled)", shrink=0.8)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.grid(True, alpha=0.12, axis="x")

    # ── C: Waterfall (highest-risk patient) ───────────────────────────────
    ax_c = fig.add_subplot(gs[1,0]); ax_c.set_facecolor("#0b1220")
    top_i  = int(np.argmax(preds))
    sv_p   = sv[top_i]
    sfi    = np.argsort(np.abs(sv_p))[::-1]
    sv_s   = sv_p[sfi]
    fn_s   = [feat_names[i] for i in sfi]
    bc2    = ["#ff3d5a" if v>0 else "#00d4ff" for v in sv_s]
    ax_c.barh(fn_s, sv_s, color=bc2, edgecolor="#1a2a40")
    ax_c.axvline(0, color="#4a6080", lw=1.5, ls="--")
    ax_c.set_title(
        f"C. Patient Waterfall\n(Risk Score={preds[top_i]*100:.1f}%)",
        color="#00d4ff", fontsize=12, pad=8)
    ax_c.set_xlabel("SHAP value")
    ax_c.tick_params(labelsize=9)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    rp = mpatches.Patch(color="#ff3d5a", label="↑ Increases Risk")
    bp = mpatches.Patch(color="#00d4ff", label="↓ Reduces Risk")
    ax_c.legend(handles=[rp,bp], fontsize=8,
                facecolor="#0b1220", edgecolor="#1a2a40")

    # ── D: Dependence plot (top 2 features) ──────────────────────────────
    ax_d = fig.add_subplot(gs[1,1]); ax_d.set_facecolor("#0b1220")
    top2 = np.argsort(msv)[-2:]
    f1i,f2i = int(top2[0]),int(top2[1])
    sc2 = ax_d.scatter(X_ex[:,f1i], sv[:,f1i],
                       c=X_ex[:,f2i], cmap="plasma",
                       s=25, alpha=0.7, edgecolors="none")
    ax_d.axhline(0, color="#4a6080", lw=1, ls="--")
    ax_d.set_xlabel(f"{feat_names[f1i]} (scaled)")
    ax_d.set_ylabel(f"SHAP({feat_names[f1i]})")
    ax_d.set_title(
        f"D. Dependence: {feat_names[f1i]}\n(colour={feat_names[f2i]})",
        color="#00d4ff", fontsize=12, pad=8)
    plt.colorbar(sc2, ax=ax_d, label=feat_names[f2i], shrink=0.8)
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)
    ax_d.grid(True, alpha=0.15)

    plt.tight_layout(rect=[0,0,1,0.97])
    path = str(PDIR/"shap_vitals_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [XAI] SHAP analysis → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 ── PREDICT SINGLE PATIENT
# ══════════════════════════════════════════════════════════════════════════════

def predict_patient(image_path, vital_input):
    """
    Full fusion inference: CNN + XGBoost → FusionHead → result figure.
    """
    # Load models
    cnn     = load_model(str(MDIR/"cnn_best.h5"))
    fusion  = load_model(str(MDIR/"fusion_head_best.h5"))
    scaler  = joblib.load(MDIR/"advanced_scaler.pkl")
    xgb     = joblib.load(MDIR/"advanced_risk_model.pkl")
    genc    = joblib.load(MDIR/"advanced_gender_encoder.pkl")
    cfg     = json.load(open(MDIR/"fusion_config.json"))
    fnames  = cfg["feature_names"]

    # Encode gender
    g = vital_input.get("Gender","Male")
    try: vital_input["Gender"] = int(genc.transform([g])[0])
    except: vital_input["Gender"] = 0

    v_raw = np.array([float(vital_input.get(f,0.0))
                       for f in fnames],
                      dtype=np.float32).reshape(1,-1)
    v_sc  = scaler.transform(v_raw).astype(np.float32)

    xgb_prob = xgb.predict_proba(v_sc)           # (1,2)

    arr, disp = _load_img(image_path)
    feat_model = Model(cnn.input,
                       [cnn.get_layer("dense_128").output, cnn.output])
    cnn_feat, cnn_prob = feat_model.predict(arr, verbose=0)
    cnn_prob_val = float(cnn_prob[0][0])
    img_feat = np.column_stack([cnn_feat, [[cnn_prob_val]]])

    vit_feat = np.concatenate([xgb_prob, v_sc], axis=1).astype(np.float32)

    fp     = float(fusion.predict([img_feat,vit_feat],verbose=0)[0][0])
    label  = CLASS_NAMES[int(fp>=0.5)]
    conf   = fp if fp>=0.5 else 1-fp

    print(f"\n  ╔══════════════════════════════════════════════╗")
    print(f"  ║  CNN Risk Score    : {cnn_prob_val*100:>6.2f}%            ║")
    print(f"  ║  XGBoost Risk      : {xgb_prob[0,1]*100:>6.2f}%            ║")
    print(f"  ║  FUSION PREDICTION : {label:<12}        ║")
    print(f"  ║  CONFIDENCE        : {conf*100:>6.2f}%            ║")
    print(f"  ╚══════════════════════════════════════════════╝")

    # Grad-CAM + Grad-CAM++
    cam  = GradCAM(cnn)
    capp = GradCAMPP(cnn)
    ci   = int(cnn_prob_val>=0.5)
    h1   = cam.heatmap(arr, ci)
    h2   = capp.heatmap(arr, ci)
    ov1  = cam.overlay(disp, h1)
    ov2  = capp.overlay(disp, h2)

    prefix = Path(image_path).stem
    sp = str(PDIR/f"{prefix}_result.png")
    _save_result_fig(disp, ov1, ov2, label, fp,
                     cnn_prob_val, xgb_prob[0,1],
                     vital_input, fnames, sp)

    return dict(label=label, confidence=round(conf,4),
                fusion_prob=round(fp,4),
                cnn_prob=round(cnn_prob_val,4),
                xgb_risk=round(float(xgb_prob[0,1]),4),
                figure=sp)


def _save_result_fig(orig, ov1, ov2, label, fp,
                      cnn_p, xgb_p, vital_vals, fnames, sp):
    _style()
    bc = "#ff3d5a" if label=="PNEUMONIA" else "#00ff9d"
    fig = plt.figure(figsize=(26,9), facecolor="#050a12")
    fig.suptitle(
        f"PneumoFusion  │  {label}  │  "
        f"Confidence: {(fp if fp>=0.5 else 1-fp)*100:.1f}%",
        fontsize=14, color="#00d4ff", fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(1,5, figure=fig, wspace=0.08)

    for col,(img,title) in enumerate([
        (orig,            "① Original X-ray"),
        (_lung_mask(orig),"② Lung ROI Mask"),
        (ov1,             "③ Grad-CAM"),
        (ov2,             "④ Grad-CAM++"),
    ]):
        ax = fig.add_subplot(gs[col])
        ax.imshow(img)
        ax.set_title(title, color="#00d4ff", fontsize=9, pad=5)
        ax.axis("off")
        for sp_ in ax.spines.values():
            sp_.set_edgecolor(bc); sp_.set_linewidth(2)

    ax5 = fig.add_subplot(gs[4]); ax5.set_facecolor("#0b1220")
    for i,(lbl,val,col) in enumerate([
        ("CNN Risk",    cnn_p,"#00d4ff"),
        ("XGBoost Risk",xgb_p,"#ff6b35"),
        ("Fusion Score",fp,   "#ffd166"),
    ]):
        ax5.barh([lbl],[val],color=col,edgecolor="#1a2a40",height=0.5)
        ax5.text(min(val+0.02,0.80),i,f"{val*100:.1f}%",
                 va="center",fontsize=10,color="white",fontweight="bold")
    ax5.set_xlim(0,1)
    ax5.set_title("⑤ Branch Scores",color="#00d4ff",fontsize=9,pad=5)
    ax5.tick_params(labelsize=8)
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(sp, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [output] Result figure → {sp}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 ── IMAGE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _load_img(path):
    img = cv2.imread(str(path))
    if img is None: raise FileNotFoundError(f"Cannot read: {path}")
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    disp = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    arr  = preprocess_input(disp.astype(np.float32))
    return np.expand_dims(arr,0), disp


def _lung_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    inv  = cv2.bitwise_not(gray)
    _,t  = cv2.threshold(inv,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    t    = cv2.morphologyEx(cv2.morphologyEx(
               t,cv2.MORPH_CLOSE,k,iterations=2),
               cv2.MORPH_OPEN,k,iterations=1)
    n,labels,stats,_ = cv2.connectedComponentsWithStats(t)
    areas = stats[1:,cv2.CC_STAT_AREA]
    if len(areas)>=2:
        top2 = np.argsort(areas)[-2:]+1
        mask = np.zeros_like(t)
        for i in top2: mask[labels==i]=255
    else:
        mask = t
    mask  = cv2.dilate(mask,k,iterations=1)
    m3    = np.stack([mask]*3,axis=-1)
    out   = img.copy(); out[m3==0]=0
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 ── CLI
# ══════════════════════════════════════════════════════════════════════════════

def _args():
    p = argparse.ArgumentParser(
        description="PneumoFusion XAI — One Complete File",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES
────────
  # Step 1 — Train
  python pneumofusion_xai.py --mode train \\
      --data_dir "chest-xray-pneumonia/chest_xray" \\
      --vitals_csv human_vital_signs_dataset_2024.csv

  # Step 2 — Predict
  python pneumofusion_xai.py --mode predict \\
      --image xray.jpeg \\
      --heart_rate 102 --body_temperature 38.5 \\
      --oxygen_saturation 94 --respiratory_rate 22 \\
      --systolic_bp 125 --diastolic_bp 82 \\
      --age 45 --gender Male --hrv 28.5 --map 96.3 --bmi 24.5

  # Step 3 — Compare Normal vs Pneumonia Grad-CAM
  python pneumofusion_xai.py --mode compare \\
      --normal_img    "data/test/NORMAL/im.jpeg" \\
      --pneumonia_img "data/test/PNEUMONIA/im.jpeg"

  # Step 4 — SHAP analysis
  python pneumofusion_xai.py --mode shap \\
      --vitals_csv human_vital_signs_dataset_2024.csv
        """)

    p.add_argument("--mode",
                   choices=["train","predict","compare","shap"],
                   required=True,
                   help="train | predict | compare | shap")

    # Train
    p.add_argument("--data_dir",
                   default="chest-xray-pneumonia/chest_xray")
    p.add_argument("--vitals_csv",
                   default="human_vital_signs_dataset_2024.csv")
    p.add_argument("--epochs",     type=int, default=EPOCHS_P1)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)

    # Predict
    p.add_argument("--image",             default=None)
    p.add_argument("--heart_rate",        type=float, default=80.0)
    p.add_argument("--body_temperature",  type=float, default=37.0)
    p.add_argument("--oxygen_saturation", type=float, default=97.0)
    p.add_argument("--respiratory_rate",  type=float, default=16.0)
    p.add_argument("--systolic_bp",       type=float, default=120.0)
    p.add_argument("--diastolic_bp",      type=float, default=80.0)
    p.add_argument("--age",               type=float, default=45.0)
    p.add_argument("--gender",            default="Male")
    p.add_argument("--hrv",               type=float, default=30.0)
    p.add_argument("--map",               type=float, default=93.0)
    p.add_argument("--bmi",               type=float, default=22.5)

    # Compare
    p.add_argument("--normal_img",    default=None)
    p.add_argument("--pneumonia_img", default=None)

    return p.parse_args()


def main():
    args = _args()

    print("\n╔" + "═"*62 + "╗")
    print("║  PneumoFusion XAI  —  XGBoost + ResNet50 + Grad-CAM + SHAP  ║")
    print("╚" + "═"*62 + "╝\n")

    # ── TRAIN ─────────────────────────────────────────────────────────────
    if args.mode == "train":
        if not os.path.isdir(args.data_dir):
            print(f"  [error] data_dir not found: {args.data_dir}")
            print("  Correct path example: chest-xray-pneumonia/chest_xray")
            sys.exit(1)
        if not os.path.isfile(args.vitals_csv):
            print(f"  [error] vitals CSV not found: {args.vitals_csv}")
            sys.exit(1)

        t0 = time.time()
        vital_results = train_vitals(args.vitals_csv)
        cnn_model     = train_cnn(args.data_dir, args.epochs,
                                   EPOCHS_P2, args.batch_size)
        fusion_model, n_vit = train_fusion(args.data_dir,
                                            vital_results, cnn_model)
        run_shap(vital_results)
        print(f"\n  ✅ Done in {(time.time()-t0)/60:.1f} min")
        print(f"  All outputs → {OUT.resolve()}")

    # ── PREDICT ───────────────────────────────────────────────────────────
    elif args.mode == "predict":
        if not args.image:
            print("  [error] --image required"); sys.exit(1)
        vitals = {
            "Heart Rate":               args.heart_rate,
            "Respiratory Rate":         args.respiratory_rate,
            "Body Temperature":         args.body_temperature,
            "Oxygen Saturation":        args.oxygen_saturation,
            "Systolic Blood Pressure":  args.systolic_bp,
            "Diastolic Blood Pressure": args.diastolic_bp,
            "Age":                      args.age,
            "Gender":                   args.gender,
            "Derived_HRV":              args.hrv,
            "Derived_MAP":              args.map,
            "Derived_BMI":              args.bmi,
        }
        result = predict_patient(args.image, vitals)
        print(json.dumps(result, indent=2))

    # ── COMPARE ───────────────────────────────────────────────────────────
    elif args.mode == "compare":
        if not args.normal_img or not args.pneumonia_img:
            print("  [error] --normal_img and --pneumonia_img required")
            sys.exit(1)
        cnn = load_model(str(MDIR/"cnn_best.h5"))
        compare_gradcam(cnn, args.normal_img, args.pneumonia_img)

    # ── SHAP ──────────────────────────────────────────────────────────────
    elif args.mode == "shap":
        if not os.path.isfile(args.vitals_csv):
            print(f"  [error] CSV not found: {args.vitals_csv}")
            sys.exit(1)
        vital_results = train_vitals(args.vitals_csv)
        run_shap(vital_results)


if __name__ == "__main__":
    main()