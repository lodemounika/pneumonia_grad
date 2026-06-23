"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                                                                                  ║
║        PneumoNet — ResNet50 Transfer Learning for Pneumonia Detection           ║
║        Complete Single-File Implementation                                       ║
║                                                                                  ║
║  Features:                                                                       ║
║    ✔ ResNet50 pretrained ImageNet feature extractor                              ║
║    ✔ 2-Phase training (frozen base → fine-tune top 30 layers)                   ║
║    ✔ Lung-ROI masking (Otsu + connected components)                              ║
║    ✔ Data augmentation (rotation, zoom, flip, shift, brightness)                 ║
║    ✔ Class-weighted loss for imbalanced data                                     ║
║    ✔ Metrics: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC/AUC        ║
║    ✔ Grad-CAM explainability (conv5_block3_out)                                  ║
║    ✔ SHAP DeepExplainer feature attribution                                      ║
║    ✔ Training curve plots (Accuracy / Loss / AUC)                                ║
║    ✔ Erase outputs utility                                                       ║
║                                                                                  ║
║  Dataset:                                                                        ║
║    Kaggle: paultimothymooney/chest-xray-pneumonia                                ║
║    data/train/NORMAL,  data/train/PNEUMONIA                                      ║
║    data/val/NORMAL,    data/val/PNEUMONIA                                        ║
║    data/test/NORMAL,   data/test/PNEUMONIA                                       ║
║                                                                                  ║
║  Usage:                                                                          ║
║    # Full train + evaluate                                                       ║
║    python pneumonet_complete.py --mode train --data_dir ./data                   ║
║                                                                                  ║
║    # Predict single image                                                        ║
║    python pneumonet_complete.py --mode predict \                                 ║
║        --model outputs/model/pneumonet_best.h5 \                                 ║
║        --image path/to/xray.jpeg                                                 ║
║                                                                                  ║
║    # Predict + SHAP                                                              ║
║    python pneumonet_complete.py --mode predict \                                 ║
║        --model outputs/model/pneumonet_best.h5 \                                 ║
║        --image path/to/xray.jpeg --data_dir ./data --shap                        ║
║                                                                                  ║
║    # Erase all outputs                                                           ║
║    python pneumonet_complete.py --mode erase                                     ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import shutil
import argparse
import warnings
import json
import time
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc,
    precision_score, recall_score, f1_score, accuracy_score,
)
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
from tqdm import tqdm

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[info] shap not installed — SHAP disabled.  pip install shap")

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL CONFIG
# ─────────────────────────────────────────────────────────────────────────────
SEED        = 42
IMG_SIZE    = 224
CHANNELS    = 3
BATCH_SIZE  = 32
EPOCHS_P1   = 20        # phase 1: frozen base
EPOCHS_P2   = 10        # phase 2: fine-tune
LR_P1       = 1e-4
LR_P2       = 1e-5
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

np.random.seed(SEED)
tf.random.set_seed(SEED)

# Output directories
OUT_DIR    = Path("outputs")
MODEL_DIR  = OUT_DIR / "model"
PLOT_DIR   = OUT_DIR / "plots"
RESULT_DIR = OUT_DIR / "results"

for _d in [OUT_DIR, MODEL_DIR, PLOT_DIR, RESULT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# Custom colormap for Grad-CAM
CMAP_GRADCAM = LinearSegmentedColormap.from_list(
    "pneumo_gradcam",
    ["#000000", "#0d2137", "#f39c12", "#e74c3c", "#ffffff"],
)

# ─────────────────────────────────────────────────────────────────────────────
#  PLOT STYLE
# ─────────────────────────────────────────────────────────────────────────────
def _apply_style():
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor" : "#0b1220",
        "axes.facecolor"   : "#0b1220",
        "axes.edgecolor"   : "#1a2a40",
        "axes.labelcolor"  : "#c0d0e0",
        "xtick.color"      : "#6080a0",
        "ytick.color"      : "#6080a0",
        "text.color"       : "#c0d0e0",
        "grid.color"       : "#1a2a40",
        "grid.linestyle"   : "--",
        "font.family"      : "monospace",
    })


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 ── DATA PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def build_data_generators(data_dir: str, batch_size: int = BATCH_SIZE):
    """
    Build augmented train generator and clean val/test generators.

    Preprocessing
    -------------
    • Resize to 224 × 224 px
    • Normalize via ResNet50 preprocess_input (ImageNet mean subtraction)

    Augmentation (train only)
    -------------------------
    • Rotation       ±15 °
    • Width shift    ±10 %
    • Height shift   ±10 %
    • Zoom           0.90 – 1.10 ×
    • Horizontal flip
    • Brightness     ±20 %
    """
    train_gen_cfg = ImageDataGenerator(
        preprocessing_function = preprocess_input,
        rotation_range         = 15,
        width_shift_range      = 0.10,
        height_shift_range     = 0.10,
        zoom_range             = 0.10,
        horizontal_flip        = True,
        brightness_range       = [0.80, 1.20],
        fill_mode              = "nearest",
    )
    eval_gen_cfg = ImageDataGenerator(preprocessing_function=preprocess_input)

    def _flow(cfg, split, shuffle=True):
        return cfg.flow_from_directory(
            os.path.join(data_dir, split),
            target_size = (IMG_SIZE, IMG_SIZE),
            batch_size  = batch_size,
            class_mode  = "binary",
            classes     = CLASS_NAMES,
            shuffle     = shuffle,
            seed        = SEED,
        )

    train_gen = _flow(train_gen_cfg, "train", shuffle=True)
    val_gen   = _flow(eval_gen_cfg,  "val",   shuffle=False)
    test_gen  = _flow(eval_gen_cfg,  "test",  shuffle=False)

    print(f"\n{'─'*58}")
    print(f"  Dataset : {data_dir}")
    print(f"  Train   : {train_gen.n:>6,}  images")
    print(f"  Val     : {val_gen.n:>6,}  images")
    print(f"  Test    : {test_gen.n:>6,}  images")
    print(f"  Classes : {train_gen.class_indices}")
    print(f"{'─'*58}\n")
    return train_gen, val_gen, test_gen


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 ── MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

def build_model(fine_tune: bool = False) -> tf.keras.Model:
    """
    Architecture
    ────────────
    ResNet50 (ImageNet, no top)
        └─ GlobalAveragePooling2D
           └─ Dense(512, relu) → BatchNorm → Dropout(0.50)
              └─ Dense(256, relu) → BatchNorm → Dropout(0.30)
                 └─ Dense(1, sigmoid)   ← P(PNEUMONIA)

    Parameters
    ----------
    fine_tune : bool
        False → entire ResNet50 frozen  (Phase 1)
        True  → top 30 layers trainable (Phase 2)
    """
    base = ResNet50(
        weights     = "imagenet",
        include_top = False,
        input_shape = (IMG_SIZE, IMG_SIZE, CHANNELS),
    )
    base.trainable = fine_tune
    if fine_tune:
        # Freeze everything except top 30 layers
        for layer in base.layers[:-30]:
            layer.trainable = False

    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, CHANNELS), name="input")
    x   = base(inp, training=False)
    x   = layers.GlobalAveragePooling2D(name="gap")(x)
    x   = layers.Dense(512, activation="relu",    name="dense_512")(x)
    x   = layers.BatchNormalization(name="bn_512")(x)
    x   = layers.Dropout(0.50, name="drop_512")(x)
    x   = layers.Dense(256, activation="relu",    name="dense_256")(x)
    x   = layers.BatchNormalization(name="bn_256")(x)
    x   = layers.Dropout(0.30, name="drop_256")(x)
    out = layers.Dense(1,   activation="sigmoid", name="output")(x)

    return models.Model(inp, out, name="PneumoNet_ResNet50")


def compile_model(model: tf.keras.Model, lr: float) -> tf.keras.Model:
    model.compile(
        optimizer = optimizers.Adam(learning_rate=lr),
        loss      = "binary_crossentropy",
        metrics   = [
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 ── TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def _get_callbacks(model_path: str) -> list:
    return [
        callbacks.ModelCheckpoint(
            filepath       = model_path,
            monitor        = "val_auc",
            mode           = "max",
            save_best_only = True,
            verbose        = 1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = 0.5,
            patience = 3,
            min_lr   = 1e-7,
            verbose  = 1,
        ),
        callbacks.EarlyStopping(
            monitor              = "val_auc",
            patience             = 5,
            mode                 = "max",
            restore_best_weights = True,
            verbose              = 1,
        ),
        callbacks.CSVLogger(str(RESULT_DIR / "training_log.csv")),
    ]


def _class_weights(gen) -> dict:
    w = compute_class_weight("balanced",
                             classes = np.unique(gen.classes),
                             y       = gen.classes)
    return dict(enumerate(w))


def _merge_histories(h1_hist: dict, h2_hist: dict) -> dict:
    merged = {}
    for k in h1_hist:
        merged[k] = h1_hist[k] + h2_hist.get(k, [])
    return merged


def run_training(data_dir: str,
                 epochs_p1: int = EPOCHS_P1,
                 epochs_p2: int = EPOCHS_P2,
                 batch_size: int = BATCH_SIZE) -> tuple:
    """
    Two-phase training:
      Phase 1 — ResNet50 frozen,  train head only  (LR = 1e-4)
      Phase 2 — Top 30 unfrozen,  fine-tune        (LR = 1e-5)
    """
    train_gen, val_gen, test_gen = build_data_generators(data_dir, batch_size)
    model_path = str(MODEL_DIR / "pneumonet_best.keras")
    cw = _class_weights(train_gen)

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    print("\n" + "═"*58)
    print("  PHASE 1 — Training head  (ResNet50 frozen)")
    print("═"*58)
    model = compile_model(build_model(fine_tune=False), lr=LR_P1)
    model.summary(line_length=78)

    h1 = model.fit(
        train_gen,
        validation_data = val_gen,
        epochs          = epochs_p1,
        callbacks       = _get_callbacks(model_path),
        class_weight    = cw,
        verbose         = 1,
    )

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    print("\n" + "═"*58)
    print("  PHASE 2 — Fine-tuning top 30 ResNet50 layers")
    print("═"*58)
    model_ft = compile_model(build_model(fine_tune=True), lr=LR_P2)
    model_ft.load_weights(model_path)

    h2 = model_ft.fit(
        train_gen,
        validation_data = val_gen,
        epochs          = epochs_p2,
        callbacks       = _get_callbacks(model_path),
        class_weight    = cw,
        verbose         = 1,
    )

    # Merge & save history
    history = _merge_histories(h1.history, h2.history)
    with open(RESULT_DIR / "training_history.json", "w") as f:
        json.dump({k: [float(v) for v in vs]
                   for k, vs in history.items()}, f, indent=2)

    # Generate training plots
    plot_training_curves(history)

    # Load best weights and evaluate
    best_model = load_model(model_path)
    run_evaluation(best_model, test_gen)

    return best_model, test_gen


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 ── EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation(model: tf.keras.Model, test_gen) -> dict:
    """
    Compute and save all performance metrics.

    Outputs
    -------
    outputs/results/metrics.json
    outputs/plots/training_curves.png
    outputs/plots/confusion_matrix.png
    outputs/plots/roc_curve.png
    """
    print("\n" + "═"*58)
    print("  EVALUATION ON TEST SET")
    print("═"*58)

    test_gen.reset()
    y_prob = model.predict(test_gen, verbose=1).ravel()
    y_true = test_gen.classes[: len(y_prob)]
    y_pred = (y_prob >= 0.5).astype(int)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    print(f"\n  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=CLASS_NAMES)}")

    metrics = {
        "accuracy" : round(float(acc),     4),
        "precision": round(float(prec),    4),
        "recall"   : round(float(rec),     4),
        "f1_score" : round(float(f1),      4),
        "auc"      : round(float(roc_auc), 4),
    }
    with open(RESULT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curve(fpr, tpr, roc_auc)

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 ── PERFORMANCE PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_curves(history: dict):
    """
    Save accuracy, loss, and AUC curves for both train and validation sets.
    Saved to: outputs/plots/training_curves.png
    """
    _apply_style()
    epochs = range(1, len(history["accuracy"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    fig.suptitle("PneumoNet — Training Curves",
                 fontsize=15, color="#00d4ff", fontweight="bold", y=1.02)

    cfg = [
        ("accuracy",  "val_accuracy",  "Accuracy",  "#00d4ff", "#ff6b35"),
        ("loss",      "val_loss",      "Loss",      "#00ff9d", "#ff3d5a"),
        ("auc",       "val_auc",       "AUC",       "#ffd166", "#c77dff"),
    ]
    for ax, (tk, vk, title, c1, c2) in zip(axes, cfg):
        if tk in history:
            ax.plot(epochs, history[tk], color=c1, lw=2.5,
                    label=f"Train {title}")
            ax.fill_between(epochs, history[tk], alpha=0.07, color=c1)
        if vk in history:
            ax.plot(epochs, history[vk], color=c2, lw=2.5, ls="--",
                    label=f"Val {title}", alpha=0.85)
            ax.fill_between(epochs, history[vk], alpha=0.05, color=c2)
        ax.set_title(title, color="#00d4ff", fontsize=12)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(title,   fontsize=10)
        ax.legend(fontsize=9, facecolor="#0b1220", edgecolor="#1a2a40")
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = PLOT_DIR / "training_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [plot] Training curves  → {path}")


def plot_confusion_matrix(y_true, y_pred):
    """
    Save confusion matrix heatmap.
    Saved to: outputs/plots/confusion_matrix.png
    """
    _apply_style()
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle("Confusion Matrix", fontsize=13,
                 color="#00d4ff", fontweight="bold")

    sns.heatmap(
        cm,
        annot       = True,
        fmt         = "d",
        cmap        = "Blues",
        xticklabels = CLASS_NAMES,
        yticklabels = CLASS_NAMES,
        linewidths  = 0.5,
        linecolor   = "#1a2a40",
        ax          = ax,
        annot_kws   = {"size": 16, "weight": "bold"},
    )
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.72,
                    f"({cm[i, j] / total * 100:.1f}%)",
                    ha="center", va="center",
                    fontsize=9, color="#aabbcc")

    ax.set_xlabel("Predicted Label", labelpad=8)
    ax.set_ylabel("True Label",      labelpad=8)
    ax.tick_params(colors="#c0d0e0")

    plt.tight_layout()
    path = PLOT_DIR / "confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [plot] Confusion matrix → {path}")


def plot_roc_curve(fpr, tpr, roc_auc):
    """
    Save ROC curve with AUC.
    Saved to: outputs/plots/roc_curve.png
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle("ROC Curve", fontsize=13,
                 color="#00d4ff", fontweight="bold")

    ax.plot(fpr, tpr, color="#00d4ff", lw=2.5,
            label=f"PneumoNet  (AUC = {roc_auc:.4f})")
    ax.fill_between(fpr, tpr, alpha=0.12, color="#00d4ff")
    ax.plot([0, 1], [0, 1], color="#4a6080", lw=1, ls="--",
            label="Random baseline")

    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate (1 – Specificity)", fontsize=10)
    ax.set_ylabel("True Positive Rate (Sensitivity)",      fontsize=10)
    ax.legend(fontsize=10, loc="lower right",
              facecolor="#0b1220", edgecolor="#1a2a40")
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = PLOT_DIR / "roc_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [plot] ROC curve        → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 ── PREPROCESSING  (Lung ROI Mask)
# ══════════════════════════════════════════════════════════════════════════════

def apply_lung_roi_mask(image: np.ndarray) -> np.ndarray:
    """
    Isolate lung regions using Otsu thresholding + morphology.
    Zeroes out background, spine edges, and non-lung regions.

    Parameters
    ----------
    image : uint8 (H, W, 3)  RGB

    Returns
    -------
    masked : uint8 (H, W, 3)  — non-lung pixels set to 0
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    inv  = cv2.bitwise_not(gray)                       # lungs appear dark

    _, thresh = cv2.threshold(inv, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kern   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kern, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN,  kern, iterations=1)

    # Keep the 2 largest connected components (left + right lung)
    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
    areas = stats[1:, cv2.CC_STAT_AREA]          # skip background label 0
    if len(areas) >= 2:
        top2 = np.argsort(areas)[-2:] + 1
        mask = np.zeros_like(opened)
        for idx in top2:
            mask[labels == idx] = 255
    else:
        mask = opened

    mask = cv2.dilate(mask, kern, iterations=1)  # include pleura
    mask_3c = np.stack([mask] * 3, axis=-1)

    masked = image.copy()
    masked[mask_3c == 0] = 0
    return masked


def load_and_preprocess(image_path: str) -> tuple:
    """
    Load image from disk and prepare for model inference.

    Returns
    -------
    img_array   : float32  (1, 224, 224, 3)  — ResNet50-normalised
    img_display : uint8    (224, 224, 3)     — original RGB for display
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    img         = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_display = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_array   = preprocess_input(img_display.astype(np.float32))
    img_array   = np.expand_dims(img_array, axis=0)
    return img_array, img_display


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 ── GRAD-CAM
# ══════════════════════════════════════════════════════════════════════════════

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).

    Uses the gradients of the target class signal flowing into the
    last ResNet50 convolutional layer (conv5_block3_out) to generate
    a spatial heatmap highlighting discriminative lung regions.

    Reference
    ---------
    Selvaraju et al. 2017 — https://arxiv.org/abs/1610.02391
    """

    def __init__(self, model: tf.keras.Model,
                 layer_name: str = "conv5_block3_out"):
        self.model      = model
        self.layer_name = layer_name
        self.grad_model = tf.keras.Model(
            inputs  = model.inputs,
            outputs = [model.get_layer(layer_name).output, model.output],
        )

    def compute_heatmap(self, img_array: np.ndarray,
                        class_idx: int = 1) -> np.ndarray:
        """
        Parameters
        ----------
        img_array : (1, 224, 224, 3)  preprocessed
        class_idx : 1 = PNEUMONIA, 0 = NORMAL

        Returns
        -------
        heatmap : float32 (224, 224) normalised to [0, 1]
        """
        img_tensor = tf.cast(img_array, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            conv_out, preds = self.grad_model(img_tensor, training=False)
            # Binary sigmoid: use raw output or its complement
            score = preds[:, 0] if class_idx == 1 else (1 - preds[:, 0])

        grads       = tape.gradient(score, conv_out)      # (1, H, W, C)
        pooled      = tf.reduce_mean(grads, axis=(0,1,2)) # (C,)
        heatmap     = conv_out[0] @ pooled[..., tf.newaxis]
        heatmap     = tf.squeeze(tf.nn.relu(heatmap)).numpy()

        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE)).astype(np.float32)

    def overlay_heatmap(self, img_display: np.ndarray,
                        heatmap: np.ndarray,
                        alpha: float = 0.45) -> np.ndarray:
        """Blend JET colormap heatmap onto original image."""
        h_uint8  = np.uint8(255 * heatmap)
        colormap = cv2.applyColorMap(h_uint8, cv2.COLORMAP_JET)
        colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
        blended  = ((1 - alpha) * img_display + alpha * colormap
                    ).clip(0, 255).astype(np.uint8)
        return blended


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 ── SHAP
# ══════════════════════════════════════════════════════════════════════════════

class SHAPAnalyzer:
    """
    SHAP GradientExplainer for pixel-level feature attribution.

    Uses a background distribution of normal X-rays to establish a
    baseline expectation, then computes Shapley values for each input pixel.

    Reference
    ---------
    Lundberg & Lee 2017 — https://arxiv.org/abs/1705.07874
    """

    def __init__(self, model: tf.keras.Model, background: np.ndarray):
        """
        Parameters
        ----------
        model      : trained Keras model
        background : float32 (N, 224, 224, 3)  reference (NORMAL) images
        """
        if not SHAP_AVAILABLE:
            raise RuntimeError("Install shap: pip install shap")
        self.model     = model
        self.explainer = shap.GradientExplainer(model, background)

    def explain(self, img_array: np.ndarray) -> np.ndarray:
        """Returns SHAP values: float32 (1, 224, 224, 3)"""
        sv = self.explainer.shap_values(img_array)
        return sv[0] if isinstance(sv, list) else sv

    def plot_shap(self, shap_values: np.ndarray,
                  img_array:   np.ndarray,
                  save_path:   str):
        """
        3-panel SHAP figure:
          Left  — |SHAP| pixel heatmap
          Mid   — Input X-ray (display normalised)
          Right — Bar chart of top 10 activated grid regions
        """
        _apply_style()
        sv_abs  = np.abs(shap_values[0]).mean(axis=-1)     # (224, 224)

        # Grid-block importance (8 × 8 blocks)
        block = IMG_SIZE // 8
        labels, vals = [], []
        for i in range(8):
            for j in range(8):
                region = sv_abs[i*block:(i+1)*block, j*block:(j+1)*block]
                labels.append(f"R({i},{j})")
                vals.append(float(region.mean()))

        top_idx = np.argsort(vals)[-10:][::-1]

        fig = plt.figure(figsize=(17, 6), facecolor="#0b1220")
        fig.suptitle("SHAP Feature Attribution",
                     fontsize=14, color="#00d4ff", fontweight="bold")
        gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.32)

        # Panel 1 – SHAP heatmap
        ax1 = fig.add_subplot(gs[0])
        im  = ax1.imshow(sv_abs, cmap="hot")
        plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        ax1.set_title("|SHAP| Pixel Map", color="#00d4ff", fontsize=11)
        ax1.axis("off")

        # Panel 2 – Input image (de-normalise for display)
        ax2  = fig.add_subplot(gs[1])
        disp = img_array[0].copy()
        disp = (disp - disp.min()) / (disp.max() - disp.min() + 1e-8)
        ax2.imshow(disp)
        ax2.set_title("Input X-ray\n(preprocessed)", color="#00d4ff",
                      fontsize=11)
        ax2.axis("off")

        # Panel 3 – Bar chart
        ax3 = fig.add_subplot(gs[2])
        ax3.set_facecolor("#0b1220")
        bar_vals   = [vals[i] for i in top_idx]
        bar_labels = [labels[i] for i in top_idx]
        bar_colors = ["#00d4ff" if v > np.median(bar_vals) else "#ff6b35"
                      for v in bar_vals]
        ax3.barh(bar_labels, bar_vals, color=bar_colors, edgecolor="#1a2a40")
        ax3.set_xlabel("|SHAP| mean", color="#c0d0e0", fontsize=9)
        ax3.set_title("Top Activated\nGrid Regions",
                      color="#00d4ff", fontsize=11)
        ax3.tick_params(colors="#6080a0")
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)

        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  [plot] SHAP analysis    → {save_path}")


def _load_background_images(data_dir: str, n: int = 50) -> np.ndarray:
    """Load N normal X-rays as SHAP background reference."""
    normal_dir = Path(data_dir) / "train" / "NORMAL"
    exts = {".jpeg", ".jpg", ".png"}
    files = [f for f in normal_dir.rglob("*") if f.suffix.lower() in exts][:n]
    bg = []
    for f in files:
        arr, _ = load_and_preprocess(str(f))
        bg.append(arr[0])
    return np.array(bg, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 ── PREDICTION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def predict_image(model_path: str,
                  image_path: str,
                  data_dir:   str  = None,
                  run_shap:   bool = False) -> dict:
    """
    Complete inference pipeline for a single chest X-ray.

    Steps
    -----
    1. Load & ResNet50-preprocess image
    2. Apply lung ROI mask
    3. Forward pass → label + confidence
    4. Grad-CAM heatmap (conv5_block3_out)
    5. Composite 5-panel result figure
    6. SHAP attribution (if run_shap=True and data_dir provided)

    Parameters
    ----------
    model_path : str   path to .h5 model
    image_path : str   path to chest X-ray image
    data_dir   : str   dataset root (for SHAP background)
    run_shap   : bool  whether to run SHAP analysis

    Returns
    -------
    result : dict
        label, confidence, p_pneumonia, p_normal,
        gradcam_path, shap_path
    """
    print(f"\n{'─'*58}")
    print(f"  Image : {image_path}")
    print(f"  Model : {model_path}")
    print(f"{'─'*58}")

    model = load_model(model_path)
    img_array, img_display = load_and_preprocess(image_path)

    # ── Lung ROI ──────────────────────────────────────────────────────────
    img_roi = apply_lung_roi_mask(img_display)

    # ── Prediction ────────────────────────────────────────────────────────
    prob       = float(model.predict(img_array, verbose=0)[0][0])
    label      = CLASS_NAMES[int(prob >= 0.5)]
    confidence = prob if prob >= 0.5 else 1 - prob

    print(f"\n  ┌───────────────────────────────────────────────┐")
    print(f"  │  PREDICTION : {label:<8}                       │")
    print(f"  │  CONFIDENCE : {confidence*100:>6.2f}%                         │")
    print(f"  │  P(Pneumonia) = {prob:.4f}   P(Normal) = {1-prob:.4f}  │")
    print(f"  └───────────────────────────────────────────────┘")

    # ── Grad-CAM ──────────────────────────────────────────────────────────
    cam       = GradCAM(model)
    heatmap   = cam.compute_heatmap(img_array, class_idx=int(prob >= 0.5))
    overlay   = cam.overlay_heatmap(img_display, heatmap)

    prefix         = Path(image_path).stem
    gradcam_path   = str(PLOT_DIR / f"{prefix}_gradcam.png")
    _save_gradcam_figure(img_display, img_roi, heatmap, overlay,
                         label, prob, confidence, gradcam_path)

    result = {
        "label"       : label,
        "confidence"  : round(confidence, 4),
        "p_pneumonia" : round(prob,        4),
        "p_normal"    : round(1 - prob,    4),
        "gradcam_path": gradcam_path,
        "shap_path"   : None,
    }

    # ── SHAP ──────────────────────────────────────────────────────────────
    if run_shap:
        if not SHAP_AVAILABLE:
            print("  [skip] SHAP not installed.")
        elif not data_dir:
            print("  [skip] --data_dir required for SHAP.")
        else:
            try:
                print("\n  Running SHAP (this may take a minute)…")
                background = _load_background_images(data_dir, n=50)
                analyzer   = SHAPAnalyzer(model, background)
                sv         = analyzer.explain(img_array)
                shap_path  = str(PLOT_DIR / f"{prefix}_shap.png")
                analyzer.plot_shap(sv, img_array, shap_path)
                result["shap_path"] = shap_path
            except Exception as exc:
                print(f"  [warn] SHAP failed: {exc}")

    print(f"\n  Grad-CAM figure → {gradcam_path}")
    return result


def _save_gradcam_figure(original, roi_masked, heatmap, overlay,
                          label, prob, confidence, save_path):
    """5-panel composite: Original | ROI Mask | Heatmap | Overlay | Probabilities."""
    _apply_style()
    border_col = "#ff3d5a" if label == "PNEUMONIA" else "#00ff9d"

    fig = plt.figure(figsize=(22, 5), facecolor="#050a12")
    fig.suptitle(
        f"PneumoNet Grad-CAM  │  Prediction: {label}"
        f"  │  Confidence: {confidence*100:.1f}%",
        fontsize=13, color="#00d4ff", fontweight="bold", y=1.02,
    )
    gs = gridspec.GridSpec(1, 5, figure=fig, wspace=0.06)

    panels = [
        (original,   "1. Original X-ray",    None),
        (roi_masked, "2. Lung ROI Mask",      None),
        (heatmap,    "3. Grad-CAM Heatmap",   CMAP_GRADCAM),
        (overlay,    "4. Grad-CAM Overlay",   None),
    ]
    for i, (img, title, cmap) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        kw = {"cmap": cmap, "vmin": 0, "vmax": 1} if cmap else {}
        im = ax.imshow(img, **kw)
        if cmap:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, color="#00d4ff", fontsize=9, pad=5)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(border_col)
            spine.set_linewidth(2)

    # Panel 5 — probability bar chart
    ax5 = fig.add_subplot(gs[4])
    ax5.set_facecolor("#0b1220")
    bars = ax5.barh(
        ["NORMAL", "PNEUMONIA"],
        [1 - prob,  prob],
        color  = ["#00ff9d", "#ff3d5a"],
        edgecolor = "#1a2a40",
        height = 0.5,
    )
    for bar, val in zip(bars, [1-prob, prob]):
        ax5.text(min(val + 0.02, 0.85),
                 bar.get_y() + bar.get_height() / 2,
                 f"{val*100:.1f}%",
                 va="center", fontsize=11,
                 color="white", fontweight="bold")
    ax5.set_xlim(0, 1)
    ax5.set_xlabel("Probability", color="#c0d0e0", fontsize=9)
    ax5.set_title("5. Class\nProbabilities", color="#00d4ff",
                  fontsize=9, pad=5)
    ax5.tick_params(colors="#6080a0")
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)
    for spine in ["left", "bottom"]:
        ax5.spines[spine].set_edgecolor(border_col)
        ax5.spines[spine].set_linewidth(1.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [plot] Grad-CAM figure  → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 ── ERASE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

def erase_outputs(target: str = "all", confirm: bool = False):
    """
    Delete generated files.

    Parameters
    ----------
    target  : "all"    — erase entire outputs/ directory
              "plots"  — erase only outputs/plots/*.png
              "model"  — erase only outputs/model/*.h5
    confirm : bool     — must be True to actually delete (safety guard)

    Examples
    --------
    from pneumonet_complete import erase_outputs
    erase_outputs("all",   confirm=True)
    erase_outputs("plots", confirm=True)
    erase_outputs("model", confirm=True)
    """
    target_map = {
        "all"   : OUT_DIR,
        "plots" : PLOT_DIR,
        "model" : MODEL_DIR,
    }
    if target not in target_map:
        print(f"  [error] target must be one of: {list(target_map)}")
        return

    path = target_map[target]
    if not path.exists():
        print(f"  [info] Nothing to erase — {path} does not exist.")
        return

    if not confirm:
        print(f"  [warn] Set confirm=True to erase: {path.resolve()}")
        _print_tree(path)
        return

    if target == "all":
        shutil.rmtree(path)
        print(f"  ✓ Erased all outputs from {path.resolve()}")
    else:
        exts = {".png", ".jpg", ".jpeg", ".h5"}
        deleted = 0
        for f in path.rglob("*"):
            if f.is_file() and f.suffix.lower() in exts:
                f.unlink()
                deleted += 1
                print(f"  Deleted: {f.name}")
        print(f"  ✓ Deleted {deleted} file(s) from {path.resolve()}")


def _print_tree(root: Path):
    print(f"\n  Contents of {root}:")
    total_mb = 0.0
    for f in sorted(root.rglob("*")):
        if f.is_file():
            mb = f.stat().st_size / 1_048_576
            total_mb += mb
            rel = f.relative_to(root)
            print(f"    {str(rel):<50} {mb:>6.2f} MB")
    print(f"    {'─'*58}")
    print(f"    Total: {total_mb:.2f} MB\n")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 11 ── CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description="PneumoNet — ResNet50 Pneumonia Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Train on Kaggle chest-xray-pneumonia dataset
  python pneumonet_complete.py --mode train --data_dir ./data

  # Predict single image + Grad-CAM
  python pneumonet_complete.py --mode predict \\
      --model outputs/model/pneumonet_best.h5 \\
      --image path/to/xray.jpeg

  # Predict + SHAP
  python pneumonet_complete.py --mode predict \\
      --model outputs/model/pneumonet_best.h5 \\
      --image path/to/xray.jpeg \\
      --data_dir ./data --shap

  # Erase all outputs
  python pneumonet_complete.py --mode erase

  # Erase only plots
  python pneumonet_complete.py --mode erase --erase_target plots
        """,
    )
    p.add_argument("--mode",
                   choices=["train", "predict", "erase"],
                   required=True,
                   help="Operating mode")

    # Training args
    p.add_argument("--data_dir",   type=str, default="./data",
                   help="Root data directory (train/val/test splits)")
    p.add_argument("--epochs",     type=int, default=EPOCHS_P1,
                   help="Phase-1 epochs (default: 20)")
    p.add_argument("--epochs_ft",  type=int, default=EPOCHS_P2,
                   help="Phase-2 fine-tune epochs (default: 10)")
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                   help="Batch size (default: 32)")

    # Prediction args
    p.add_argument("--model",      type=str, default=None,
                   help="Path to .h5 model file (predict mode)")
    p.add_argument("--image",      type=str, default=None,
                   help="Path to chest X-ray image (predict mode)")
    p.add_argument("--shap",       action="store_true",
                   help="Run SHAP analysis (requires --data_dir)")

    # Erase args
    p.add_argument("--erase_target",
                   choices=["all", "plots", "model"],
                   default="all",
                   help="What to erase (default: all)")

    return p.parse_args()


def main():
    args = _parse_args()
    t0   = time.time()

    print("\n" + "╔" + "═"*56 + "╗")
    print("║  PneumoNet — ResNet50 Pneumonia Detector" + " "*15 + "║")
    print("╚" + "═"*56 + "╝\n")

    # ── TRAIN ─────────────────────────────────────────────────────────────
    if args.mode == "train":
        if not os.path.isdir(args.data_dir):
            print(f"  [error] data_dir not found: {args.data_dir}")
            print("  Download: kaggle datasets download "
                  "paultimothymooney/chest-xray-pneumonia")
            sys.exit(1)
        model, test_gen = run_training(
            data_dir  = args.data_dir,
            epochs_p1 = args.epochs,
            epochs_p2 = args.epochs_ft,
            batch_size = args.batch_size,
        )
        elapsed = (time.time() - t0) / 60
        print(f"\n  Training complete in {elapsed:.1f} min")
        print(f"  Outputs saved to: {OUT_DIR.resolve()}")

    # ── PREDICT ───────────────────────────────────────────────────────────
    elif args.mode == "predict":
        if not args.model or not args.image:
            print("  [error] --model and --image required in predict mode.")
            sys.exit(1)
        result = predict_image(
            model_path = args.model,
            image_path = args.image,
            data_dir   = args.data_dir,
            run_shap   = args.shap,
        )
        print("\n  Full result:")
        print(json.dumps(result, indent=2))

    # ── ERASE ─────────────────────────────────────────────────────────────
    elif args.mode == "erase":
        erase_outputs(target=args.erase_target, confirm=True)


if __name__ == "__main__":
    main()
