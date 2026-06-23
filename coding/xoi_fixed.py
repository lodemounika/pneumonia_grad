"""
EXPLAINABLE AI - Grad-CAM + SHAP + Fusion XAI
Run: python xoi_fixed.py
No arguments needed - just run and get all outputs
"""
import os, sys, json, warnings
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

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from pathlib import Path

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False

# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR   = "chest-xray-pneumonia/chest_xray"
VITALS_CSV = "human_vital_signs_dataset_2024.csv"
MODEL_DIR  = Path("pneumofusion_outputs/models")
OUT_DIR    = Path("xai_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE    = 224
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
SEED        = 42
np.random.seed(SEED)

VITAL_FEATURES = [
    "Heart Rate", "Respiratory Rate", "Body Temperature",
    "Oxygen Saturation", "Systolic Blood Pressure",
    "Diastolic Blood Pressure", "Age", "Gender",
    "Derived_HRV", "Derived_MAP", "Derived_BMI",
]

SAMPLE_VITALS = {
    "Heart Rate": 102.0, "Respiratory Rate": 22.0,
    "Body Temperature": 38.5, "Oxygen Saturation": 94.0,
    "Systolic Blood Pressure": 125.0, "Diastolic Blood Pressure": 82.0,
    "Age": 45.0, "Gender": "Male",
    "Derived_HRV": 28.5, "Derived_MAP": 96.3, "Derived_BMI": 24.5,
}


# ══════════════════════════════════════════════════════════════════════════════
# PLOT STYLE
# ══════════════════════════════════════════════════════════════════════════════
def _style():
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor": "#050a12",
        "axes.facecolor":   "#0b1220",
        "axes.edgecolor":   "#1a2a40",
        "axes.labelcolor":  "#c0d0e0",
        "xtick.color":      "#6080a0",
        "ytick.color":      "#6080a0",
        "text.color":       "#c0d0e0",
        "grid.color":       "#1a2a40",
        "grid.linestyle":   "--",
        "font.family":      "monospace",
    })


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    img         = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_display = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_array   = preprocess_input(img_display.astype(np.float32))
    return np.expand_dims(img_array, 0), img_display


def lung_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    inv  = cv2.bitwise_not(gray)
    _, t = cv2.threshold(inv, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    t    = cv2.morphologyEx(t, cv2.MORPH_CLOSE, k, iterations=2)
    t    = cv2.morphologyEx(t, cv2.MORPH_OPEN,  k, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(t)
    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) >= 2:
        top2 = np.argsort(areas)[-2:] + 1
        mask = np.zeros_like(t)
        for i in top2:
            mask[labels == i] = 255
    else:
        mask = t
    mask  = cv2.dilate(mask, k, iterations=1)
    mask3 = np.stack([mask] * 3, axis=-1)
    out   = img.copy()
    out[mask3 == 0] = 0
    return out


def find_test_images():
    normal_dir = Path(DATA_DIR) / "test" / "NORMAL"
    pneumo_dir = Path(DATA_DIR) / "test" / "PNEUMONIA"
    exts = {".jpeg", ".jpg", ".png"}
    n_imgs = [f for f in normal_dir.rglob("*")
               if f.suffix.lower() in exts]
    p_imgs = [f for f in pneumo_dir.rglob("*")
               if f.suffix.lower() in exts]
    if not n_imgs:
        raise FileNotFoundError(f"No NORMAL images in {normal_dir}")
    if not p_imgs:
        raise FileNotFoundError(f"No PNEUMONIA images in {pneumo_dir}")
    return str(n_imgs[0]), str(p_imgs[0])


# ══════════════════════════════════════════════════════════════════════════════
# GRAD-CAM  — FIXED VERSION (no grad_model, uses tf.Variable)
# ══════════════════════════════════════════════════════════════════════════════
"""
HOW GRAD-CAM WORKS:
  1. Run image through ResNet50 → get conv feature maps (7x7x2048)
  2. Continue through rest of model layers → get prediction
  3. Compute gradients of prediction w.r.t. conv feature maps
  4. Average pool gradients → channel importance weights
  5. Weighted sum of feature maps → heatmap
  6. ReLU + normalize + resize to 224x224
  7. Overlay on original image

NORMAL lungs:   Low uniform activation → cool/blue colors
PNEUMONIA lungs: High focal activation → hot/red colors in lower lobes
"""


def compute_gradcam(model, img_array, class_idx=1):
    """
    Grad-CAM using tf.Variable to avoid tensor graph KeyError.
    This is the CORRECT approach for Keras functional models.
    """
    # Get the resnet50 sub-model inside our CNN
    resnet = model.get_layer("resnet50")

    # Use tf.Variable so GradientTape can watch it automatically
    img_t = tf.Variable(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        # Step 1: Run through ResNet50 → get conv feature maps
        conv_out = resnet(img_t, training=False)  # shape: (1, 7, 7, 2048)

        # Step 2: Run through remaining layers after resnet50
        x = conv_out
        after_resnet = False
        for layer in model.layers:
            if layer.name == "resnet50":
                after_resnet = True
                continue
            if after_resnet:
                x = layer(x, training=False)

        preds = x  # final sigmoid output

        # Step 3: Get score for target class
        score = preds[:, 0] if class_idx == 1 else (1 - preds[:, 0])

    # Step 4: Compute gradients w.r.t. conv feature maps
    grads = tape.gradient(score, conv_out)

    if grads is None:
        print("    [warn] Gradients are None, returning blank heatmap")
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    # Step 5: Global average pool → channel weights
    pooled  = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    conv_np = conv_out.numpy()[0]  # shape: (7, 7, 2048)

    # Step 6: Weighted sum of feature maps
    heatmap = np.zeros(conv_np.shape[:2], dtype=np.float32)
    for i, w in enumerate(pooled):
        heatmap += w * conv_np[:, :, i]

    # Step 7: ReLU + normalize + resize
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    return heatmap.astype(np.float32)


def overlay_heatmap(img_display, heatmap, alpha=0.45,
                     colormap=cv2.COLORMAP_JET):
    h_u8  = np.uint8(255 * heatmap)
    cmap  = cv2.applyColorMap(h_u8, colormap)
    cmap  = cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)
    return ((1 - alpha) * img_display
            + alpha * cmap).clip(0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# XAI 1 — NORMAL vs PNEUMONIA GRAD-CAM COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
"""
NORMAL vs PNEUMONIA DIFFERENCE:

NORMAL:
  - Low uniform activation across both lungs
  - Cool blue/cyan colors
  - No concentrated hotspots
  - Model sees clear air spaces = healthy

PNEUMONIA:
  - High focal activation in infected regions
  - Hot orange/red colors
  - Concentrated hotspots in lower lobes
  - Model sees dense white patches = infection
"""


def xai_gradcam_comparison(cnn_model, normal_path, pneumo_path):
    print("\n  Running Grad-CAM Comparison...")
    _style()

    results = {}
    for tag, path in [("normal", normal_path),
                       ("pneumonia", pneumo_path)]:
        arr, disp = load_image(path)
        cls  = 0 if tag == "normal" else 1
        prob = float(cnn_model.predict(arr, verbose=0)[0][0])

        print(f"    Computing Grad-CAM for {tag}...")
        h1 = compute_gradcam(cnn_model, arr, cls)

        cam_c = (cv2.COLORMAP_COOL if tag == "normal"
                  else cv2.COLORMAP_JET)

        results[tag] = {
            "disp" : disp,
            "roi"  : lung_mask(disp),
            "cam"  : overlay_heatmap(disp, h1, colormap=cam_c),
            "h1"   : h1,
            "prob" : prob,
        }
        print(f"    {tag}: P(Pneumonia)={prob*100:.1f}%  "
              f"Mean activation={h1.mean():.4f}")

    col_c = {"normal": "#00ff9d", "pneumonia": "#ff3d5a"}
    rows  = [
        "Original + Lung ROI",
        "Grad-CAM Heatmap",
        "Activation Distribution",
    ]

    fig = plt.figure(figsize=(14, 18), facecolor="#050a12")
    fig.suptitle(
        "Explainable AI — Normal vs Pneumonia Grad-CAM\n"
        "NORMAL: Cool colors, Low activation  |  "
        "PNEUMONIA: Hot colors, High focal activation",
        fontsize=13, color="#00d4ff",
        fontweight="bold", y=0.995)
    gs = gridspec.GridSpec(3, 2, figure=fig,
                            hspace=0.08, wspace=0.06)

    for col, tag in enumerate(["normal", "pneumonia"]):
        d  = results[tag]
        bc = col_c[tag]

        for row in range(3):
            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor("#0b1220")

            if row == 0:
                combined = np.concatenate(
                    [d["disp"], d["roi"]], axis=1)
                ax.imshow(combined)
                ax.axvline(IMG_SIZE, color=bc,
                            lw=1.5, ls="--", alpha=0.6)
                ax.text(IMG_SIZE // 2, 8, "Original",
                        color="white", ha="center",
                        fontsize=8, va="top",
                        bbox=dict(facecolor="#050a12",
                                   alpha=0.6, pad=2))
                ax.text(IMG_SIZE * 3 // 2, 8, "Lung ROI",
                        color="#00d4ff", ha="center",
                        fontsize=8, va="top",
                        bbox=dict(facecolor="#050a12",
                                   alpha=0.6, pad=2))
                ax.set_title(
                    f"{'NORMAL' if tag == 'normal' else 'PNEUMONIA'}\n"
                    f"P(Pneumonia) = {d['prob'] * 100:.1f}%",
                    color=bc, fontsize=12,
                    fontweight="bold", pad=6)
                ax.axis("off")

            elif row == 1:
                ax.imshow(d["cam"])
                msg = ("Low uniform activation\nNo hotspots"
                       if tag == "normal"
                       else "High focal activation\n"
                            "Hotspots in lower lobes")
                ax.text(5, IMG_SIZE - 5, msg,
                        color=bc, fontsize=8, va="bottom",
                        bbox=dict(facecolor="#050a12",
                                   alpha=0.8, pad=3))
                ax.axis("off")

            elif row == 2:
                ax.hist(d["h1"].ravel(), bins=60,
                        color=bc, alpha=0.85, density=True)
                ax.axvline(
                    d["h1"].mean(), color="white",
                    lw=2, ls="--",
                    label=f"Mean={d['h1'].mean():.3f}")
                ax.set_xlabel("Activation [0-1]",
                              fontsize=8, color="#c0d0e0")
                ax.set_ylabel("Density", fontsize=8)
                ax.legend(fontsize=8,
                           facecolor="#0b1220",
                           edgecolor="#1a2a40")
                ax.grid(True, alpha=0.2)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

            for sp in ax.spines.values():
                sp.set_edgecolor(bc)
                sp.set_linewidth(2)

            if col == 0 and row < 2:
                ax.set_ylabel(rows[row], color="#c0d0e0",
                              fontsize=8, rotation=90,
                              labelpad=4)
                ax.yaxis.set_label_coords(-0.04, 0.5)

    handles = [
        mpatches.Patch(
            color="#00ff9d",
            label="NORMAL — cool colors, low uniform activation"),
        mpatches.Patch(
            color="#ff3d5a",
            label="PNEUMONIA — hot colors, high focal activation"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               fontsize=9, facecolor="#0b1220",
               edgecolor="#1a2a40",
               bbox_to_anchor=(0.5, 0.002))

    plt.tight_layout(rect=[0, 0.04, 1, 0.99])
    path = str(OUT_DIR / "1_gradcam_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

    n, p = results["normal"], results["pneumonia"]
    print(f"\n  Activation Statistics:")
    print(f"    NORMAL    mean={n['h1'].mean():.4f}  "
          f"max={n['h1'].max():.4f}")
    print(f"    PNEUMONIA mean={p['h1'].mean():.4f}  "
          f"max={p['h1'].max():.4f}")
    print(f"  Saved -> {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# XAI 2 — SHAP ANALYSIS ON VITAL SIGNS
# ══════════════════════════════════════════════════════════════════════════════
"""
HOW SHAP WORKS:
  Game theory: each vital sign = player, prediction = prize
  SHAP value = fair contribution of each feature to prediction
  SHAP > 0 → pushes toward PNEUMONIA
  SHAP < 0 → pushes toward NORMAL (protective)

4 Panels:
  A - Mean |SHAP|: Which vitals matter most overall?
  B - Beeswarm:   How does each value affect risk?
  C - Waterfall:  Why did THIS patient get this score?
  D - Dependence: How do two features interact?
"""


def xai_shap_analysis(vitals_csv):
    if not SHAP_OK:
        print("  [skip] SHAP not available. Run: pip install shap")
        return None

    print("\n  Running SHAP Analysis...")

    # Load and prepare data (same as vital.py)
    data = pd.read_csv(vitals_csv)
    data = _fix_cols(data)
    data = data[VITAL_FEATURES + ["Risk Category"]].copy()

    le = LabelEncoder()
    ge = LabelEncoder()
    data["Risk Category"] = le.fit_transform(data["Risk Category"])
    data["Gender"]        = ge.fit_transform(data["Gender"])

    X = data.drop("Risk Category", axis=1)
    y = data["Risk Category"]

    sc       = MinMaxScaler()
    X_scaled = sc.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2,
        random_state=42, stratify=y)

    # Train XGBoost (same parameters as vital.py)
    model = XGBClassifier(
        n_estimators=400, max_depth=7,
        learning_rate=0.03, subsample=0.9,
        colsample_bytree=0.9, gamma=0.1,
        reg_alpha=0.1, reg_lambda=1,
        objective="binary:logistic",
        eval_metric="logloss", random_state=42)
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train),
                         (X_test, y_test)],
              verbose=False)

    feat_names = list(X.columns)

    # Compute SHAP values
    print("  Computing SHAP TreeExplainer values...")
    explainer = shap.TreeExplainer(model)
    X_ex      = X_test[:200]
    sv_raw    = explainer.shap_values(X_ex)
    sv        = sv_raw[1] if isinstance(sv_raw, list) else sv_raw
    preds     = model.predict_proba(X_ex)[:, 1]

    _style()
    fig = plt.figure(figsize=(22, 16), facecolor="#050a12")
    fig.suptitle(
        "SHAP Analysis — XGBoost Vital Signs\n"
        "Which vital signs matter most for Pneumonia prediction?",
        fontsize=15, color="#00d4ff",
        fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig,
                            hspace=0.45, wspace=0.38)

    mean_sv = np.abs(sv).mean(axis=0)
    s_idx   = np.argsort(mean_sv)[::-1]

    # Panel A — Mean |SHAP| bar chart
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor("#0b1220")
    bc_a = ["#ff3d5a" if mean_sv[i] > mean_sv.mean()
            else "#00d4ff" for i in s_idx]
    bars = ax_a.bar(
        [feat_names[i] for i in s_idx],
        mean_sv[s_idx],
        color=bc_a, edgecolor="#1a2a40")
    for bar, val in zip(bars, mean_sv[s_idx]):
        ax_a.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + mean_sv.max() * 0.015,
            f"{val:.3f}", ha="center",
            fontsize=8, color="white")
    ax_a.set_title(
        "A. Mean |SHAP| — Feature Importance",
        color="#00d4ff", fontsize=11, pad=10)
    ax_a.set_ylabel("|SHAP value|")
    ax_a.tick_params(axis="x", rotation=35, labelsize=8)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.grid(True, alpha=0.2, axis="y")

    # Panel B — Beeswarm plot
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor("#0b1220")
    for idx in range(len(feat_names)):
        jitter = np.random.normal(idx, 0.08, size=len(sv))
        sc2 = ax_b.scatter(
            sv[:, idx], jitter,
            c=X_ex[:, idx], cmap="RdYlBu_r",
            alpha=0.65, s=18, edgecolors="none")
    ax_b.set_yticks(range(len(feat_names)))
    ax_b.set_yticklabels(feat_names, fontsize=9)
    ax_b.axvline(0, color="#4a6080", lw=2, ls="--")
    ax_b.set_title(
        "B. Beeswarm — Each dot = one patient",
        color="#00d4ff", fontsize=11, pad=10)
    ax_b.set_xlabel(
        "← Normal Risk    SHAP value    Pneumonia Risk →",
        fontsize=8)
    plt.colorbar(sc2, ax=ax_b,
                  label="Feature value (red=high, blue=low)",
                  shrink=0.75)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.grid(True, alpha=0.12, axis="x")

    # Panel C — Waterfall for highest risk patient
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_facecolor("#0b1220")
    top_i  = int(np.argmax(preds))
    sv_p   = sv[top_i]
    sfi    = np.argsort(np.abs(sv_p))[::-1]
    sv_s   = sv_p[sfi]
    fn_s   = [feat_names[i] for i in sfi]
    bc_c   = ["#ff3d5a" if v > 0 else "#00d4ff" for v in sv_s]
    bars_c = ax_c.barh(fn_s, sv_s,
                        color=bc_c, edgecolor="#1a2a40")
    ax_c.axvline(0, color="#4a6080", lw=2, ls="--")
    for bar, val in zip(bars_c, sv_s):
        ax_c.text(
            val + (0.002 if val >= 0 else -0.002),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}", va="center", fontsize=8,
            ha="left" if val >= 0 else "right",
            color="white")
    ax_c.set_title(
        f"C. Waterfall — Highest Risk Patient "
        f"(Risk={preds[top_i] * 100:.1f}%)",
        color="#00d4ff", fontsize=11, pad=10)
    ax_c.set_xlabel("SHAP value contribution")
    ax_c.tick_params(labelsize=9)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    rp = mpatches.Patch(color="#ff3d5a",
                         label="Increases Pneumonia Risk")
    bp = mpatches.Patch(color="#00d4ff",
                         label="Reduces Risk (Protective)")
    ax_c.legend(handles=[rp, bp], fontsize=8,
                 facecolor="#0b1220", edgecolor="#1a2a40")

    # Panel D — Dependence plot (top 2 features)
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_facecolor("#0b1220")
    top2 = np.argsort(mean_sv)[-2:]
    f1i, f2i = int(top2[0]), int(top2[1])
    sc3 = ax_d.scatter(
        X_ex[:, f1i], sv[:, f1i],
        c=X_ex[:, f2i], cmap="plasma",
        s=25, alpha=0.75, edgecolors="none")
    ax_d.axhline(0, color="#4a6080", lw=1.5, ls="--")
    z  = np.polyfit(X_ex[:, f1i], sv[:, f1i], 1)
    xr = np.linspace(X_ex[:, f1i].min(),
                      X_ex[:, f1i].max(), 100)
    ax_d.plot(xr, np.poly1d(z)(xr), color="#ffd166",
              lw=2, ls="--", label="Trend")
    ax_d.set_xlabel(f"{feat_names[f1i]} (scaled)",
                     fontsize=9)
    ax_d.set_ylabel(f"SHAP({feat_names[f1i]})",
                     fontsize=9)
    ax_d.set_title(
        f"D. Dependence: {feat_names[f1i]}\n"
        f"colour = {feat_names[f2i]}",
        color="#00d4ff", fontsize=11, pad=10)
    plt.colorbar(sc3, ax=ax_d,
                  label=feat_names[f2i], shrink=0.75)
    ax_d.legend(fontsize=8, facecolor="#0b1220",
                 edgecolor="#1a2a40")
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)
    ax_d.grid(True, alpha=0.15)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = str(OUT_DIR / "2_shap_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"\n  Feature Importance Ranking:")
    for rank, i in enumerate(s_idx):
        bar = "█" * int(mean_sv[i] / mean_sv[s_idx[0]] * 20)
        print(f"    {rank+1:2}. {feat_names[i]:<30} "
              f"{mean_sv[i]:.4f}  {bar}")
    print(f"  Saved -> {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# XAI 3 — FUSION XAI DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
"""
FUSION MODEL EXPLANATION:
  Branch 1 (CNN):     X-ray  → ResNet50 → 128 features + probability
  Branch 2 (XGBoost): Vitals → 400 trees → risk probability
  Fusion Head:        Concatenate both → Dense → Final prediction

  High CNN + High XGBoost → Very confident PNEUMONIA
  Low  CNN + Low  XGBoost → Very confident NORMAL
  Mixed signals           → Moderate confidence
"""


def xai_fusion_dashboard(image_path, vital_values):
    print("\n  Running Fusion XAI Dashboard...")

    # Check all model files exist
    for p in [MODEL_DIR / "cnn_best.h5",
               MODEL_DIR / "fusion_head_best.h5",
               MODEL_DIR / "advanced_scaler.pkl"]:
        if not p.exists():
            print(f"  [skip] Model not found: {p}")
            return None

    # Load all models
    cnn_model    = load_model(str(MODEL_DIR / "cnn_best.h5"))
    fusion_model = load_model(
        str(MODEL_DIR / "fusion_head_best.h5"))
    scaler       = joblib.load(
        MODEL_DIR / "advanced_scaler.pkl")
    xgb_model    = joblib.load(
        MODEL_DIR / "advanced_risk_model.pkl")
    cfg          = json.load(
        open(MODEL_DIR / "fusion_config.json"))
    feat_names   = cfg["feature_names"]

    # Encode gender
    vit = vital_values.copy()
    genc_path = MODEL_DIR / "advanced_gender_encoder.pkl"
    if genc_path.exists():
        genc = joblib.load(genc_path)
        g    = vit.get("Gender", "Male")
        try:
            vit["Gender"] = int(genc.transform([g])[0])
        except Exception:
            vit["Gender"] = 0

    # Build vital feature vector
    v_raw = np.array(
        [float(vit.get(f, 0.0)) for f in feat_names],
        dtype=np.float32).reshape(1, -1)
    v_sc  = scaler.transform(v_raw).astype(np.float32)
    xgb_p = xgb_model.predict_proba(v_sc)

    # CNN features and prediction
    arr, disp = load_image(image_path)
    from tensorflow.keras.models import Model as KModel
    fm = KModel(
        cnn_model.input,
        [cnn_model.get_layer("dense_128").output,
         cnn_model.output])
    cnn_feat, cnn_prob_arr = fm.predict(arr, verbose=0)
    cnn_prob = float(cnn_prob_arr[0][0])
    img_feat = np.column_stack([cnn_feat, [[cnn_prob]]])

    # Fusion prediction
    vit_feat = np.concatenate(
        [xgb_p, v_sc], axis=1).astype(np.float32)
    fp = float(
        fusion_model.predict(
            [img_feat, vit_feat], verbose=0)[0][0])

    label = CLASS_NAMES[int(fp >= 0.5)]
    conf  = fp if fp >= 0.5 else 1 - fp

    print(f"\n  CNN Branch Score   : {cnn_prob*100:.2f}%")
    print(f"  XGBoost Risk Score : {xgb_p[0,1]*100:.2f}%")
    print(f"  FUSION PREDICTION  : {label}")
    print(f"  CONFIDENCE         : {conf*100:.2f}%")

    # Grad-CAM
    print("    Computing Grad-CAM...")
    ci  = int(cnn_prob >= 0.5)
    h1  = compute_gradcam(cnn_model, arr, ci)
    ov1 = overlay_heatmap(disp, h1)

    bc = "#ff3d5a" if label == "PNEUMONIA" else "#00ff9d"

    _style()
    fig = plt.figure(figsize=(24, 10), facecolor="#050a12")
    fig.suptitle(
        f"Fusion XAI Dashboard  |  Prediction: {label}"
        f"  |  Confidence: {conf*100:.1f}%\n"
        f"CNN: {cnn_prob*100:.1f}%    "
        f"XGBoost: {xgb_p[0,1]*100:.1f}%    "
        f"Fusion: {fp*100:.1f}%",
        fontsize=13, color="#00d4ff",
        fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(2, 4, figure=fig,
                            hspace=0.38, wspace=0.15)

    # Row 1: Image panels
    for col, (img, title) in enumerate([
        (disp,             "Original X-ray"),
        (lung_mask(disp),  "Lung ROI Mask"),
        (ov1,              "Grad-CAM Heatmap"),
    ]):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img)
        ax.set_title(title, color="#00d4ff",
                      fontsize=9, pad=5)
        ax.axis("off")
        for sp in ax.spines.values():
            sp.set_edgecolor(bc)
            sp.set_linewidth(2)

    # Branch scores panel
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.set_facecolor("#0b1220")
    for i, (lbl, val, col) in enumerate([
        ("CNN\nBranch",    cnn_prob,    "#00d4ff"),
        ("XGBoost\nRisk",  xgb_p[0, 1], "#ff6b35"),
        ("Fusion\nResult", fp,           "#ffd166"),
    ]):
        ax4.barh([lbl], [val], color=col,
                  edgecolor="#1a2a40", height=0.5)
        ax4.text(min(val + 0.02, 0.82), i,
                  f"{val*100:.1f}%",
                  va="center", fontsize=11,
                  color="white", fontweight="bold")
    ax4.set_xlim(0, 1)
    ax4.set_title("Branch Scores",
                   color="#00d4ff", fontsize=9, pad=5)
    ax4.tick_params(labelsize=8)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    # Vital signs bar
    ax5 = fig.add_subplot(gs[1, :2])
    ax5.set_facecolor("#0b1220")
    vnames = [k for k in vital_values
               if isinstance(vital_values[k], (int, float))]
    vvals  = [float(vital_values[k]) for k in vnames]
    vc     = ["#ff3d5a" if v > np.mean(vvals) else "#00d4ff"
               for v in vvals]
    brs    = ax5.barh(vnames, vvals,
                       color=vc, edgecolor="#1a2a40")
    for bar, val in zip(brs, vvals):
        ax5.text(bar.get_width() * 0.02,
                  bar.get_y() + bar.get_height() / 2,
                  f"{val}", va="center",
                  fontsize=8, color="white")
    ax5.set_title("Vital Signs Input",
                   color="#00d4ff", fontsize=9, pad=5)
    ax5.tick_params(labelsize=8)
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)

    # Activation histogram
    ax6 = fig.add_subplot(gs[1, 2:])
    ax6.set_facecolor("#0b1220")
    ax6.hist(h1.ravel(), bins=60, color="#ff3d5a",
              alpha=0.80, label="Grad-CAM", density=True)
    ax6.axvline(h1.mean(), color="white", lw=2, ls="--",
                 label=f"Mean={h1.mean():.3f}")
    ax6.set_title("Activation Distribution",
                   color="#00d4ff", fontsize=9, pad=5)
    ax6.set_xlabel("Activation [0-1]", fontsize=8)
    ax6.set_ylabel("Density", fontsize=8)
    ax6.legend(fontsize=8, facecolor="#0b1220",
                edgecolor="#1a2a40")
    ax6.grid(True, alpha=0.2)
    ax6.spines["top"].set_visible(False)
    ax6.spines["right"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = str(OUT_DIR / "3_fusion_xai_dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved -> {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _fix_cols(df):
    lc = {c.lower().strip(): c for c in df.columns}
    mp = {
        "Heart Rate":
            ["heart rate", "heartrate", "hr"],
        "Respiratory Rate":
            ["respiratory rate", "resp rate", "rr"],
        "Body Temperature":
            ["body temperature", "temperature", "temp"],
        "Oxygen Saturation":
            ["oxygen saturation", "spo2"],
        "Systolic Blood Pressure":
            ["systolic blood pressure", "systolic bp", "sbp"],
        "Diastolic Blood Pressure":
            ["diastolic blood pressure", "diastolic bp", "dbp"],
        "Age":
            ["age"],
        "Gender":
            ["gender", "sex"],
        "Derived_HRV":
            ["derived_hrv", "hrv"],
        "Derived_MAP":
            ["derived_map", "map"],
        "Derived_BMI":
            ["derived_bmi", "bmi"],
        "Risk Category":
            ["risk category", "risk_category",
             "label", "diagnosis", "target"],
    }
    rename = {}
    for std, aliases in mp.items():
        for a in aliases:
            if a in lc and lc[a] != std:
                rename[lc[a]] = std
                break
    df = df.rename(columns=rename)

    if "Derived_HRV" not in df.columns:
        df["Derived_HRV"] = (
            1000.0 / (df["Heart Rate"] + 1e-6)
            if "Heart Rate" in df.columns else 30.0)
    if "Derived_MAP" not in df.columns:
        if ({"Systolic Blood Pressure",
              "Diastolic Blood Pressure"}
                <= set(df.columns)):
            df["Derived_MAP"] = (
                df["Diastolic Blood Pressure"]
                + (df["Systolic Blood Pressure"]
                   - df["Diastolic Blood Pressure"]) / 3.0)
        else:
            df["Derived_MAP"] = 93.0
    if "Derived_BMI" not in df.columns:
        df["Derived_BMI"] = 22.5
    if "Risk Category" not in df.columns:
        df["Risk Category"] = 0
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — runs everything automatically
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "╔" + "═" * 62 + "╗")
    print("║  Explainable AI — Grad-CAM + SHAP + Fusion XAI Dashboard  ║")
    print("╚" + "═" * 62 + "╝\n")

    # Step 1: Find test images
    print("  Step 1: Finding test images...")
    try:
        normal_img, pneumo_img = find_test_images()
        print(f"  Normal    : {normal_img}")
        print(f"  Pneumonia : {pneumo_img}")
    except FileNotFoundError as e:
        print(f"  [error] {e}")
        sys.exit(1)

    # Step 2: Load CNN model
    print("\n  Step 2: Loading CNN model...")
    if not (MODEL_DIR / "cnn_best.h5").exists():
        print(f"  [error] cnn_best.h5 not found in {MODEL_DIR}")
        print("  Run: python fusion.py --mode train ...")
        sys.exit(1)
    cnn_model = load_model(str(MODEL_DIR / "cnn_best.h5"))
    print("  CNN model loaded successfully")

    # Step 3: Grad-CAM comparison
    print("\n  Step 3: Normal vs Pneumonia Grad-CAM...")
    xai_gradcam_comparison(cnn_model, normal_img, pneumo_img)

    # Step 4: SHAP analysis
    print("\n  Step 4: SHAP vital signs analysis...")
    if os.path.isfile(VITALS_CSV):
        xai_shap_analysis(VITALS_CSV)
    else:
        print(f"  [skip] {VITALS_CSV} not found")

    # Step 5: Fusion dashboard
    print("\n  Step 5: Fusion XAI dashboard...")
    xai_fusion_dashboard(pneumo_img, SAMPLE_VITALS)

    # Done
    print("\n" + "=" * 60)
    print("  ALL XAI OUTPUTS SAVED")
    print("=" * 60)
    print(f"  Folder: {OUT_DIR.resolve()}")
    for f in sorted(OUT_DIR.glob("*.png")):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:<45} {size_kb:.1f} KB")
    print("=" * 60)


if __name__ == "__main__":
    main()
