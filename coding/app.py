#!/usr/bin/env python3
# AI Multimodal Pneumonia Clinical Decision System
# Run: streamlit run app.py

import os, sys, json, sqlite3, hashlib, warnings, requests
from io import BytesIO
from datetime import datetime
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.resnet50 import preprocess_input

try:
    import shap; SHAP_OK = True
except: SHAP_OK = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.units import inch
    REPORTLAB_OK = True
except: REPORTLAB_OK = False

# ── OUTPUT FOLDER PATHS ──────────────────────────────────────────────────────
# All folders created by your training files
FUSION_MODEL_DIR  = Path("pneumofusion_outputs/models")   # fusion.py output
FUSION_PLOT_DIR   = Path("pneumofusion_outputs/plots")    # fusion.py plots
XAI_OUTPUT_DIR    = Path("xai_outputs")                   # xoi_fixed.py output
PNEUMO_MODEL_DIR  = Path("outputs/model")                 # pneumonia.py output
PNEUMO_PLOT_DIR   = Path("outputs/plots")                 # pneumonia.py plots
DB_PATH           = "pneumonia_system.db"
IMG_SIZE          = 224
CLASS_NAMES       = ["NORMAL", "PNEUMONIA"]

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Pneumonia Clinical System", page_icon="🫁",
    layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp{background:#0a0f1e}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1b2a,#1a2744);border-right:1px solid #00d4ff33}
.mtitle{font-family:monospace;font-size:1.7rem;font-weight:bold;color:#00d4ff;text-align:center;padding:1rem 0 .5rem;border-bottom:2px solid #00d4ff44;margin-bottom:1.5rem}
.shdr{font-family:monospace;color:#00d4ff;font-size:1.1rem;font-weight:bold;border-left:4px solid #00d4ff;padding-left:.8rem;margin:1rem 0 .8rem}
.pred-p{background:#ff3d5a22;border:2px solid #ff3d5a;border-radius:12px;padding:1.5rem;text-align:center}
.pred-n{background:#00ff9d22;border:2px solid #00ff9d;border-radius:12px;padding:1.5rem;text-align:center}
.risk-h{background:#ff3d5a22;border:2px solid #ff3d5a;border-radius:8px;padding:.8rem;color:#ff3d5a;font-weight:bold;text-align:center}
.risk-m{background:#ffd16622;border:2px solid #ffd166;border-radius:8px;padding:.8rem;color:#ffd166;font-weight:bold;text-align:center}
.risk-l{background:#00ff9d22;border:2px solid #00ff9d;border-radius:8px;padding:.8rem;color:#00ff9d;font-weight:bold;text-align:center}
.chat-u{background:#1a2744;border-radius:12px 12px 0 12px;padding:.8rem 1rem;margin:.4rem 0;color:#c0d0e0}
.chat-a{background:#0d1b2a;border-radius:12px 12px 12px 0;padding:.8rem 1rem;margin:.4rem 0;color:#00ff9d;border:1px solid #00ff9d33}
.ocard{background:#0d1b2a;border:1px solid #00d4ff33;border-radius:10px;padding:1rem;margin:.4rem 0}
</style>""", unsafe_allow_html=True)

# ── DATABASE ─────────────────────────────────────────────────────────────────
def init_db():
    c = sqlite3.connect(DB_PATH)
    c.execute("""CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL, password TEXT NOT NULL,
        email TEXT, created TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS reports(
        id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT,
        patient_name TEXT, vitals TEXT, prediction TEXT,
        confidence REAL, risk TEXT, timestamp TEXT)""")
    c.commit(); c.close()

def hash_pw(p): return hashlib.sha256(p.encode()).hexdigest()

def register_user(u, p, e=""):
    try:
        c = sqlite3.connect(DB_PATH)
        c.execute("INSERT INTO users VALUES(NULL,?,?,?,?)",
                   (u, hash_pw(p), e, datetime.now().isoformat()))
        c.commit(); c.close()
        return True, "Registration successful!"
    except sqlite3.IntegrityError:
        return False, "Username already exists."

def login_user(u, p):
    c = sqlite3.connect(DB_PATH)
    r = c.execute("SELECT id FROM users WHERE username=? AND password=?",
                   (u, hash_pw(p))).fetchone()
    c.close(); return r is not None

def save_report(user, name, vitals, pred, conf, risk):
    c = sqlite3.connect(DB_PATH)
    c.execute("INSERT INTO reports VALUES(NULL,?,?,?,?,?,?,?)",
               (user, name, json.dumps(vitals), pred, conf, risk,
                datetime.now().isoformat()))
    c.commit(); c.close()

def get_reports(user):
    c = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM reports WHERE username=? ORDER BY timestamp DESC",
        c, params=(user,))
    c.close(); return df

# ── LOAD ALL MODELS ───────────────────────────────────────────────────────────
@st.cache_resource
def load_all_models():
    """Load models from all output folders"""
    m = {}
    # From fusion.py output (pneumofusion_outputs/models/)
    if (FUSION_MODEL_DIR/"cnn_best.h5").exists():
        m["cnn"] = load_model(str(FUSION_MODEL_DIR/"cnn_best.h5"))
    if (FUSION_MODEL_DIR/"fusion_head_best.h5").exists():
        m["fusion"] = load_model(str(FUSION_MODEL_DIR/"fusion_head_best.h5"))
    if (FUSION_MODEL_DIR/"advanced_risk_model.pkl").exists():
        m["xgb"] = joblib.load(FUSION_MODEL_DIR/"advanced_risk_model.pkl")
    if (FUSION_MODEL_DIR/"advanced_scaler.pkl").exists():
        m["scaler"] = joblib.load(FUSION_MODEL_DIR/"advanced_scaler.pkl")
    if (FUSION_MODEL_DIR/"advanced_gender_encoder.pkl").exists():
        m["genc"] = joblib.load(FUSION_MODEL_DIR/"advanced_gender_encoder.pkl")
    if (FUSION_MODEL_DIR/"fusion_config.json").exists():
        m["config"] = json.load(open(FUSION_MODEL_DIR/"fusion_config.json"))
    # From pneumonia.py output (outputs/model/)
    for fn in ["pneumonet_best.h5","pneumonia_model.keras","pneumonia_model.h5"]:
        p = PNEUMO_MODEL_DIR/fn
        if p.exists() and "cnn" not in m:
            m["cnn"] = load_model(str(p)); break
    # Load saved metrics
    for mf in [(FUSION_MODEL_DIR.parent/"results/fusion_metrics.json"),
               (PNEUMO_MODEL_DIR.parent/"results/metrics.json")]:
        if mf.exists():
            m[mf.stem] = json.load(open(mf))
    return m

# ── IMAGE PROCESSING ──────────────────────────────────────────────────────────
def load_preprocess(img_rgb):
    disp = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    arr  = preprocess_input(disp.astype(np.float32))
    return np.expand_dims(arr, 0), disp

def lung_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    inv  = cv2.bitwise_not(gray)
    _, t = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    t    = cv2.morphologyEx(t, cv2.MORPH_CLOSE, k, iterations=2)
    t    = cv2.morphologyEx(t, cv2.MORPH_OPEN, k, iterations=1)
    _, lbl, stats, _ = cv2.connectedComponentsWithStats(t)
    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) >= 2:
        top2 = np.argsort(areas)[-2:]+1
        mask = np.zeros_like(t)
        for i in top2: mask[lbl==i] = 255
    else: mask = t
    mask  = cv2.dilate(mask, k, iterations=1)
    mask3 = np.stack([mask]*3, axis=-1)
    out   = img.copy(); out[mask3==0] = 0
    return out

# ── GRAD-CAM (tf.Variable approach) ──────────────────────────────────────────
def compute_gradcam(cnn_model, img_array, class_idx=1):
    """
    GRAD-CAM EXPLANATION:
    NORMAL lungs   -> low uniform activation -> cool blue colors
    PNEUMONIA lungs -> high focal activation -> hot red colors in lower lobes
    """
    try:
        resnet = cnn_model.get_layer("resnet50")
        img_t  = tf.Variable(img_array, dtype=tf.float32)
        with tf.GradientTape() as tape:
            conv_out = resnet(img_t, training=False)
            x = conv_out; after = False
            for layer in cnn_model.layers:
                if layer.name == "resnet50": after = True; continue
                if after: x = layer(x, training=False)
            score = x[:,0] if class_idx==1 else (1-x[:,0])
        grads = tape.gradient(score, conv_out)
        if grads is None:
            return np.zeros((IMG_SIZE,IMG_SIZE), dtype=np.float32)
        pooled = tf.reduce_mean(grads, axis=(0,1,2)).numpy()
        cn = conv_out.numpy()[0]
        h  = np.zeros(cn.shape[:2], dtype=np.float32)
        for i, w in enumerate(pooled): h += w * cn[:,:,i]
        h = np.maximum(h, 0)
        if h.max() > 0: h /= h.max()
        return cv2.resize(h, (IMG_SIZE,IMG_SIZE)).astype(np.float32)
    except Exception as e:
        st.warning(f"Grad-CAM: {e}")
        return np.zeros((IMG_SIZE,IMG_SIZE), dtype=np.float32)

def overlay_cam(img, h, alpha=0.45, cmap=cv2.COLORMAP_JET):
    u8   = np.uint8(255*h)
    cm   = cv2.cvtColor(cv2.applyColorMap(u8, cmap), cv2.COLOR_BGR2RGB)
    return ((1-alpha)*img + alpha*cm).clip(0,255).astype(np.uint8)

def gradcam_fig(orig, roi, h, ov, label, prob):
    fig, axes = plt.subplots(1, 4, figsize=(18,5))
    fig.patch.set_facecolor("#050a12")
    bc = "#ff3d5a" if label=="PNEUMONIA" else "#00ff9d"
    for ax, img, t in zip(axes, [orig,roi,h,ov],
                           ["Original X-ray","Lung ROI Mask","Grad-CAM Heatmap","Grad-CAM Overlay"]):
        ax.set_facecolor("#0b1220")
        ax.imshow(img, cmap="jet" if t=="Grad-CAM Heatmap" else None,
                   vmin=0 if t=="Grad-CAM Heatmap" else None,
                   vmax=1 if t=="Grad-CAM Heatmap" else None)
        ax.set_title(t, color="#00d4ff", fontsize=10, pad=6); ax.axis("off")
        for sp in ax.spines.values(): sp.set_edgecolor(bc); sp.set_linewidth(2)
    conf = (prob if prob>=0.5 else 1-prob)*100
    fig.suptitle(f"Grad-CAM Explainability  |  {label}  |  {conf:.1f}% confidence",
                 fontsize=12, color="#00d4ff", fontweight="bold", y=1.02)
    plt.tight_layout(); return fig

# ── VITALS & RISK ─────────────────────────────────────────────────────────────
def calc_derived(v):
    h   = v.get("Height",170)/100
    w   = v.get("Weight",70)
    hr  = v.get("Heart Rate",80)
    sbp = v.get("Systolic Blood Pressure",120)
    dbp = v.get("Diastolic Blood Pressure",80)
    bmi = round(w/(h**2),2) if h>0 else 0
    mapp= round((2*dbp+sbp)/3,1)
    return {"BMI":bmi,"Pulse Pressure":sbp-dbp,"MAP":mapp,
            "Derived_HRV":round(1000/hr,2) if hr>0 else 0,
            "Derived_MAP":mapp,"Derived_BMI":bmi}

def classify_risk(v, pred, conf):
    score = 0; reasons = []
    spo2 = v.get("Oxygen Saturation",98)
    rr   = v.get("Respiratory Rate",16)
    temp = v.get("Body Temperature",37)
    hr   = v.get("Heart Rate",80)
    if spo2 < 92:   score+=3; reasons.append(f"🔴 Critical SpO2: {spo2}%")
    elif spo2 < 95: score+=2; reasons.append(f"🟡 Low SpO2: {spo2}%")
    if rr > 25:     score+=2; reasons.append(f"🔴 High Resp Rate: {rr}/min")
    elif rr > 20:   score+=1; reasons.append(f"🟡 Elevated RR: {rr}/min")
    if temp > 38.5: score+=2; reasons.append(f"🔴 High Fever: {temp}°C")
    elif temp > 37.5: score+=1; reasons.append(f"🟡 Mild Fever: {temp}°C")
    if hr > 100:    score+=1; reasons.append(f"🟡 Tachycardia: {hr} bpm")
    if pred == "PNEUMONIA":
        score += 2 + int(conf*2)
        reasons.append(f"🔴 AI: PNEUMONIA ({conf*100:.0f}% confidence)")
    if score >= 6:   return "🔴 HIGH RISK", reasons
    elif score >= 3: return "🟡 MODERATE RISK", reasons
    else:            return "🟢 LOW RISK", reasons

# ── PREDICTIONS ───────────────────────────────────────────────────────────────
def pred_cnn(cnn, arr):
    p = float(cnn.predict(arr, verbose=0)[0][0])
    l = CLASS_NAMES[int(p>=0.5)]
    return l, (p if p>=0.5 else 1-p), p

def pred_fusion(m, arr, vitals):
    try:
        feat = m["config"]["feature_names"]
        vit  = vitals.copy()
        if "genc" in m:
            g = vit.get("Gender","Male")
            try: vit["Gender"] = int(m["genc"].transform([g])[0])
            except: vit["Gender"] = 0
        vr = np.array([float(vit.get(f,0)) for f in feat], dtype=np.float32).reshape(1,-1)
        vs = m["scaler"].transform(vr).astype(np.float32)
        xp = m["xgb"].predict_proba(vs)
        fm = Model(m["cnn"].input, [m["cnn"].get_layer("dense_128").output, m["cnn"].output])
        cf, cp = fm.predict(arr, verbose=0)
        cpp = float(cp[0][0])
        imf = np.column_stack([cf, [[cpp]]])
        vf  = np.concatenate([xp, vs], axis=1).astype(np.float32)
        fp  = float(m["fusion"].predict([imf,vf], verbose=0)[0][0])
        l   = CLASS_NAMES[int(fp>=0.5)]
        return l, (fp if fp>=0.5 else 1-fp), fp, cpp, float(xp[0,1])
    except Exception as e:
        st.warning(f"Fusion fallback to CNN: {e}")
        l,c,p = pred_cnn(m["cnn"],arr)
        return l, c, p, p, None

# ── OLLAMA LLM ────────────────────────────────────────────────────────────────
def ask_llm(q, ctx=""):
    sys_p = "You are a helpful medical AI assistant explaining pneumonia diagnosis in simple language."
    try:
        r = requests.post("http://localhost:11434/api/generate",
            json={"model":"llama2:latest","prompt":f"{sys_p}\n\n{ctx}\nUser:{q}","stream":False},
            timeout=300)
        if r.status_code == 200: return r.json().get("response","No response.")
        return f"Ollama error {r.status_code}"
    except requests.ConnectionError:
        return ("❌ Ollama not running.\n\n"
                "To enable:\n1. Download: https://ollama.ai\n"
                "2. Run: ollama pull llama2\n3. Run: ollama serve")
    except Exception as e:
        return f"Error: {e}"

# ── PDF REPORT ────────────────────────────────────────────────────────────────
def gen_pdf(pinfo, vitals, derived, pred, conf, risk, reasons):
    if not REPORTLAB_OK: return None
    buf  = BytesIO()
    doc  = SimpleDocTemplate(buf, pagesize=A4, topMargin=.5*inch, bottomMargin=.5*inch)
    stys = getSampleStyleSheet()
    T = ParagraphStyle("T",parent=stys["Title"],fontSize=18,alignment=1,
                        textColor=colors.HexColor("#1a3a6b"),spaceAfter=4)
    S = ParagraphStyle("S",parent=stys["Normal"],fontSize=9,alignment=1,
                        textColor=colors.grey,spaceAfter=10)
    H = ParagraphStyle("H",parent=stys["Heading2"],fontSize=13,
                        textColor=colors.HexColor("#1a3a6b"),spaceAfter=4)
    N = ParagraphStyle("N",parent=stys["Normal"],fontSize=10,spaceAfter=3)

    def tbl(data, widths):
        t = Table(data, colWidths=widths)
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1a3a6b")),
            ("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("FONTSIZE",(0,0),(-1,-1),10),
            ("GRID",(0,0),(-1,-1),.5,colors.HexColor("#cccccc")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f2f2f2")]),
            ("PADDING",(0,0),(-1,-1),6)]))
        return t

    pc = colors.HexColor("#cc0000") if pred=="PNEUMONIA" else colors.HexColor("#006600")
    story = [
        Paragraph("AI Pneumonia Clinical Decision Report", T),
        Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",S),
        HRFlowable(width="100%",thickness=2,color=colors.HexColor("#1a3a6b")),
        Spacer(1,.15*inch), Paragraph("1. Patient Information",H),
        tbl([["Field","Value"],["Name",pinfo.get("name","N/A")],
              ["Age",str(pinfo.get("age","N/A"))],["Gender",pinfo.get("gender","N/A")],
              ["Date",datetime.now().strftime("%Y-%m-%d")]],[2.5*inch,4*inch]),
        Spacer(1,.15*inch), Paragraph("2. Clinical Vitals",H),
    ]
    nr = {"Heart Rate":"60–100 bpm","Respiratory Rate":"12–20 /min",
           "Body Temperature":"36.1–37.2 °C","Oxygen Saturation":"95–100 %",
           "Systolic Blood Pressure":"90–120 mmHg","Diastolic Blood Pressure":"60–80 mmHg"}
    vrows = [["Vital Sign","Value","Normal Range"]]
    for k,v in vitals.items():
        if k not in ["Weight","Height","Gender","Age","Derived_HRV","Derived_MAP","Derived_BMI"]:
            vrows.append([k,str(v),nr.get(k,"—")])
    story += [tbl(vrows,[3*inch,1.5*inch,2*inch]),Spacer(1,.15*inch),
              Paragraph("3. Derived Features",H)]
    drows = [["Feature","Value","Significance"]]
    sig = {"BMI":"Normal: 18.5–24.9","Pulse Pressure":"Normal: 40–60 mmHg","MAP":"Normal: 70–100 mmHg"}
    for k,v in derived.items():
        if k not in ["Derived_HRV","Derived_MAP","Derived_BMI"]:
            drows.append([k,str(v),sig.get(k,"—")])
    t4 = Table([["Parameter","Result"],["Prediction",pred],
                 ["Confidence",f"{conf*100:.1f}%"],
                 ["Risk",risk.replace("🔴","").replace("🟡","").replace("🟢","").strip()]],
                colWidths=[3*inch,3.5*inch])
    t4.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1a3a6b")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),11),
        ("TEXTCOLOR",(1,1),(1,1),pc),("FONTNAME",(1,1),(1,1),"Helvetica-Bold"),
        ("GRID",(0,0),(-1,-1),.5,colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f2f2f2")]),
        ("PADDING",(0,0),(-1,-1),8)]))
    story += [tbl(drows,[2.5*inch,1.5*inch,2.5*inch]),Spacer(1,.15*inch),
              Paragraph("4. AI Diagnosis",H),t4,Spacer(1,.15*inch)]
    if reasons:
        story.append(Paragraph("5. Risk Factors",H))
        for r in reasons:
            story.append(Paragraph(f"• {r.replace('🔴','').replace('🟡','').replace('🟢','').strip()}",N))
        story.append(Spacer(1,.1*inch))
    story.append(Paragraph("6. Medical Advice",H))
    adv = (["• Rest and adequate hydration","• Take prescribed antibiotics",
             "• Monitor oxygen saturation daily","• Avoid smoking",
             "• Emergency care if breathing worsens","• Follow-up X-ray in 4–6 weeks"]
           if pred=="PNEUMONIA"
           else ["• Maintain healthy lifestyle","• Regular medical checkups",
                 "• Keep vaccinations current","• Annual lung function tests"])
    for a in adv: story.append(Paragraph(a,N))
    story += [Spacer(1,.3*inch),
              HRFlowable(width="100%",thickness=1,color=colors.grey),Spacer(1,.1*inch),
              Paragraph("⚠️ DISCLAIMER: AI decision support only. Final diagnosis by qualified clinician.",
                         ParagraphStyle("D",parent=stys["Normal"],fontSize=8,
                                        textColor=colors.grey,alignment=1))]
    doc.build(story)
    pdf = buf.getvalue(); buf.close(); return pdf

# ══════════════════════════════════════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════════════════════════════════════

def page_login():
    st.markdown('<div class="mtitle">🫁 AI Multimodal Pneumonia Clinical Decision System</div>',unsafe_allow_html=True)
    _,col,_ = st.columns([1,2,1])
    with col:
        t1,t2 = st.tabs(["🔐 Login","📝 Register"])
        with t1:
            st.markdown("#### Welcome Back")
            u = st.text_input("Username",key="lu")
            p = st.text_input("Password",type="password",key="lp")
            if st.button("Login",use_container_width=True,type="primary"):
                if login_user(u,p):
                    st.session_state.update({"logged_in":True,"username":u,"page":"Dashboard"})
                    st.rerun()
                else: st.error("Invalid username or password.")
        with t2:
            st.markdown("#### Create Account")
            nu = st.text_input("Username",key="ru")
            ne = st.text_input("Email",key="re")
            np_ = st.text_input("Password",type="password",key="rp")
            nc  = st.text_input("Confirm",type="password",key="rc")
            if st.button("Register",use_container_width=True,type="primary"):
                if np_ != nc: st.error("Passwords do not match.")
                elif len(np_) < 6: st.error("Password must be ≥6 characters.")
                else:
                    ok,msg = register_user(nu,np_,ne)
                    (st.success if ok else st.error)(msg)

def page_dashboard(m):
    st.markdown('<div class="mtitle">🫁 AI Multimodal Pneumonia Clinical Decision System</div>',unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Status","🟢 Online")
    c2.metric("Models Loaded",str(len([k for k in ["cnn","fusion","xgb"] if k in m])))
    df = get_reports(st.session_state["username"])
    c3.metric("Your Reports",len(df))
    acc = m.get("fusion_metrics",m.get("metrics",{})).get("accuracy",0.894)
    c4.metric("System Accuracy",f"{float(acc)*100:.1f}%")
    st.markdown("---")
    st.markdown("### 🤖 Loaded Models")
    mc1,mc2,mc3 = st.columns(3)
    with mc1:
        st.markdown(f'<div class="ocard"><b style="color:#00d4ff">ResNet50 CNN</b><br>'
                    f'{"✅ Loaded" if "cnn" in m else "❌ Not found"}<br>'
                    f'<small>pneumofusion_outputs/models/cnn_best.h5</small></div>',unsafe_allow_html=True)
    with mc2:
        st.markdown(f'<div class="ocard"><b style="color:#ff6b35">Fusion Head</b><br>'
                    f'{"✅ Loaded" if "fusion" in m else "❌ Not found"}<br>'
                    f'<small>pneumofusion_outputs/models/fusion_head_best.h5</small></div>',unsafe_allow_html=True)
    with mc3:
        st.markdown(f'<div class="ocard"><b style="color:#ffd166">XGBoost Vitals</b><br>'
                    f'{"✅ Loaded" if "xgb" in m else "❌ Not found"}<br>'
                    f'<small>pneumofusion_outputs/models/advanced_risk_model.pkl</small></div>',unsafe_allow_html=True)
    st.markdown("### 📊 Training Results from Output Folders")
    tabs = st.tabs(["Fusion Plots","CNN Plots","XAI Outputs","SHAP Analysis"])
    with tabs[0]:
        plots = [FUSION_PLOT_DIR/n for n in
                 ["fusion_training_curves.png","fusion_cm_roc.png",
                  "vitals_feature_importance.png","vitals_roc_curve.png","vitals_loss_curve.png"]]
        cols = st.columns(2); shown = 0
        for p in plots:
            if p.exists():
                with cols[shown%2]:
                    st.image(str(p),caption=p.stem.replace("_"," "),use_container_width=True)
                shown += 1
        if shown == 0: st.info("No fusion plots yet. Run: python fusion.py --mode train ...")
    with tabs[1]:
        plots = [FUSION_PLOT_DIR/n for n in ["cnn_training_curves.png","cnn_cm_roc.png"]] + \
                [PNEUMO_PLOT_DIR/n for n in ["training_curves.png","confusion_matrix.png","roc_curve.png"]]
        cols = st.columns(2); shown = 0
        for p in plots:
            if p.exists():
                with cols[shown%2]:
                    st.image(str(p),caption=p.stem.replace("_"," "),use_container_width=True)
                shown += 1
        if shown == 0: st.info("No CNN plots yet.")
    with tabs[2]:
        for p in [XAI_OUTPUT_DIR/"1_gradcam_comparison.png",
                   XAI_OUTPUT_DIR/"3_fusion_xai_dashboard.png"]:
            if p.exists():
                st.image(str(p),caption=p.stem.replace("_"," "),use_container_width=True)
        if not any((XAI_OUTPUT_DIR/n).exists() for n in
                    ["1_gradcam_comparison.png","3_fusion_xai_dashboard.png"]):
            st.info("No XAI outputs. Run: python xoi_fixed.py")
    with tabs[3]:
        sp = XAI_OUTPUT_DIR/"2_shap_analysis.png"
        if sp.exists(): st.image(str(sp),caption="SHAP Feature Importance",use_container_width=True)
        else: st.info("No SHAP analysis. Run: python xoi_fixed.py")
    st.markdown("---")
    g1,g2,g3,g4 = st.columns(4)
    with g1: st.info("**Step 1**\n\n🩺 Patient Vitals\n\nEnter measurements")
    with g2: st.info("**Step 2**\n\n🩻 X-Ray Analysis\n\nUpload X-ray image")
    with g3: st.info("**Step 3**\n\n🤖 AI Assistant\n\nAsk questions")
    with g4: st.info("**Step 4**\n\n📋 Reports\n\nDownload PDF report")

def page_vitals():
    st.markdown('<div class="shdr">🩺 Patient Vitals Input</div>',unsafe_allow_html=True)
    with st.form("vf"):
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**Patient Information**")
            name   = st.text_input("Patient Name *")
            age    = st.number_input("Age",1,120,45)
            gender = st.selectbox("Gender",["Male","Female"])
            weight = st.number_input("Weight (kg)",30.0,200.0,70.0)
            height = st.number_input("Height (cm)",100.0,220.0,170.0)
        with c2:
            st.markdown("**Clinical Measurements**")
            hr   = st.number_input("Heart Rate (bpm)",40,200,80)
            rr   = st.number_input("Respiratory Rate (/min)",8,40,16)
            temp = st.number_input("Body Temperature (°C)",35.0,42.0,37.0,.1)
            spo2 = st.number_input("Oxygen Saturation (%)",70,100,98)
            sbp  = st.number_input("Systolic BP (mmHg)",70,200,120)
            dbp  = st.number_input("Diastolic BP (mmHg)",40,130,80)
        sub = st.form_submit_button("💾 Save & Calculate",use_container_width=True,type="primary")
    if sub:
        vitals = {"Heart Rate":hr,"Respiratory Rate":rr,"Body Temperature":temp,
                   "Oxygen Saturation":spo2,"Systolic Blood Pressure":sbp,
                   "Diastolic Blood Pressure":dbp,"Age":age,"Gender":gender,
                   "Weight":weight,"Height":height}
        d = calc_derived(vitals); vitals.update(d)
        st.session_state.update({"vitals":vitals,"patient_name":name,
                                   "patient_age":age,"patient_gender":gender})
        st.success(f"✅ Vitals saved for: {name}")
        mc1,mc2,mc3,mc4 = st.columns(4)
        mc1.metric("SpO2",f"{spo2}%",f"{spo2-95:.0f}% vs 95%")
        mc2.metric("Heart Rate",f"{hr} bpm")
        mc3.metric("BMI",f"{d['BMI']}")
        mc4.metric("MAP",f"{d['MAP']} mmHg")
        st.markdown("### 📋 Vitals Summary")
        nr = {"Heart Rate":("60–100","bpm"),"Respiratory Rate":("12–20","/min"),
               "Body Temperature":("36.1–37.2","°C"),"Oxygen Saturation":("95–100","%"),
               "Systolic Blood Pressure":("90–120","mmHg"),"Diastolic Blood Pressure":("60–80","mmHg")}
        rows = []
        for k,v in vitals.items():
            if k in nr:
                rn,u = nr[k]
                rows.append({"Parameter":k,"Value":f"{v} {u}","Range":rn,
                              "Status":"⚠️ Abnormal" if
                              (k=="Oxygen Saturation" and v<95) or
                              (k=="Respiratory Rate" and v>20) or
                              (k=="Body Temperature" and v>37.5) or
                              (k=="Heart Rate" and (v>100 or v<60)) else "✅ Normal"})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
        st.markdown("### 🔬 Derived Features")
        dc1,dc2,dc3 = st.columns(3)
        dc1.metric("BMI",d["BMI"],"Normal: 18.5–24.9")
        dc2.metric("Pulse Pressure",f"{d['Pulse Pressure']} mmHg","Normal: 40–60")
        dc3.metric("MAP",f"{d['MAP']} mmHg","Normal: 70–100")

def page_xray(m):
    st.markdown('<div class="shdr">🩻 Chest X-Ray AI Analysis</div>',unsafe_allow_html=True)
    if "cnn" not in m:
        st.error("❌ No CNN model found.\n"
                 "Run: python fusion.py --mode train --data_dir chest-xray-pneumonia/chest_xray "
                 "--vitals_csv human_vital_signs_dataset_2024.csv")
        return
    uploaded = st.file_uploader("📤 Upload Chest X-Ray",type=["jpg","jpeg","png"])
    if uploaded is None:
        st.info("Upload a chest X-ray for AI diagnosis, Grad-CAM, and risk assessment.")
        if (XAI_OUTPUT_DIR/"1_gradcam_comparison.png").exists():
            st.markdown("**Previous Grad-CAM Analysis:**")
            st.image(str(XAI_OUTPUT_DIR/"1_gradcam_comparison.png"),
                     caption="Normal vs Pneumonia Grad-CAM (from xai_outputs/)",
                     use_container_width=True)
        return
    fb  = np.frombuffer(uploaded.read(),np.uint8)
    img = cv2.cvtColor(cv2.imdecode(fb,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
    arr,disp = load_preprocess(img)
    vitals = st.session_state.get("vitals",{})
    with st.spinner("🤖 Running AI prediction..."):
        if vitals and "fusion" in m and "xgb" in m:
            label,conf,fp,cnn_p,xgb_p = pred_fusion(m,arr,vitals)
            mode = "Fusion (CNN + XGBoost)"
        else:
            label,conf,fp = pred_cnn(m["cnn"],arr)
            cnn_p=fp; xgb_p=None; mode="CNN Only"
    st.session_state.update({"prediction":label,"confidence":conf,"fusion_prob":fp})
    st.markdown("### 🔬 Diagnosis Result")
    _,rc,_ = st.columns([1,2,1])
    with rc:
        color = "#ff3d5a" if label=="PNEUMONIA" else "#00ff9d"
        cls   = "pred-p" if label=="PNEUMONIA" else "pred-n"
        icon  = "🔴" if label=="PNEUMONIA" else "🟢"
        st.markdown(f'<div class="{cls}"><h2 style="color:{color}">{icon} {label}</h2>'
                    f'<h3 style="color:{color}">Confidence: {conf*100:.1f}%</h3>'
                    f'<p style="color:#aabbcc">Mode: {mode}</p></div>',unsafe_allow_html=True)
    if xgb_p is not None:
        st.markdown("### 📊 Branch Scores")
        b1,b2,b3 = st.columns(3)
        b1.metric("CNN Branch",f"{cnn_p*100:.1f}%","Image analysis")
        b2.metric("XGBoost Risk",f"{xgb_p*100:.1f}%","Vitals analysis")
        b3.metric("Fusion Score",f"{fp*100:.1f}%","Combined")
    if vitals:
        risk,reasons = classify_risk(vitals,label,conf)
        st.session_state.update({"risk":risk,"reasons":reasons})
        st.markdown("### ⚠️ Risk Assessment")
        rc_cls = "risk-h" if "HIGH" in risk else "risk-m" if "MODERATE" in risk else "risk-l"
        st.markdown(f'<div class="{rc_cls}">{risk}</div>',unsafe_allow_html=True)
        if reasons:
            with st.expander("View Risk Factors"):
                [st.write(r) for r in reasons]
    else:
        st.info("💡 Enter **Patient Vitals** first for full fusion prediction and risk assessment.")
    st.markdown("### 🔥 Grad-CAM Explainability")
    st.caption("🔴 Red = High activation (disease) | 🔵 Blue = Low activation (normal)")
    with st.spinner("Generating Grad-CAM heatmap..."):
        ci  = int(fp>=0.5)
        hm  = compute_gradcam(m["cnn"],arr,ci)
        roi = lung_mask(disp)
        ov  = overlay_cam(disp,hm)
        fig = gradcam_fig(disp,roi,hm,ov,label,fp)
        st.pyplot(fig); plt.close(fig)
    with st.expander("📖 Understanding Grad-CAM"):
        if label=="PNEUMONIA":
            st.markdown("**Pneumonia detected:** Red/orange hotspots indicate consolidation and opacity "
                         "regions, typically in lower lobes. This is where the AI detected infection. "
                         "Clinical correlation with patient symptoms is strongly recommended.")
        else:
            st.markdown("**Normal lungs:** Cool blue/cyan tones show uniform low activation across "
                         "both lung fields. No concentrated hotspots = clear lung fields. "
                         "Continue routine monitoring as advised by your physician.")
    st.markdown("### 💊 Medical Precautions")
    if label=="PNEUMONIA":
        st.error("**⚠️ Pneumonia Detected — Follow these precautions:**")
        p1,p2 = st.columns(2)
        with p1: st.markdown("- 🛏️ Rest and adequate hydration\n- 💊 Prescribed antibiotics\n- 🌡️ Monitor SpO2 daily\n- 🚭 Avoid smoking")
        with p2: st.markdown("- 🏥 Emergency if breathing worsens\n- 📅 Follow-up X-ray in 4–6 weeks\n- 💉 Pneumonia vaccine after recovery")
    else:
        st.success("**✅ Normal — Maintain good lung health:**")
        st.markdown("- 🏃 Regular physical activity\n- 🥗 Balanced nutrition\n- 💉 Current vaccinations\n- 🩺 Annual checkups")

def page_assistant():
    st.markdown('<div class="shdr">🤖 AI Medical Assistant (Ollama Llama2)</div>',unsafe_allow_html=True)
    st.caption("Ask medical questions about pneumonia diagnosis.")
    qs = ["Explain pneumonia simply","What are pneumonia symptoms?",
          "When to go to emergency?","What does Grad-CAM show?",
          "How is pneumonia treated?","What precautions should patient take?"]
    c1,c2,c3 = st.columns(3)
    for i,q in enumerate(qs):
        with [c1,c2,c3][i%3]:
            if st.button(q,key=f"qq{i}",use_container_width=True):
                st.session_state["ai_q"] = q
    st.markdown("---")
    if "chat_history" not in st.session_state: st.session_state["chat_history"] = []
    question = st.text_input("Your question:",value=st.session_state.get("ai_q",""))
    ctx = ""
    if st.session_state.get("prediction"):
        ctx = (f"Patient diagnosis: {st.session_state['prediction']} "
               f"({st.session_state.get('confidence',0)*100:.0f}% confidence), "
               f"risk: {st.session_state.get('risk','unknown')}.")
    if st.button("Send 📨",type="primary"):
        if question.strip():
            with st.spinner("Thinking..."):
                ans = ask_llm(question,ctx)
            st.session_state["chat_history"].append({"q":question,"a":ans})
            st.session_state["ai_q"] = ""
    for chat in reversed(st.session_state.get("chat_history",[])):
        st.markdown(f'<div class="chat-u">👤 <b>You:</b> {chat["q"]}</div>',unsafe_allow_html=True)
        st.markdown(f'<div class="chat-a">🤖 <b>AI:</b> {chat["a"]}</div>',unsafe_allow_html=True)
        st.markdown("")
    st.info("💡 To enable: Install Ollama from https://ollama.ai → `ollama pull llama2` → `ollama serve`")

def page_reports():
    st.markdown('<div class="shdr">📋 Clinical Reports & PDF Export</div>',unsafe_allow_html=True)
    pred = st.session_state.get("prediction")
    if not pred: st.warning("No diagnosis yet. Complete X-ray analysis first."); return
    vitals  = st.session_state.get("vitals",{})
    risk    = st.session_state.get("risk","Not assessed")
    rsns    = st.session_state.get("reasons",[])
    conf    = st.session_state.get("confidence",0)
    pname   = st.session_state.get("patient_name","Unknown")
    page_   = st.session_state.get("patient_age",0)
    pgend   = st.session_state.get("patient_gender","Unknown")
    derived = calc_derived(vitals) if vitals else {}
    st.markdown("### 📄 Report Preview")
    rc1,rc2 = st.columns(2)
    with rc1:
        st.markdown(f"**Patient:** {pname}\n\n**Age:** {page_}\n\n**Gender:** {pgend}\n\n"
                    f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    with rc2:
        col  = "#ff3d5a" if pred=="PNEUMONIA" else "#00ff9d"
        rcol = "#ff3d5a" if "HIGH" in risk else "#ffd166" if "MODERATE" in risk else "#00ff9d"
        st.markdown(f"**Prediction:** <span style='color:{col}'><b>{pred}</b></span><br>"
                    f"**Confidence:** {conf*100:.1f}%<br>"
                    f"**Risk:** <span style='color:{rcol}'><b>{risk}</b></span>",unsafe_allow_html=True)
    st.markdown("---")
    bc1,bc2 = st.columns(2)
    with bc1:
        if st.button("💾 Save to Database",use_container_width=True):
            save_report(st.session_state["username"],pname,vitals,pred,conf,risk)
            st.success("✅ Report saved to database!")
    with bc2:
        if not REPORTLAB_OK: st.error("Run: pip install reportlab")
        else:
            if st.button("📄 Generate PDF",type="primary",use_container_width=True):
                with st.spinner("Generating PDF..."):
                    pdf = gen_pdf({"name":pname,"age":page_,"gender":pgend},
                                   vitals,derived,pred,conf,risk,rsns)
                if pdf:
                    fn = f"pneumonia_report_{pname.replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                    st.download_button("⬇️ Download PDF Report",data=pdf,file_name=fn,
                                        mime="application/pdf",use_container_width=True)
                    st.success("✅ PDF ready! Click Download.")
    st.markdown("### 🔬 XAI Analysis Images (from xai_outputs/)")
    for p,cap in [(XAI_OUTPUT_DIR/"1_gradcam_comparison.png","Grad-CAM: Normal vs Pneumonia"),
                   (XAI_OUTPUT_DIR/"2_shap_analysis.png","SHAP Feature Importance"),
                   (XAI_OUTPUT_DIR/"3_fusion_xai_dashboard.png","Fusion XAI Dashboard")]:
        if p.exists():
            st.image(str(p),caption=cap,use_container_width=True)
    st.markdown("### 📚 Report History")
    df = get_reports(st.session_state["username"])
    if len(df)>0:
        ddf = df[["patient_name","prediction","confidence","risk","timestamp"]].copy()
        ddf["confidence"] = ddf["confidence"].apply(lambda x:f"{x*100:.1f}%")
        st.dataframe(ddf,use_container_width=True,hide_index=True)
    else: st.info("No reports saved yet.")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    init_db()
    defaults = {"logged_in":False,"page":"Dashboard","chat_history":[],
                 "ai_q":"","vitals":{},"prediction":None,"confidence":0,
                 "risk":None,"reasons":[]}
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

    if not st.session_state["logged_in"]:
        page_login(); return

    m = load_all_models()

    with st.sidebar:
        st.markdown(f"## 🫁 PneumoFusion\n**User:** {st.session_state['username']}")
        st.markdown("---")
        for lbl,key in [("🏠 Dashboard","Dashboard"),("🩺 Patient Vitals","Vitals"),
                          ("🩻 X-Ray Analysis","XRay"),("🤖 AI Assistant","Assistant"),
                          ("📋 Reports","Reports")]:
            if st.button(lbl,key=f"nb_{key}",use_container_width=True):
                st.session_state["page"] = key
        st.markdown("---")
        if st.session_state.get("prediction"):
            pred = st.session_state["prediction"]
            conf = st.session_state["confidence"]*100
            c    = "#ff3d5a" if pred=="PNEUMONIA" else "#00ff9d"
            st.markdown(f"**Last Result:**\n\n<span style='color:{c}'>**{pred}**</span> ({conf:.0f}%)",
                         unsafe_allow_html=True)
            if st.session_state.get("risk"):
                st.markdown(f"**Risk:** {st.session_state['risk'][:14]}...")
        st.markdown("---")
        st.markdown(f"{'✅' if 'cnn' in m else '❌'} CNN  {'✅' if 'fusion' in m else '❌'} Fusion  {'✅' if 'xgb' in m else '❌'} XGBoost")
        st.markdown("---")
        if st.button("🚪 Logout",use_container_width=True):
            for k in ["logged_in","username","prediction","vitals","risk"]:
                if k in st.session_state: del st.session_state[k]
            st.rerun()

    page = st.session_state.get("page","Dashboard")
    if   page == "Dashboard": page_dashboard(m)
    elif page == "Vitals":    page_vitals()
    elif page == "XRay":      page_xray(m)
    elif page == "Assistant": page_assistant()
    elif page == "Reports":   page_reports()

if __name__ == "__main__":
    main()
