# 🫁 PneumoFusion

> *Teaching a computer to look at chest X-rays and vital signs the way a doctor does — together, not separately.*

<div align="center">
  <img src="<img width="1659" height="1059" alt="image" src="https://github.com/user-attachments/assets/5ed7f41d-3626-4d9d-9da3-1ab68c7cf359" />

---

## So, what is this?

Picture a doctor looking at a chest X-ray. They don't *just* look at the image — they also glance at the patient's oxygen levels, check if there's a fever, notice the breathing rate. All of that happens almost instinctively, in their head, at the same time.

Most AI pneumonia-detection tools don't do that. They look at the X-ray. Just the X-ray. They never ask "but what's the oxygen saturation?" — and that's a problem, because two patients can have nearly identical X-rays and very different conditions.

**PneumoFusion** tries to close that gap. It runs a chest X-ray through a CNN, runs the patient's vitals through a separate model, and then *fuses* both opinions into one final, more confident diagnosis. And because "the AI said so" isn't good enough in medicine, it also shows you exactly *why* it thinks what it thinks — with heatmaps on the X-ray and a breakdown of which vital signs mattered most.

It's not trying to replace doctors. It's trying to be the assistant that never gets tired after the 80th X-ray of the day.

---

## The two brains

**Brain #1 — looks at the X-ray.**
A ResNet50 trained in two stages (first gently, then more aggressively) on 5,863 chest X-rays. By itself, it gets things right about 83% of the time.

**Brain #2 — looks at the numbers.**
An XGBoost model trained on 11 vital signs — heart rate, oxygen saturation, temperature, the usual suspects — plus three values it calculates on its own (HRV, MAP, BMI). On its own, it's surprisingly sharp: ~98% accuracy.

**The fusion** takes what both brains "saw" — not just their final answer, but their internal feature representations — and combines them through a small neural network. The result: fewer missed pneumonia cases. Specifically, *53% fewer* than the X-ray model working alone. That's the number that actually matters in a hospital.

```
            Chest X-Ray                 Vital Signs
                │                            │
          ResNet50 CNN                  XGBoost
                │                            │
        129 features                   13 features
                └──────────┬─────────────────┘
                           │
                    Fusion Head (142-dim)
                           │
                  "Pneumonia or not?"
                           │
              ┌────────────┴────────────┐
          Grad-CAM                    SHAP
       (show the X-ray)          (show the vitals)
```

---

## Why bother explaining itself?

Because a model that says "87% pneumonia" and nothing else is asking a doctor to trust it blindly. That doesn't fly in medicine, and honestly it shouldn't.

So every prediction comes with receipts:

- **Grad-CAM** lights up the exact region of the lung that triggered the CNN's decision. Healthy lungs glow a calm blue. Pneumonia cases show a hot, focused patch — usually in the lower lobes, which lines up with how bacterial pneumonia actually behaves. That's not a coincidence; it's a good sign the model learned something real.
- **SHAP** does the same thing for the vital signs side. It'll tell you, for instance, that this particular patient's risk score is being driven mostly by low oxygen saturation and a high respiratory rate — not just "the model is 91% confident," but *why*.

---

## What's actually in the box

| Piece | What it does |
|---|---|
| 🩻 `fusion.py` | Trains the ResNet50 CNN and the Fusion Head |
| 💉 `vital.py` | Trains the XGBoost model on vital signs |
| 🔍 `xoi_fixed.py` | Generates all the Grad-CAM and SHAP visuals |
| 🖥️ `app.py` | The Streamlit app — where a doctor actually clicks around |
| 🤖 Ollama Llama2 | A local AI assistant patients/doctors can literally talk to about the diagnosis |
| 📄 ReportLab | Spits out a proper PDF report at the end, because paperwork still exists |

Everything runs on a regular CPU. No GPU required, no cloud bill, no sending patient data anywhere it shouldn't go.

---

## Getting it running

```bash
git clone https://github.com/lodemounika/pneumonia_grad
cd pneumonia_grad

pip install tensorflow xgboost scikit-learn shap opencv-python
pip install streamlit reportlab joblib pandas numpy matplotlib seaborn
```

Drop the datasets in:
```
chest-xray-pneumonia/chest_xray/   ← Kaggle pneumonia X-rays
human_vital_signs_dataset_2024.csv ← vitals data
```

Then train everything (grab a coffee, it takes about 75 minutes on CPU):
```bash
python fusion.py --mode train \
    --data_dir "chest-xray-pneumonia/chest_xray" \
    --vitals_csv human_vital_signs_dataset_2024.csv
```

Generate the explainability visuals:
```bash
python xoi_fixed.py
```

And finally, launch the actual app:
```bash
streamlit run app.py
```
It'll pop up at `localhost:8501`. Log in, enter some vitals, upload an X-ray, and watch it work.

*(Want the AI chat assistant too? `ollama pull llama2` then `ollama serve` in a separate terminal — keep it running in the background.)*

---

## Does it actually work? Numbers below.

| Model | Accuracy | F1 | AUC |
|---|---|---|---|
| X-ray only (CNN) | 82.69% | 82.9% | 93.94% |
| Vitals only (XGBoost) | 97.8% | 97.6% | 99.1% |
| **Fused together** | **89.42%** | **91.52%** | **95.75%** |

The number I actually care about: the CNN alone missed **72 pneumonia cases** out of 390 in testing. Fused with vitals, that dropped to **34**. Half as many patients slipping through the cracks. That's the whole point of this project, really — everything else is in service of that one number.

---

## The data behind it

- **Chest X-rays**: 5,863 images from the [Kaggle Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), originally collected from pediatric patients in Guangzhou and published by Kermany et al. in *Cell* (2018).
- **Vital signs**: The Human Vital Signs Dataset 2024 — 11 clinical features per patient, plus three I calculate myself (heart rate variability, mean arterial pressure, BMI).

One honest caveat: these two datasets come from different patients. Nobody in this project has *both* an X-ray and matching vitals recorded at the same visit — that data simply isn't publicly available yet. I worked around it with label-matched sampling during fusion training, but a real clinical deployment would need genuinely matched patient records. It's on the future-work list.

---

## What's next, if I keep going

- A dataset where the same patient actually has both an X-ray *and* vitals (the real fix for the caveat above)
- Telling bacterial pneumonia apart from viral — right now it's just "pneumonia, yes or no"
- A version light enough to run on a phone, for places that don't have a spare laptop lying around
- Some way of saying "I'm not sure about this one" instead of always giving a confident number
- An actual clinical trial, comparing this against real radiologists, because that's the only review that really counts

---

## Built on the shoulders of

- Rajpurkar et al., *CheXNet* (2017) — proof that CNNs can read X-rays well
- Kermany et al., *Cell* (2018) — the dataset and the transfer-learning approach
- He et al., *Deep Residual Learning* (2016) — ResNet, obviously
- Selvaraju et al., *Grad-CAM* (2017) — the heatmaps
- Lundberg & Lee, *SHAP* (2017) — the explainability for the vitals side
- Chen & Guestrin, *XGBoost* (2016) — the vitals model itself

---

## Who made this

**Lode Mounika** 

Dept. of CSE (AI & ML), Nalla Malla Reddy Engineering College — 2025–26

---



*If this was useful or interesting, a ⭐ helps more than you'd think.*
