# =====================================================
# ADVANCED HUMAN VITAL SIGNS RISK PREDICTION SYSTEM
# INDUSTRY LEVEL IMPLEMENTATION
# =====================================================

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)

from xgboost import XGBClassifier

# =====================================================
# LOAD DATA
# =====================================================

data = pd.read_csv("human_vital_signs_dataset_2024.csv")

# =====================================================
# SELECT IMPORTANT FEATURES
# =====================================================

selected_features = [
    "Heart Rate",
    "Respiratory Rate",
    "Body Temperature",
    "Oxygen Saturation",
    "Systolic Blood Pressure",
    "Diastolic Blood Pressure",
    "Age",
    "Gender",
    "Derived_HRV",
    "Derived_MAP",
    "Derived_BMI"
]

data = data[selected_features + ["Risk Category"]]

# =====================================================
# PREPROCESSING
# =====================================================

label_encoder = LabelEncoder()
data["Risk Category"] = label_encoder.fit_transform(data["Risk Category"])

gender_encoder = LabelEncoder()
data["Gender"] = gender_encoder.fit_transform(data["Gender"])

X = data.drop("Risk Category", axis=1)
y = data["Risk Category"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Stratified split (important for medical datasets)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# MODEL WITH OPTIMIZED PARAMETERS
# =====================================================

model = XGBClassifier(
    n_estimators=400,
    max_depth=7,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

eval_set = [(X_train, y_train), (X_test, y_test)]

model.fit(
    X_train,
    y_train,
    eval_set=eval_set,
    verbose=False
)

# =====================================================
# EVALUATION
# =====================================================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)

print("\n🔥 FINAL TEST ACCURACY:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.savefig("advanced_confusion_matrix.png")
plt.show()

# =====================================================
# ROC CURVE + AUC
# =====================================================

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label="AUC = %0.4f" % roc_auc)
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("roc_curve.png")
plt.show()

print("🔥 AUC Score:", roc_auc)

# =====================================================
# PRECISION RECALL CURVE
# =====================================================

precision, recall, _ = precision_recall_curve(y_test, y_prob)

plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.savefig("precision_recall_curve.png")
plt.show()

# =====================================================
# LOSS CURVE
# =====================================================

results = model.evals_result()
train_loss = results["validation_0"]["logloss"]
test_loss = results["validation_1"]["logloss"]

plt.figure()
plt.plot(train_loss, label="Train Loss")
plt.plot(test_loss, label="Validation Loss")
plt.legend()
plt.title("Loss Curve")
plt.savefig("advanced_loss_curve.png")
plt.show()

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

importance = model.feature_importances_
feature_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8,6))
plt.barh(feature_df["Feature"], feature_df["Importance"])
plt.gca().invert_yaxis()
plt.title("Feature Importance")
plt.savefig("advanced_feature_importance.png")
plt.show()

print("\n🔥 Feature Importance Ranking:\n")
print(feature_df)

# =====================================================
# CROSS VALIDATION
# =====================================================

kfold = StratifiedKFold(n_splits=5)
cv_scores = cross_val_score(model, X_scaled, y, cv=kfold)

print("\n🔥 5-Fold Cross Validation Accuracy:")
print("Mean CV Accuracy:", cv_scores.mean())

# =====================================================
# SAVE FINAL OPTIMIZED MODEL
# =====================================================

joblib.dump(model, "advanced_risk_model.pkl")
joblib.dump(scaler, "advanced_scaler.pkl")
joblib.dump(label_encoder, "advanced_label_encoder.pkl")
joblib.dump(gender_encoder, "advanced_gender_encoder.pkl")

print("\n✅ Advanced Model Saved Successfully")