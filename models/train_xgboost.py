"""
HemoSmart - Transfusion Prediction Model (XGBoost + SHAP)
------------------------------------------------------------
Trains an XGBoost classifier on the simulated patient dataset to predict
whether a patient will need a blood transfusion, then generates SHAP
explainability output.

Requires: data/patient_data.csv (run create_dataset.py first)

Run:
    python models/train_xgboost.py

Outputs:
    models/xgb_model.pkl        -> trained model
    models/label_encoder.pkl    -> encoder for surgery_type
    models/shap_summary.png     -> feature importance plot
    models/metrics.txt          -> accuracy / precision / recall / F1
"""

import pandas as pd
import pickle
import shap
import matplotlib
matplotlib.use("Agg")  # no display needed, just save the plot
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
data = pd.read_csv("data/patient_data.csv")
print("Loaded dataset:", data.shape)

# ----------------------------------------------------------------------
# 2. Preprocess
# ----------------------------------------------------------------------
le = LabelEncoder()
data["surgery_type"] = le.fit_transform(data["surgery_type"])

X = data.drop("transfusion_needed", axis=1)
y = data["transfusion_needed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------------------------------------------------
# 3. Train XGBoost
# ----------------------------------------------------------------------
print("\nTraining XGBoost...")
model = XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss",
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\nAccuracy: {acc:.4f}")
print(report)

with open("models/metrics.txt", "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n\n")
    f.write(report)

# ----------------------------------------------------------------------
# 4. SHAP explainability
# ----------------------------------------------------------------------
print("Generating SHAP explanations...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("models/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved SHAP summary plot -> models/shap_summary.png")

# ----------------------------------------------------------------------
# 5. Save model + encoder
# ----------------------------------------------------------------------
with open("models/xgb_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("\nSaved models/xgb_model.pkl and models/label_encoder.pkl")
print("Done. Hand these files + metrics.txt off to your teammates.")