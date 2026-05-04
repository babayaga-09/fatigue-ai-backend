import numpy as np
import json
import pickle
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBClassifier

print("Retraining models locally to fix version mismatch...")

# ── Load scaler params ─────────────────────────────────────────
with open("scaler_params.json") as f:
    sp = json.load(f)

X_min = np.array(sp["X_min"], dtype=np.float32)
X_max = np.array(sp["X_max"], dtype=np.float32)
y_min = sp["y_min"][0]
y_max = sp["y_max"][0]

# ── Generate synthetic training data ──────────────────────────
# We use realistic football workload ranges to train the
# sklearn/xgboost models locally so versions match exactly
np.random.seed(42)
N = 2000

passes       = np.random.randint(20, 90,  N).astype(float)
def_actions  = np.random.randint(0,  15,  N).astype(float)
carries      = np.random.randint(0,  30,  N).astype(float)
pressures    = np.random.randint(0,  25,  N).astype(float)
shots        = np.random.randint(0,  8,   N).astype(float)
fouls        = np.random.randint(0,  6,   N).astype(float)
rest_days    = np.random.randint(1,  10,  N).astype(float)

workload_score = (
    passes      * 0.3 +
    def_actions * 0.4 +
    carries     * 0.2 +
    fouls       * 0.5
)

# Pass accuracy — decreases with high workload, low rest
base_acc = 82.0
acc = (
    base_acc
    - (workload_score / 30)
    + (rest_days * 0.5)
    + np.random.normal(0, 3, N)
)
acc = np.clip(acc, 55, 98)

# Form momentum — rolling avg approximation
form = acc * 0.7 + np.random.normal(80, 5, N) * 0.3
form = np.clip(form, 55, 98)

# Injury risk — high workload + low rest = risk
workload_75 = np.percentile(workload_score, 75)
rest_25     = np.percentile(rest_days, 25)
injury_label = (
    (workload_score > workload_75) & (rest_days < rest_25)
).astype(int)

print(f"Injury positive rate: {injury_label.mean()*100:.1f}%")

# ── Feature matrices ───────────────────────────────────────────
X_injury = np.column_stack([
    passes, def_actions, carries,
    pressures, fouls, rest_days, workload_score
])

X_form = np.column_stack([
    passes, acc, def_actions, rest_days, workload_score
])

# ── Train XGBoost injury model ─────────────────────────────────
print("Training XGBoost injury model...")
injury_model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    eval_metric='logloss',
    random_state=42,
)
injury_model.fit(X_injury, injury_label)
train_acc = (injury_model.predict(X_injury) == injury_label).mean()
print(f"  Injury model accuracy: {train_acc*100:.1f}%")

# ── Train GBM form model ───────────────────────────────────────
print("Training GBM form momentum model...")
form_model = GradientBoostingRegressor(
    n_estimators=150,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42,
)
form_model.fit(X_form, form)
print(f"  Form model R²: {form_model.score(X_form, form):.3f}")

# ── Save both models ───────────────────────────────────────────
with open("injury_model.pkl", "wb") as f:
    pickle.dump(injury_model, f)

with open("form_model.pkl", "wb") as f:
    pickle.dump(form_model, f)

# ── Update scaler_params with workload stats ───────────────────
sp["workload_75"] = float(workload_75)
sp["rest_25"]     = float(rest_25)

with open("scaler_params.json", "w") as f:
    json.dump(sp, f, indent=2)

print("\n✅ All models retrained and saved locally!")
print("   injury_model.pkl — XGBoost")
print("   form_model.pkl   — GBM")
print("   scaler_params.json — updated")

# ── Quick sanity checks ────────────────────────────────────────
test = np.array([[65, 6, 12, 8, 2, 4, 65*0.3+6*0.4+12*0.2+2*0.5]])
prob = injury_model.predict_proba(test)[0][1]
print(f"\n🔍 Injury risk for average midfielder: {prob*100:.1f}%")

test_form = np.array([[65, 82.0, 6, 4, 65*0.3+6*0.4+12*0.2+2*0.5]])
momentum  = form_model.predict(test_form)[0]
print(f"🔍 Form momentum for same player: {momentum:.1f}%")