# =========================================================
# Risk-Based Adaptive Authentication Framework
# Author: Ramandeep Singh
# Course: Mater of Information Technology, WhiteCliffe NZ
# =========================================================

import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support,
    roc_curve
)

# =========================================================
# EXTENDED METRICS
# =========================================================
def extended_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    tpr = tp / (tp + fn)

    print("\n📊 Extended Evaluation Metrics")
    print(f"False Positive Rate: {fpr:.3f}")
    print(f"False Negative Rate: {fnr:.3f}")
    print(f"Detection Rate (TPR): {tpr:.3f}")

    return fpr, fnr, tpr


# =========================================================
# 1. SETUP
# =========================================================
np.random.seed(42)
random.seed(42)

os.makedirs("results", exist_ok=True)

NUM_SAMPLES = 2000
DEVICES = ["Mobile", "Laptop"]
IP_RISK_MAP = {"Low": 0.1, "Medium": 0.4, "High": 0.8}

ATTACK_TYPES = [
    "credential_stuffing",
    "impossible_travel",
    "bruteforce",
    "bot_activity"
]

# =========================================================
# 2. USER PROFILES
# =========================================================
def generate_user_profile(user_id):
    return {
        "home_location": random.choice(["NZ", "India", "US"]),
        "preferred_device": random.choice(["Mobile", "Laptop"]),
        "base_behavior": np.clip(np.random.beta(8, 2), 0, 1)
    }

user_profiles = {i: generate_user_profile(i) for i in range(1, 151)}
# =========================================================
# 3. ATTACK ENGINE
# =========================================================
def inject_attack_features(row, attack_type):

    if attack_type == "credential_stuffing":
        row["failed_attempts"] += np.random.randint(3, 8)
        row["device_trusted"] = 0

    elif attack_type == "impossible_travel":
        row["ip_risk"] = "High"

    elif attack_type == "bruteforce":
        row["failed_attempts"] += np.random.randint(5, 12)
        row["behavior_score"] *= 0.5

    elif attack_type == "bot_activity":
        row["behavior_score"] = np.random.uniform(0.0, 0.3)
        row["device_trusted"] = 0

    return row
# =========================================================
# 4. TEMPORAL STATE
# =========================================================
user_state = {
    i: {"last_failed_attempts": 0, "risk_trend": 0.0}
    for i in range(1, 151)
}

# =========================================================
# 5. DATA GENERATION
# =========================================================

data = []

for _ in range(NUM_SAMPLES):
    user_id = random.randint(1, 150)
    state = user_state[user_id]
    profile = user_profiles[user_id]

    # temporal update
    failed_attempts = np.random.poisson(1.2) + state["last_failed_attempts"]
    state["last_failed_attempts"] = failed_attempts

    # FIXED TEMPORAL MEMORY MODEL
    state["risk_trend"] = 0.7 * state["risk_trend"] + 0.3 * (failed_attempts > 2)
    trend_factor = state["risk_trend"]

    device = random.choice(DEVICES)

    behavior_score = np.clip(
        np.random.normal(profile["base_behavior"], 0.15) * (1 - trend_factor),
        0, 1
    )

    ip_risk = random.choices(
        ["Low", "Medium", "High"],
        weights=[0.6, 0.3, 0.1]
    )[0]

    device_trusted = 1 if device == profile["preferred_device"] else 0
    mfa_used = random.choice([0, 1])

    row = {
        "device_trusted": device_trusted,
        "ip_risk": ip_risk,
        "failed_attempts": failed_attempts,
        "behavior_score": round(behavior_score, 2),
        "mfa_used": mfa_used
    }

    is_attack = np.random.rand() < 0.15

    if is_attack:
        row = inject_attack_features(row, random.choice(ATTACK_TYPES))

    temporal_factor = np.clip(trend_factor * 1.5, 0, 1)

    risk_prob = np.clip((
        0.22 * IP_RISK_MAP[row["ip_risk"]] +
        0.18 * (1 - row["behavior_score"]) +
        0.22 * np.tanh(row["failed_attempts"] / 4) +
        0.15 * (1 - row["device_trusted"]) +
        0.08 * (1 - row["mfa_used"]) +
        0.15 * temporal_factor
    ), 0, 1)

    label = "Attack" if np.random.rand() < risk_prob else "Legit"
    row["label"] = label

    data.append(row)
# =========================================================
# CREATE DATAFRAME
# =========================================================
df = pd.DataFrame(data)
# =========================================================
# 6. FEATURES
# =========================================================
df_encoded = df.copy()
df_encoded["ip_risk"] = df_encoded["ip_risk"].map(IP_RISK_MAP)

X = df_encoded[
    ["device_trusted", "ip_risk", "failed_attempts", "behavior_score", "mfa_used"]
]

y = (df_encoded["label"] == "Attack").astype(int)

# =========================================================
# 7. SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# =========================================================
# 8. BASELINE IAM
# =========================================================
def baseline_rule(row):
    return int(
        row["ip_risk"] > 0.6 or
        row["failed_attempts"] > 3 or
        row["behavior_score"] < 0.4
    )

baseline_preds = X_test.apply(baseline_rule, axis=1)

# =========================================================
# 9. ML MODEL (STABLE)
# =========================================================
start = time.time()

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

ml_probs = model.predict_proba(X_test)[:, 1]

# recall improvement tuning (kept simple but effective)
optimal_threshold = 0.30
ml_preds = (ml_probs >= optimal_threshold).astype(int)

end = time.time()

print(f"\nTraining Time (ML Model): {round(end - start, 3)} seconds")

from sklearn.utils.class_weight import compute_class_weight

# Improve decision calibration (VERY IMPORTANT for recall/precision balance)
from sklearn.calibration import CalibratedClassifierCV

calibrated_model = CalibratedClassifierCV(model, method="isotonic", cv=3)
calibrated_model.fit(X_train, y_train)

ml_probs = calibrated_model.predict_proba(X_test)[:, 1]

# =========================================================
# 10. ADAPTIVE POLICY ENGINE
# =========================================================
adaptive_threshold = np.percentile(ml_probs, 75)
print("\n🔵 Adaptive Threshold:", adaptive_threshold)

def policy_engine(p):
    if p < adaptive_threshold * 0.8:
        return "ALLOW"
    elif p < adaptive_threshold:
        return "MFA_CHALLENGE"
    else:
        return "DENY"

policy_decisions = [policy_engine(p) for p in ml_probs]

print("\n================ POLICY ENGINE OUTPUT =================")
print(pd.DataFrame({"Risk": ml_probs, "Decision": policy_decisions}).head())

# =========================================================
# 11. EVALUATION
# =========================================================
print("\n================ BASELINE IAM =================")
print(classification_report(y_test, baseline_preds))

print("\n================ RISK-BASED MODEL =================")
print(classification_report(y_test, ml_preds))

roc = roc_auc_score(y_test, ml_probs)
print("\nROC-AUC:", roc)

def metrics(name, y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )
    print(f"\n{name}")
    print(f"Precision: {p:.3f}")
    print(f"Recall:    {r:.3f}")
    print(f"F1 Score:  {f1:.3f}")

metrics("Baseline IAM", y_test, baseline_preds)
metrics("Risk-Based Model", y_test, ml_preds)

cm = confusion_matrix(y_test, ml_preds)
print("\nConfusion Matrix:\n", cm)

fpr, fnr, tpr = extended_metrics(y_test, ml_preds)

# ================= TABLE =================
evaluation_summary = pd.DataFrame({
    "Metric": ["Precision", "Recall", "F1", "FPR", "FNR", "TPR"],
    "Value": [
        precision_recall_fscore_support(y_test, ml_preds, average="binary")[0],
        precision_recall_fscore_support(y_test, ml_preds, average="binary")[1],
        precision_recall_fscore_support(y_test, ml_preds, average="binary")[2],
        fpr, fnr, tpr
    ]
})

evaluation_summary.to_csv("results/evaluation_summary.csv", index=False)

def thesis_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    roc_auc = roc_auc_score(y_true, y_prob)

    print("\n📊 EVALUATION")
    print(f"Accuracy:     {accuracy:.3f}")
    print(f"Precision:    {precision:.3f}")
    print(f"Recall:       {recall:.3f}")
    print(f"Specificity:  {specificity:.3f}")
    print(f"F1 Score:     {f1:.3f}")
    print(f"ROC-AUC:      {roc_auc:.3f}")

thesis_metrics(y_test, ml_preds, ml_probs)
# =========================================================
# 12. VISUALIZATION
# =========================================================

plt.figure(figsize=(8,5))
risk_signal = X_test["ip_risk"] + X_test["failed_attempts"]
plt.hist(risk_signal, bins=15, edgecolor="black")
plt.title("Distribution of Risk Scores in Authentication Requests")
plt.xlabel("Aggregated Risk Score (Normalized)")
plt.ylabel("Frequency of Login Events")
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig("results/risk_distribution.png", dpi=300)
plt.close()

plt.figure(figsize=(8,5))
b = pd.Series(baseline_preds).value_counts()
m = pd.Series(ml_preds).value_counts()

plt.bar([0,1], b.values, width=0.4, label="Baseline")
plt.bar([0.4,1.4], m.values, width=0.4, label="ML Model")

plt.xticks([0.2,1.2], ["Legit", "Attack"])
plt.title("Authentication Decision Distribution Comparison")
plt.xlabel("Decision Type")
plt.ylabel("Count of Requests")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig("results/decision_comparison.png", dpi=300)
plt.close()

fpr_curve, tpr_curve, _ = roc_curve(y_test, ml_probs)

plt.figure(figsize=(6,5))
plt.plot(fpr_curve, tpr_curve, label=f"AUC={roc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.title("ROC Curve (Model Separability)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig("results/roc_curve.png", dpi=300)
plt.close()

# =========================================================
# 12. EXPORT
# =========================================================
results_df = X_test.copy()
results_df["Actual"] = y_test.values
results_df["Predicted"] = ml_preds
results_df["Risk"] = ml_probs
results_df["Policy"] = policy_decisions

results_df.to_csv("results/final_results.csv", index=False)

print("\n✅ ALL OUTPUTS GENERATED SUCCESSFULLY")
print("📁 Check /results folder for graphs + evaluation files")
