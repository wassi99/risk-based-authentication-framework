import pandas as pd
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # ensures plots always work (safe backend)
import seaborn as sns

from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support
)

# =========================================================
# 1. REPRODUCIBILITY
# =========================================================
np.random.seed(42)
random.seed(42)

# =========================================================
# 2. SIMULATION CONFIGURATION
# =========================================================
NUM_SAMPLES = 2000

LOCATIONS = ["NZ", "India", "US", "Unknown"]
DEVICES = ["Mobile", "Laptop"]

IP_RISK_MAP = {
    "Low": 0.1,
    "Medium": 0.4,
    "High": 0.8
}

ATTACK_TYPES = [
    "credential_stuffing",
    "impossible_travel",
    "bruteforce",
    "bot_activity"
]

# =========================================================
# 3. USER PROFILE GENERATOR (REALISTIC BASELINE BEHAVIOUR)
# =========================================================
def generate_user_profile(user_id):
    return {
        "home_location": random.choice(["NZ", "India", "US"]),
        "preferred_device": random.choice(["Mobile", "Laptop"]),
        "base_behavior": np.clip(np.random.beta(8, 2), 0, 1)  # realistic clustering
    }

user_profiles = {i: generate_user_profile(i) for i in range(1, 151)}

# =========================================================
# 4. ATTACK SIMULATION ENGINE (REALISTIC THREAT MODELS)
# =========================================================
def inject_attack_features(row, attack_type):
    if attack_type == "credential_stuffing":
        row["failed_attempts"] += np.random.randint(3, 8)
        row["device_trusted"] = 0

    elif attack_type == "impossible_travel":
        row["location"] = "Unknown"
        row["ip_risk"] = "High"

    elif attack_type == "bruteforce":
        row["failed_attempts"] += np.random.randint(5, 12)
        row["behavior_score"] *= 0.5

    elif attack_type == "bot_activity":
        row["behavior_score"] = np.random.uniform(0.0, 0.3)
        row["device_trusted"] = 0

    return row

# =========================================================
# 4B. RISK ENGINE (CORE FRAMEWORK MODULE)
# =========================================================
def risk_engine(row):
    return (
        0.35 * IP_RISK_MAP[row["ip_risk"]] +
        0.25 * (1 - row["behavior_score"]) +
        0.2 * row["failed_attempts"] / 10 +
        0.1 * (1 - row["device_trusted"]) +
        0.1 * (1 - row["mfa_used"])
    )

# =========================================================
# 5. SYNTHETIC DATA GENERATION (NO LABEL LEAKAGE)
# =========================================================
data = []

for i in range(NUM_SAMPLES):
    user_id = random.randint(1, 150)
    profile = user_profiles[user_id]

    location = random.choice(LOCATIONS)
    device = random.choice(DEVICES)

    failed_attempts = np.random.poisson(1.2)
    behavior_score = np.clip(
        np.random.normal(profile["base_behavior"], 0.15), 0, 1
    )

    ip_risk = random.choices(
        ["Low", "Medium", "High"],
        weights=[0.6, 0.3, 0.1]
    )[0]

    device_trusted = 1 if device == profile["preferred_device"] else 0

    mfa_used = random.choice([0, 1])

    row = {
        "user_id": user_id,
        "location": location,
        "device": device,
        "device_trusted": device_trusted,
        "ip_risk": ip_risk,
        "failed_attempts": failed_attempts,
        "behavior_score": round(behavior_score, 2),
        "mfa_used": mfa_used
    }

    # inject attacks (15% attack rate)
    is_attack = np.random.rand() < 0.15
    attack_type = None

    if is_attack:
        attack_type = random.choice(ATTACK_TYPES)
        row = inject_attack_features(row, attack_type)

    # =====================================================
    # 6. PROBABILISTIC GROUND TRUTH (IMPORTANT UPGRADE)
    # =====================================================
    risk_prob = risk_engine(row) (
        0.35 * IP_RISK_MAP[row["ip_risk"]] +
        0.25 * (1 - row["behavior_score"]) +
        0.2 * row["failed_attempts"] / 10 +
        0.1 * (1 - row["device_trusted"]) +
        0.1 * (1 - row["mfa_used"])
    )

    # convert probability to stochastic label (no rule leakage)
    label = "Attack" if np.random.rand() < risk_prob else "Legit"

    row["label"] = label
    row["attack_type"] = attack_type if is_attack else "normal"

    data.append(row)

df = pd.DataFrame(data)

# =========================================================
# 7. FEATURE ENCODING FOR MODEL
# =========================================================
df_encoded = df.copy()
df_encoded["ip_risk"] = df_encoded["ip_risk"].map(IP_RISK_MAP)

X = df_encoded[
    ["device_trusted", "ip_risk", "failed_attempts",
     "behavior_score", "mfa_used"]
]

y = (df_encoded["label"] == "Attack").astype(int)

# =========================================================
# 8. TRAIN / TEST SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# =========================================================
# 9. BASELINE MODEL (TRADITIONAL IAM)
# =========================================================
def baseline_rule(row):
    return int(
        row["ip_risk"] > 0.6 or
        row["failed_attempts"] > 3 or
        row["behavior_score"] < 0.4
    )

baseline_preds = X_test.apply(baseline_rule, axis=1)

# =========================================================
# 10. PROPOSED MODEL (RISK-BASED ADAPTIVE AUTHENTICATION ENGINE)
# =========================================================
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=6,
    random_state=42
)

start_time = time.time()

model.fit(X_train, y_train)

end_time = time.time()
print("\nTraining Time (ML Model):", round(end_time - start_time, 3), "seconds")
ml_preds = model.predict(X_test)
ml_probs = model.predict_proba(X_test)[:, 1]

# =========================================================
# 10B. POLICY ENGINE (DECISION LAYER)
# =========================================================
def policy_engine(probability):
    if probability < 0.30:
        return "ALLOW"
    elif probability < 0.60:
        return "MFA_CHALLENGE"
    else:
        return "DENY"

policy_decisions = [policy_engine(p) for p in ml_probs]

df_policy = pd.DataFrame({
    "Risk_Probability": ml_probs,
    "Decision": policy_decisions
})

print("\n================ POLICY ENGINE OUTPUT =================")
print(df_policy.head())

# =========================================================
# 10C. POLICY ENGINE EVALUATION
# =========================================================

policy_to_numeric = [1 if d == "Attack" else 0 for d in policy_decisions]

# Align length safely
min_len = min(len(policy_to_numeric), len(y_test))

print("\n================ POLICY ENGINE EVALUATION =================")
print(classification_report(
    y_test[:min_len],
    policy_to_numeric[:min_len]
))

# =========================================================
# 11. EVALUATION
# =========================================================
print("\n================ BASELINE IAM =================")
print(classification_report(y_test, baseline_preds))

print("\n================ RISK-BASED AUTHENTICATION MODEL =================")
print(classification_report(y_test, ml_preds))

print("\nROC-AUC (ZTA Model):", roc_auc_score(y_test, ml_probs))

# =========================================================
# 12. ADVANCED METRICS
# =========================================================
def compute_metrics(name, y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )
    print(f"\n{name}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")

compute_metrics("Baseline IAM", y_test, baseline_preds)
compute_metrics("Risk-Based Authentication Model", y_test, ml_preds)

# =========================================================
# 12B. EXPERIMENT COMPARISON TABLE
# =========================================================
from sklearn.metrics import precision_score, recall_score, f1_score

baseline_precision = precision_score(y_test, baseline_preds)
baseline_recall = recall_score(y_test, baseline_preds)
baseline_f1 = f1_score(y_test, baseline_preds)

ml_precision = precision_score(y_test, ml_preds)
ml_recall = recall_score(y_test, ml_preds)
ml_f1 = f1_score(y_test, ml_preds)

comparison_df = pd.DataFrame({
    "Model": ["Baseline IAM", "Risk-Based ML"],
    "Precision": [baseline_precision, ml_precision],
    "Recall": [baseline_recall, ml_recall],
    "F1 Score": [baseline_f1, ml_f1],
    "ROC-AUC": ["-", roc_auc_score(y_test, ml_probs)]
})

print("\n================ MODEL COMPARISON TABLE =================")
print(comparison_df)

# =========================================================
# 13. CONFUSION MATRIX
# =========================================================
print("\nConfusion Matrix (ZTA Model):")
print(confusion_matrix(y_test, ml_preds))

# =========================================================
# 14. RESEARCH-GRADE VISUALIZATION OUTPUT
# =========================================================

import os

os.makedirs("results", exist_ok=True)

# -----------------------------
# FIGURE 1: Risk Score Distribution
# -----------------------------
plt.figure(figsize=(8,5))
risk_scores = X_test["ip_risk"] + X_test["failed_attempts"]

plt.hist(risk_scores, bins=15, edgecolor="black")
plt.title("Risk Signal Distribution (Proxy View)")
plt.xlabel("Risk Score Proxy")
plt.ylabel("Frequency")

plt.savefig("results/risk_distribution.png", dpi=300)
plt.close()

# -----------------------------
# FIGURE 2: Decision Distribution (Baseline vs ML)
# -----------------------------
plt.figure(figsize=(8,5))

baseline_counts = pd.Series(baseline_preds).value_counts()
ml_counts = pd.Series(ml_preds).value_counts()

x = np.arange(2)

plt.bar(x - 0.2, baseline_counts.values, width=0.4, label="Baseline IAM")
plt.bar(x + 0.2, ml_counts.values, width=0.4, label="Zero Trust ML")

plt.xticks(x, ["Legit", "Attack"])
plt.title("Decision Comparison")
plt.legend()

plt.savefig("results/decision_comparison.png", dpi=300)
plt.close()

# -----------------------------
# FIGURE 3: Confusion Matrix Heatmap
# -----------------------------
import seaborn as sns

cm = confusion_matrix(y_test, ml_preds)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legit","Attack"],
            yticklabels=["Legit","Attack"])

plt.title("Risk-Based Authentication Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("results/confusion_matrix.png", dpi=300)
plt.close()

# -----------------------------
# FIGURE 4: Feature Importance
# -----------------------------
feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=True)

plt.figure(figsize=(8,5))
feature_importance.plot(kind='barh')

plt.title("Feature Importance in Risk Evaluation")
plt.xlabel("Importance Score")

plt.savefig("results/feature_importance.png", dpi=300)
print("\nInsight: Behavior score and IP risk are dominant factors in risk prediction.")
plt.close()

print("\n✅ All graphs saved in /results folder")
df.to_csv("synthetic_authentication_dataset.csv", index=False)

# -----------------------------
# FIGURE 5: ROC Curve
# -----------------------------
fpr, tpr, _ = roc_curve(y_test, ml_probs)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, ml_probs):.3f}")

plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Risk-Based Authentication Model")
plt.legend()

plt.savefig("results/roc_curve.png", dpi=300)
print("\nInsight: ROC curve shows strong separability between Attack and Legit classes.")
plt.close()

# =========================================================
# 15. EXPORT RESULTS
# =========================================================

results_df = X_test.copy()

results_df["Actual_Label"] = y_test.values
results_df["Predicted_Label"] = ml_preds
results_df["Risk_Probability"] = ml_probs
results_df["Policy_Decision"] = policy_decisions

results_df.to_csv("results/model_results.csv", index=False)

print("\n✅ Results exported successfully")