🧠 Risk-Based Adaptive Authentication Framework
(Hybrid Machine Learning + Policy Engine Simulation)
📌 Overview

This research presents a Risk-Based Adaptive Authentication Framework inspired by Zero Trust security principles for cloud-based identity and access management environments.

The framework simulates intelligent authentication decision-making by integrating three complementary components:

A probabilistic Risk Engine
A Machine Learning classification model
A rule-based Policy Engine

Together, these components dynamically evaluate authentication requests based on contextual and behavioural signals, including:

Device trust level
IP reputation score
Failed login attempts
Behavioural anomaly score
Multi-Factor Authentication (MFA) usage

This system was developed as part of a Master of Information Technology (Research Project) at Whitecliffe.

🎯 Research Objectives

The study aims to:

Design and simulate a risk-based adaptive authentication system aligned with Zero Trust principles
Compare traditional IAM (rule-based) authentication with AI-driven risk-based authentication
Evaluate the effectiveness of context-aware risk scoring mechanisms
Improve detection of malicious authentication attempts using behavioural analytics
Propose a lightweight and deployable security decision-making framework for cloud environments
🏗 System Architecture

The proposed framework consists of three core modules:

1️⃣ Risk Engine

Computes probabilistic risk scores using contextual authentication signals:

IP risk level
Device trust status
Behavioural deviation from baseline patterns
MFA usage
Login attempt frequency and anomalies
2️⃣ Machine Learning Engine

A supervised learning model (Random Forest Classifier) trained on synthetic authentication data to detect malicious patterns and classify login attempts as:

Legitimate
Attack
3️⃣ Policy Engine

Transforms computed risk probabilities into actionable authentication decisions:

ALLOW → Low-risk authentication
MFA_CHALLENGE → Medium-risk or uncertain behaviour
DENY → High-risk or likely attack
⚙️ Experimental Design

The study follows a simulation-based experimental methodology.

🔹 Dataset Generation
2000 synthetic authentication events
150 simulated user profiles
15% injected malicious attack scenarios
🔹 Attack Simulation Types
Credential stuffing
Impossible travel detection
Brute-force attempts
Bot-like automated activity
🔹 Experimental Setup

Two authentication systems are compared:

System	Description
Baseline IAM	Static rule-based authentication using threshold logic
Risk-Based Model	Machine learning-driven adaptive authentication system
📊 Evaluation Metrics

System performance is evaluated using:

Accuracy
Precision
Recall
F1-score
ROC-AUC
False Positive Rate
False Negative Rate
Confusion Matrix Analysis
🧪 Outputs Generated

The simulation produces the following artifacts:

Risk distribution visualisations
Confusion matrix heatmaps
ROC curves
Feature importance analysis
Model comparison metrics
CSV exports of predictions and decisions

All outputs are stored in:

/results
🛠 Technologies Used
Python
NumPy
Pandas
Scikit-learn
Matplotlib
Seaborn
🚀 Execution Instructions

Install dependencies:

pip install -r requirements.txt

Run simulation:

python simulator.py

Outputs generated:

Console evaluation metrics
/results directory containing graphs and evaluation files
📌 Research Contribution

This study contributes to the intersection of:

Zero Trust security architectures
Machine learning-based authentication systems
Risk-adaptive Identity and Access Management (IAM)

It demonstrates how combining:

behavioural analytics
probabilistic risk scoring
supervised machine learning
and policy-based decision systems

can significantly enhance authentication security decision-making in cloud environments.

👨‍💻 Author

Ramandeep Singh
Master of Information Technology
Whitecliffe
