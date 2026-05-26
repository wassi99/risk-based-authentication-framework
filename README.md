🧠 Risk-Based Adaptive Authentication Framework
(Hybrid Machine Learning + Policy Engine Simulation)
📌 Overview

This project presents a Risk-Based Adaptive Authentication Framework inspired by Zero Trust security principles for cloud-based environments.

The framework simulates intelligent authentication decision-making using a combination of:

A probabilistic Risk Engine
A Machine Learning-based classification model
A Policy Engine for access decisions

It evaluates authentication requests dynamically using contextual and behavioural signals such as:

Device trust level
IP reputation score
Failed login attempts
Behavioural anomaly score
MFA usage

This research was developed as part of a Master of Information Technology (Research Project) at Whitecliffe, New Zealand.

🎯 Research Objectives
To design and simulate a risk-based adaptive authentication system
To compare traditional IAM vs AI-driven risk-based authentication
To evaluate the effectiveness of context-aware risk scoring
To improve detection of malicious authentication attempts
To propose a lightweight, deployable security decision framework
🏗 System Architecture

The framework consists of three core modules:

1️⃣ Risk Engine

Computes probabilistic risk scores based on contextual signals:

IP risk level
Device trust status
Behavioural deviation
MFA usage
Login attempt patterns
2️⃣ Machine Learning Engine

A supervised learning model (Random Forest Classifier) that learns attack patterns from synthetic authentication data.

3️⃣ Policy Engine

Transforms risk probabilities into actionable decisions:

ALLOW
MFA_CHALLENGE
DENY
⚙️ Experiment Design

The study follows a controlled simulation-based experimental design:

🔹 Dataset Generation
2000 synthetic authentication events
150 simulated user profiles
Attack injection (15% malicious traffic)
🔹 Experimental Setup

Two systems are compared:

System	Description
Baseline IAM	Rule-based authentication using static thresholds
Risk-Based Model	ML-based adaptive authentication system
🔹 Evaluation Strategy

Both systems are evaluated using:

Precision
Recall
F1-score
ROC-AUC
Confusion Matrix Analysis
📊 Evaluation Metrics
Accuracy
Precision
Recall
F1-score
False Positive Rate
False Negative Rate
ROC-AUC
🧪 Output & Results

The simulator generates:

Risk distribution plots
Confusion matrices
ROC curves
Feature importance charts
Model comparison tables
CSV export of predictions

All outputs are saved in:

/results
🛠 Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
🚀 How to Run
1. Install dependencies
pip install -r requirements.txt
2. Run simulator
python simulator.py
3. Outputs generated
Console evaluation metrics
/results folder containing graphs
synthetic_authentication_dataset.csv
results/model_results.csv
📌 Research Contribution

This work bridges the gap between:

Theoretical Zero Trust security models
Practical risk-based authentication systems
Lightweight machine learning-based IAM frameworks

It demonstrates how adaptive risk scoring + ML classification + policy enforcement can improve authentication security decisions.

👨‍💻 Author

Ramandeep Singh
Master of Information Technology
Whitecliffe, New Zealand
