🧠 Risk-Based Adaptive Authentication Framework
(Hybrid Machine Learning + Policy Engine using Kaggle RBA Dataset)
📌 Overview

This research presents a Risk-Based Adaptive Authentication Framework inspired by Zero Trust security principles for modern cloud-based Identity and Access Management (IAM) systems.

The framework simulates intelligent authentication decision-making by integrating three core components:

A probabilistic Risk Engine
A supervised Machine Learning classification model
A rule-based Policy Engine

These components work together to evaluate authentication requests using contextual and behavioural signals extracted from a real-world Kaggle Risk-Based Authentication (RBA) dataset.

The dataset contains real authentication log attributes such as:

Login timestamps
Device and browser information
IP address and geographic signals
Authentication success/failure labels
Attack indicators (e.g., account takeover, suspicious IP activity)

This project was developed as part of a Master of Information Technology (Research Project) at Whitecliffe.

🎯 Research Objectives

The study aims to:

Design and simulate a risk-based adaptive authentication system aligned with Zero Trust principles
Evaluate traditional IAM (rule-based) vs AI-driven risk-based authentication
Improve detection of malicious authentication attempts using behavioural analytics
Demonstrate the effectiveness of context-aware machine learning-based risk scoring
Propose a lightweight and deployable authentication decision framework for cloud environments
🏗 System Architecture

The proposed framework consists of three core modules:

1️⃣ Risk Engine

Computes probabilistic risk scores using contextual authentication features derived from the Kaggle RBA dataset.

Key input features:

Device trust status
IP risk level (encoded from dataset attributes)
Failed login attempts
Behavioural anomaly score (derived feature)
MFA usage indicator

The output is a calibrated probability:

R(X)=P(Attack∣X)

produced using a Random Forest classifier with probability calibration.

2️⃣ Machine Learning Engine

A supervised learning model (Random Forest Classifier) trained on the Kaggle RBA dataset to classify authentication events.

Output classes:

Legitimate
Attack

The model learns behavioural patterns directly from real-world authentication logs rather than synthetic simulation.

3️⃣ Policy Engine

Transforms the predicted risk score into authentication decisions:

ALLOW → Low-risk authentication
MFA_CHALLENGE → Medium-risk authentication requiring additional verification
DENY → High-risk or malicious authentication attempt

Decision-making is based on an adaptive percentile-based threshold derived from model predictions.

⚙️ Experimental Design

The study follows a simulation-based evaluation approach using real-world authentication data.

🔹 Dataset Description
Dataset: Kaggle Risk-Based Authentication (RBA) Dataset
Size: ~150,000 authentication records
Nature: Real-world inspired authentication logs
Type: Binary classification (Legitimate vs Attack)
🔹 Attack Behaviour Represented in Dataset
Account takeover attempts
Suspicious IP activity
Failed authentication patterns
Unusual login behaviour
🔹 Experimental Setup

Two systems are compared:

System	Description
Baseline IAM	Rule-based authentication using static thresholds
Risk-Based Model	Machine learning-driven adaptive authentication system
📊 Evaluation Metrics

The framework is evaluated using:

Accuracy
Precision
Recall
F1-score
ROC-AUC
False Positive Rate
False Negative Rate
Confusion Matrix
🧪 Outputs Generated

The simulation generates:

Risk score distributions
ROC curve analysis
Confusion matrix visualisations
Feature impact analysis
Model evaluation metrics
CSV export of predictions

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
📌 Output Artifacts

After execution:

Console prints evaluation metrics
/results folder contains:
ROC curves
Confusion matrices
Risk distribution plots
Final evaluation CSV files
📌 Research Contribution

This study contributes to the intersection of:

Zero Trust Security Architectures
Machine Learning-based Authentication Systems
Risk-Adaptive Identity and Access Management (IAM)

It demonstrates how combining:

Behavioural analytics
Probabilistic risk scoring
Supervised machine learning
Policy-based decision systems

can significantly improve authentication security in cloud environments.

👨‍💻 Author

Ramandeep Singh
Master of Information Technology
Whitecliffe (New Zealand)
