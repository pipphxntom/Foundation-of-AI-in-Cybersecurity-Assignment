# Technical Report: AI-Driven Cyber Threat Awareness and Detection Prototype

**Course:** AI in Cyber Security Lab (ENSP355)  
**Assignment Title:** End-to-End Mini Project  
**Student Name:** [Your Name]  
**Student ID:** [Your ID]  
**Date:** [Submission Date]  
**Faculty:** Dr. Amar Saraswat  

---

## 1. Problem Overview

Modern organizations face adaptive cyber threats (phishing, credential stuffing, anomalous access, insider risk) that static rule-based systems struggle to detect. This mini-project simulates the work of an entry-level cyber security analyst by building a practical AI-enabled prototype that:

- Analyzes patterns in authentication/login data
- Applies foundational machine learning to classify events as normal or suspicious
- Prototypes a simple neural network
- Examines ethical and operational challenges

A synthetic login anomaly dataset was generated to ensure full reproducibility while covering realistic threat indicators (unusual hours, failed attempts, new devices, location risk, concurrent sessions, etc.).

## 2. Methodology

### 2.1 Dataset
- 2,000 synthetic login events
- Features: login hour, weekend flag, failed attempts, new device, location risk score, session duration, bytes uploaded, concurrent sessions, department
- Target: binary label `is_suspicious` (~21% positive class after strengthening the risk signal)
- Mild intentional bias injected for the Finance department to support ethical discussion

### 2.2 Exploratory Data Analysis
- Class distribution and department-wise suspicious rates
- Login hour histograms (normal vs suspicious)
- Box plots of failed attempts
- Correlation heatmap of numerical features
- Location risk density plots

### 2.3 Preprocessing
- Dropped identifiers (login_id, user_id)
- Label-encoded department
- Stratified train-test split (75/25)
- StandardScaler on numerical features

### 2.4 Models
- **Logistic Regression** (class_weight='balanced')
- **Decision Tree** (max_depth=6, class_weight='balanced')
- **MLPClassifier** (hidden layers 32→16, ReLU, Adam)

### 2.5 Evaluation Metrics
Accuracy, Precision, Recall, F1-score, ROC-AUC, training time.

### 2.6 Ethical Analysis
Threshold sensitivity analysis on Logistic Regression probabilities to quantify the trade-off between false positives (blocked legitimate users) and detection rate.

## 3. Results and Observations

### 3.1 Key Visualizations
(Insert the generated plots from the `plots/` folder or notebook outputs here)

- Suspicious rate varies by department
- Suspicious logins concentrate in off-hours
- Higher failed attempts strongly associated with the suspicious class
- Clear feature correlations with the target

### 3.2 Model Performance (approximate results from pipeline)

| Model                  | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Train Time |
|------------------------|----------|-----------|--------|----------|---------|------------|
| Logistic Regression    | ~0.68    | ~0.34     | ~0.57  | ~0.43    | ~0.70   | <0.01 s    |
| Decision Tree          | ~0.77    | ~0.45     | ~0.52  | ~0.48    | ~0.68   | <0.01 s    |
| MLP Neural Network     | ~0.77    | ~0.43     | ~0.29  | ~0.35    | ~0.66   | ~1.4 s     |

**Observations:**
- Decision Tree achieved the best balance of accuracy and F1 on this dataset.
- Logistic Regression offered higher recall (better at catching threats) at the cost of more false positives.
- The simple MLP matched Decision Tree accuracy but required significantly more training time and provided lower recall and no inherent explainability.
- All models substantially beat a majority-class baseline, confirming that the engineered features contain learnable signal.

### 3.3 Threshold Analysis
Raising the decision threshold from 0.5 to 0.6 dramatically reduced false positives (blocked legitimate users) while accepting a drop in recall. This illustrates a core operational trade-off.

## 4. Ethical Discussion

1. **False Positives Blocking Legitimate Users**  
   Aggressive thresholds or poorly calibrated models can lock out employees, disrupting business. Security teams must jointly define acceptable false positive rates with business stakeholders.

2. **Bias and Unfair Decisions**  
   The synthetic data contains a mild elevation in risk for the Finance department. In real data, historical biases (by role, geography, time of day patterns of certain teams, etc.) can be amplified. Mitigation requires fairness audits, balanced sampling, and human review for high-impact actions.

3. **Explainability Limitations**  
   Logistic Regression coefficients and Decision Tree rules are relatively transparent. Neural networks are largely black-box. In cyber security, analysts need to understand *why* an alert was raised to investigate effectively and to satisfy compliance requirements.

4. **Privacy Concerns**  
   Authentication logs contain sensitive behavioral and contextual data. This prototype uses only synthetic data. Real deployments must follow data minimization, purpose limitation, and strong access controls.

5. **Operational Challenges**  
   - Concept drift: attacker techniques evolve; models require continuous monitoring and retraining.  
   - Adversarial evasion: sophisticated actors may deliberately craft inputs to stay below detection thresholds.  
   - Over-reliance risk: AI should augment, not replace, human analysts.

## 5. Conclusion

This mini-project successfully demonstrates a complete foundational pipeline from raw (synthetic) authentication data to an evaluated AI prototype, while explicitly addressing ethical and operational realities. The work provides a practical base for more advanced techniques (ensemble methods, sequence models on full session logs, anomaly detection without labels, etc.) covered in later units.

## 6. References
- scikit-learn documentation (LogisticRegression, DecisionTreeClassifier, MLPClassifier, metrics)
- NIST Cybersecurity Framework (high-level concepts of detection and response)
- Introductory AI & ML lecture notes (Unit-1)

---

*Note: Replace approximate numbers with exact values from your run of the code. Insert actual plot screenshots. Customize language so the report reflects your understanding.*
