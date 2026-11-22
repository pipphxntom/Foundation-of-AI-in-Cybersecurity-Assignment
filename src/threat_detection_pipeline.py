"""
threat_detection_pipeline.py
End-to-end AI-driven cyber threat detection prototype.

Covers:
- Dataset loading / generation
- Exploratory Data Analysis (EDA) and visualizations
- Preprocessing
- Classical ML models (Logistic Regression, Decision Tree)
- Simple Deep Learning prototype (MLPClassifier)
- Performance comparison
- Ethical analysis (threshold tuning, false positive impact, bias notes)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.pipeline import Pipeline

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "synthetic_login_threats.csv"
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


def load_or_generate_data():
    """Load existing dataset or generate if missing."""
    if DATA_PATH.exists():
        print(f"Loading dataset from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    else:
        print("Dataset not found. Generating synthetic data...")
        from generate_dataset import generate_synthetic_login_data
        df = generate_synthetic_login_data(2000)
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
    return df


def perform_eda(df: pd.DataFrame):
    """Exploratory data analysis and threat landscape visualizations."""
    print("\n" + "="*60)
    print("TASK 2: CYBER SECURITY THREAT LANDSCAPE ANALYSIS")
    print("="*60)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"\nTarget distribution:\n{df['is_suspicious'].value_counts(normalize=True).round(3)}")
    
    # 1. Suspicious rate by department
    plt.figure(figsize=(10, 5))
    dept_suspicious = df.groupby('department')['is_suspicious'].mean().sort_values(ascending=False)
    dept_suspicious.plot(kind='bar', color='steelblue', edgecolor='black')
    plt.title('Suspicious Login Rate by Department', fontsize=14, fontweight='bold')
    plt.ylabel('Proportion Suspicious')
    plt.xlabel('Department')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_suspicious_by_department.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Login hour distribution: normal vs suspicious
    plt.figure(figsize=(12, 5))
    for label, name, color in [(0, 'Normal', 'green'), (1, 'Suspicious', 'crimson')]:
        subset = df[df['is_suspicious'] == label]
        plt.hist(subset['login_hour'], bins=24, alpha=0.6, label=name, color=color, edgecolor='black')
    plt.title('Login Hour Distribution: Normal vs Suspicious', fontsize=14, fontweight='bold')
    plt.xlabel('Hour of Day')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_login_hour_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Failed attempts vs suspicious
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x='is_suspicious', y='num_failed_attempts', palette=['green', 'crimson'])
    plt.title('Failed Login Attempts: Normal vs Suspicious', fontsize=14, fontweight='bold')
    plt.xlabel('Is Suspicious (0=Normal, 1=Suspicious)')
    plt.ylabel('Number of Failed Attempts')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_failed_attempts_boxplot.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Correlation heatmap of numerical features
    numerical_cols = ['login_hour', 'num_failed_attempts', 'location_risk_score',
                      'session_duration_min', 'bytes_uploaded_kb', 'concurrent_sessions',
                      'is_new_device', 'is_weekend', 'is_suspicious']
    corr = df[numerical_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True)
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_correlation_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. Location risk distribution
    plt.figure(figsize=(10, 5))
    for label, name, color in [(0, 'Normal', 'green'), (1, 'Suspicious', 'crimson')]:
        subset = df[df['is_suspicious'] == label]
        sns.kdeplot(subset['location_risk_score'], label=name, color=color, fill=True, alpha=0.4)
    plt.title('Location Risk Score Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Location Risk Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "05_location_risk_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ EDA complete. Plots saved to {PLOTS_DIR}/")
    return df


def preprocess_data(df: pd.DataFrame):
    """Preprocessing pipeline: encode, scale, split."""
    print("\n" + "="*60)
    print("TASK 3: PREPROCESSING")
    print("="*60)
    
    # Drop identifiers
    X = df.drop(columns=['login_id', 'user_id', 'is_suspicious'])
    y = df['is_suspicious']
    
    # Encode categorical
    le = LabelEncoder()
    X['department_encoded'] = le.fit_transform(X['department'])
    X = X.drop(columns=['department'])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Features after encoding: {X_train.columns.tolist()}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, le, X.columns.tolist()


def train_and_evaluate_ml(X_train, X_test, y_train, y_test):
    """Train Logistic Regression and Decision Tree; return results."""
    print("\n" + "="*60)
    print("TASK 3: BASIC MACHINE LEARNING MODELS")
    print("="*60)
    
    results = {}
    
    # Logistic Regression
    print("\n--- Logistic Regression ---")
    start = time.time()
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr.fit(X_train, y_train)
    lr_time = time.time() - start
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    
    results['Logistic Regression'] = {
        'model': lr,
        'accuracy': accuracy_score(y_test, y_pred_lr),
        'precision': precision_score(y_test, y_pred_lr, zero_division=0),
        'recall': recall_score(y_test, y_pred_lr, zero_division=0),
        'f1': f1_score(y_test, y_pred_lr, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob_lr),
        'train_time': lr_time,
        'y_pred': y_pred_lr,
        'y_prob': y_prob_lr
    }
    print(f"Accuracy: {results['Logistic Regression']['accuracy']:.4f}")
    print(f"Precision: {results['Logistic Regression']['precision']:.4f}")
    print(f"Recall: {results['Logistic Regression']['recall']:.4f}")
    print(f"F1-Score: {results['Logistic Regression']['f1']:.4f}")
    print(f"Training time: {lr_time:.3f}s")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_lr, target_names=['Normal', 'Suspicious']))
    
    # Decision Tree
    print("\n--- Decision Tree ---")
    start = time.time()
    dt = DecisionTreeClassifier(max_depth=6, random_state=42, class_weight='balanced')
    dt.fit(X_train, y_train)
    dt_time = time.time() - start
    y_pred_dt = dt.predict(X_test)
    y_prob_dt = dt.predict_proba(X_test)[:, 1]
    
    results['Decision Tree'] = {
        'model': dt,
        'accuracy': accuracy_score(y_test, y_pred_dt),
        'precision': precision_score(y_test, y_pred_dt, zero_division=0),
        'recall': recall_score(y_test, y_pred_dt, zero_division=0),
        'f1': f1_score(y_test, y_pred_dt, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob_dt),
        'train_time': dt_time,
        'y_pred': y_pred_dt,
        'y_prob': y_prob_dt
    }
    print(f"Accuracy: {results['Decision Tree']['accuracy']:.4f}")
    print(f"Precision: {results['Decision Tree']['precision']:.4f}")
    print(f"Recall: {results['Decision Tree']['recall']:.4f}")
    print(f"F1-Score: {results['Decision Tree']['f1']:.4f}")
    print(f"Training time: {dt_time:.3f}s")
    
    return results


def train_and_evaluate_dl(X_train, X_test, y_train, y_test):
    """Simple neural network using MLPClassifier."""
    print("\n" + "="*60)
    print("TASK 4: DEEP LEARNING PROTOTYPE (MLPClassifier)")
    print("="*60)
    
    start = time.time()
    mlp = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=False,
        alpha=0.001
    )
    mlp.fit(X_train, y_train)
    mlp_time = time.time() - start
    
    y_pred_mlp = mlp.predict(X_test)
    y_prob_mlp = mlp.predict_proba(X_test)[:, 1]
    
    results = {
        'model': mlp,
        'accuracy': accuracy_score(y_test, y_pred_mlp),
        'precision': precision_score(y_test, y_pred_mlp, zero_division=0),
        'recall': recall_score(y_test, y_pred_mlp, zero_division=0),
        'f1': f1_score(y_test, y_pred_mlp, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob_mlp),
        'train_time': mlp_time,
        'y_pred': y_pred_mlp,
        'y_prob': y_prob_mlp,
        'n_iterations': mlp.n_iter_
    }
    
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1']:.4f}")
    print(f"Training time: {mlp_time:.3f}s")
    print(f"Iterations until convergence: {mlp.n_iter_}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_mlp, target_names=['Normal', 'Suspicious']))
    
    return results


def compare_models(ml_results, dl_results):
    """Side-by-side comparison of ML vs DL."""
    print("\n" + "="*60)
    print("MODEL COMPARISON: ML vs DL")
    print("="*60)
    
    comparison = []
    for name, res in ml_results.items():
        comparison.append({
            'Model': name,
            'Accuracy': res['accuracy'],
            'Precision': res['precision'],
            'Recall': res['recall'],
            'F1-Score': res['f1'],
            'ROC-AUC': res['roc_auc'],
            'Train Time (s)': res['train_time']
        })
    comparison.append({
        'Model': 'MLP Neural Network',
        'Accuracy': dl_results['accuracy'],
        'Precision': dl_results['precision'],
        'Recall': dl_results['recall'],
        'F1-Score': dl_results['f1'],
        'ROC-AUC': dl_results['roc_auc'],
        'Train Time (s)': dl_results['train_time']
    })
    
    comp_df = pd.DataFrame(comparison).round(4)
    print(comp_df.to_string(index=False))
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Metrics bar chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(metrics))
    width = 0.25
    for i, (name, res) in enumerate(list(ml_results.items()) + [('MLP', dl_results)]):
        values = [res['accuracy'], res['precision'], res['recall'], res['f1']]
        axes[0].bar(x + i*width, values, width, label=name)
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(metrics)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title('Model Performance Comparison', fontweight='bold')
    axes[0].legend()
    axes[0].set_ylabel('Score')
    
    # Training time
    times = [ml_results['Logistic Regression']['train_time'],
             ml_results['Decision Tree']['train_time'],
             dl_results['train_time']]
    names = ['Logistic\nRegression', 'Decision\nTree', 'MLP\nNeural Net']
    axes[1].bar(names, times, color=['steelblue', 'orange', 'purple'], edgecolor='black')
    axes[1].set_title('Training Time Comparison', fontweight='bold')
    axes[1].set_ylabel('Seconds')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "06_model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Comparison plot saved.")
    return comp_df


def ethical_analysis(ml_results, y_test, feature_names):
    """Ethical reflection: threshold analysis and false positive impact."""
    print("\n" + "="*60)
    print("TASK 5: ETHICAL ANALYSIS & THRESHOLD TUNING")
    print("="*60)
    
    # Use Logistic Regression probabilities for threshold analysis
    y_prob = ml_results['Logistic Regression']['y_prob']
    
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    threshold_results = []
    
    print("\nThreshold Sensitivity Analysis (Logistic Regression):")
    print("-" * 70)
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'FP Rate':>10} {'Blocked Users*':>15}")
    
    for thresh in thresholds:
        y_pred_thresh = (y_prob >= thresh).astype(int)
        prec = precision_score(y_test, y_pred_thresh, zero_division=0)
        rec = recall_score(y_test, y_pred_thresh, zero_division=0)
        f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
        
        # False positive rate among actual normals
        tn = ((y_test == 0) & (y_pred_thresh == 0)).sum()
        fp = ((y_test == 0) & (y_pred_thresh == 1)).sum()
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        blocked = int(fp)  # number of legitimate users blocked in test set
        
        threshold_results.append({
            'threshold': thresh, 'precision': prec, 'recall': rec,
            'f1': f1, 'fp_rate': fp_rate, 'blocked': blocked
        })
        print(f"{thresh:>10.1f} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f} {fp_rate:>10.3f} {blocked:>15}")
    
    # Plot threshold trade-off
    thresh_df = pd.DataFrame(threshold_results)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(thresh_df['threshold'], thresh_df['precision'], 'b-o', label='Precision', linewidth=2)
    ax1.plot(thresh_df['threshold'], thresh_df['recall'], 'r-s', label='Recall', linewidth=2)
    ax1.set_xlabel('Decision Threshold', fontsize=12)
    ax1.set_ylabel('Precision / Recall', fontsize=12)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='upper left')
    ax1.set_title('Threshold Tuning: Precision-Recall Trade-off\n(Higher threshold → fewer false positives, more missed threats)', 
                  fontsize=13, fontweight='bold')
    
    ax2 = ax1.twinx()
    ax2.bar(thresh_df['threshold'], thresh_df['blocked'], alpha=0.3, color='orange', width=0.05, label='Blocked Legitimate Users')
    ax2.set_ylabel('Legitimate Users Blocked (Test Set)', color='orange')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "07_threshold_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n" + "-"*70)
    print("ETHICAL RISKS & OBSERVATIONS")
    print("-"*70)
    print("""
1. FALSE POSITIVES BLOCK LEGITIMATE USERS
   - At default threshold 0.5, some normal users are flagged.
   - Raising the threshold reduces blocked users but increases missed attacks (false negatives).
   - In production, this creates a security vs. usability tension. Security teams must decide
     acceptable risk levels with stakeholders.

2. BIASED DATA → UNFAIR DECISIONS
   - The synthetic dataset intentionally includes a mild elevation in risk for the 'Finance' department.
   - If real data reflects historical patterns (e.g., certain roles or demographics flagged more),
     the model will amplify that bias.
   - Mitigation: fairness audits, balanced sampling, human review for high-impact decisions.

3. EXPLAINABILITY LIMITATIONS
   - Logistic Regression coefficients are interpretable (feature importance).
   - Decision Trees provide rule-based explanations.
   - Neural networks (MLP) are largely black-box; harder to justify blocking a user.
   - In cyber security, explainability is critical for analyst trust and regulatory compliance.

4. PRIVACY CONCERNS
   - Login data includes user identifiers, location signals, device fingerprints.
   - Models trained on such data must comply with data protection principles (minimization, purpose limitation).
   - Synthetic data avoids real privacy risk; real deployments require strong governance.

5. OPERATIONAL CHALLENGES
   - Concept drift: attacker behavior evolves; models need continuous monitoring and retraining.
   - Adversarial evasion: sophisticated attackers may craft inputs to evade detection.
   - Over-reliance on AI can reduce human vigilance if not designed as a decision-support tool.
""")
    
    print(f"\n✓ Threshold analysis plot saved to {PLOTS_DIR}/07_threshold_analysis.png")
    return thresh_df


def main():
    print("AI-DRIVEN CYBER THREAT DETECTION PROTOTYPE")
    print("Mini Project – AI in Cyber Security Lab")
    print("="*60)
    
    # Load data
    df = load_or_generate_data()
    
    # EDA
    df = perform_eda(df)
    
    # Preprocess
    X_train, X_test, y_train, y_test, scaler, le, feature_names = preprocess_data(df)
    
    # ML models
    ml_results = train_and_evaluate_ml(X_train, X_test, y_train, y_test)
    
    # DL model
    dl_results = train_and_evaluate_dl(X_train, X_test, y_train, y_test)
    
    # Comparison
    comparison = compare_models(ml_results, dl_results)
    
    # Ethical analysis
    ethical_analysis(ml_results, y_test, feature_names)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"All plots saved in: {PLOTS_DIR}")
    print("Next steps:")
    print("  1. Review plots and metrics")
    print("  2. Customize analysis if needed")
    print("  3. Generate technical report (PDF)")
    print("  4. Push to private GitHub repository")
    print("  5. Submit GitHub link + PDF report on LMS")


if __name__ == "__main__":
    main()
