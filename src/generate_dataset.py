"""
generate_dataset.py
Generates a synthetic but realistic cybersecurity login/authentication dataset
for threat detection prototyping.

Features simulate common indicators of compromise:
- Unusual login hours
- High failed attempts
- New devices
- Location risk
- Session anomalies
- Department (for bias simulation in ethical analysis)
"""

import pandas as pd
import numpy as np
from pathlib import Path

def generate_synthetic_login_data(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic login events with normal and suspicious labels."""
    np.random.seed(random_state)
    
    # Departments for potential bias analysis
    departments = ['Engineering', 'Finance', 'HR', 'Sales', 'IT']
    dept_probs = [0.35, 0.20, 0.15, 0.20, 0.10]
    
    data = {
        'login_id': np.arange(1, n_samples + 1),
        'user_id': np.random.randint(1000, 9999, n_samples),
        'department': np.random.choice(departments, n_samples, p=dept_probs),
        'login_hour': np.random.randint(0, 24, n_samples),
        'is_weekend': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
        'num_failed_attempts': np.random.poisson(1.2, n_samples).clip(0, 15),
        'is_new_device': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'location_risk_score': np.random.beta(2, 5, n_samples),  # skewed toward low risk
        'session_duration_min': np.random.exponential(45, n_samples).clip(1, 300),
        'bytes_uploaded_kb': np.random.exponential(50, n_samples).clip(0, 2000),
        'concurrent_sessions': np.random.poisson(1.1, n_samples).clip(1, 8),
    }
    
    df = pd.DataFrame(data)
    
    # Stronger, more deterministic risk scoring for clearer learnable patterns
    risk_score = (
        (df['login_hour'].isin([0, 1, 2, 3, 4, 23])) * 1.2 +
        (df['num_failed_attempts'] >= 5) * 1.5 +
        (df['num_failed_attempts'] >= 3) * 0.6 +
        (df['is_new_device'] == 1) * 0.8 +
        (df['location_risk_score'] > 0.65) * 1.0 +
        (df['is_weekend'] == 1) * 0.3 +
        (df['bytes_uploaded_kb'] > 400) * 0.5 +
        (df['concurrent_sessions'] >= 3) * 0.7
    )
    
    # Mild department bias (Finance higher)
    risk_score += (df['department'] == 'Finance') * 0.4
    
    # Convert to probability with sharper threshold
    probability = 1 / (1 + np.exp(-(risk_score - 1.8) * 1.5))
    df['is_suspicious'] = (np.random.random(n_samples) < probability).astype(int)
    
    # Ensure reasonable class balance (~20-30% suspicious)
    suspicious_rate = df['is_suspicious'].mean()
    print(f"Generated dataset: {n_samples} samples")
    print(f"Suspicious rate: {suspicious_rate:.2%}")
    print(f"Class distribution:\n{df['is_suspicious'].value_counts()}")
    
    return df

def main():
    output_path = Path(__file__).parent.parent / "data" / "synthetic_login_threats.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = generate_synthetic_login_data(n_samples=2000)
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to: {output_path}")
    print(f"Shape: {df.shape}")
    print("\nSample rows:")
    print(df.head(8).to_string())
    print("\nFeature summary:")
    print(df.describe(include='all').round(2))

if __name__ == "__main__":
    main()
