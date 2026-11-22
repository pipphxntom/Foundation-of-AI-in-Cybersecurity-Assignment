"""
sample_validation.py
Simple script to validate the environment setup.
Run this after installing requirements to confirm everything works.
"""

import numpy as np
import pandas as pd
import sklearn
import matplotlib
import seaborn as sns

print("Environment validation successful!")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"scikit-learn version: {sklearn.__version__}")
print(f"Matplotlib version: {matplotlib.__version__}")
print(f"Seaborn version: {sns.__version__}")

# Quick smoke test
df = pd.DataFrame({'a': np.random.randn(10), 'b': np.random.randn(10)})
print(f"\nSample DataFrame created with shape: {df.shape}")
print("All required packages are importable and functional.")
