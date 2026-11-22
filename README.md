# AI-Driven Cyber Threat Awareness and Detection Prototype

**Course:** AI in Cyber Security Lab (ENSP355)  
**Assignment:** End-to-End Mini Project – Foundations of AI in Cyber Security  
**Student:** Shwetank Pandey (2301420042)  
**University:** K.R. Mangalam University  

## Overview

This mini-project implements a basic AI-driven prototype for cyber threat awareness and detection. It simulates an entry-level cyber security analyst's workflow: analyzing threat patterns from login/authentication data, applying classical machine learning for classification of normal vs. suspicious events, prototyping a simple neural network, and reflecting on ethical challenges (false positives, bias, explainability).

The project uses a **synthetic but realistic login anomaly dataset** generated programmatically. This ensures full reproducibility without external dependencies while covering the required concepts.

## Repository Structure

```
ai-cybersecurity-unit1-mini-project-<yourname>/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── synthetic_login_threats.csv   # Generated dataset
├── src/
│   ├── generate_dataset.py           # Dataset generation script
│   └── threat_detection_pipeline.py  # Main analysis + ML + DL pipeline
├── notebooks/
│   └── cybersecurity_threat_prototype.ipynb  # Interactive notebook (recommended)
├── plots/                            # Generated visualizations (auto-created)
└── reports/
    └── technical_report.pdf          # Final report (to be generated/submitted)
```

## Setup Instructions

1. **Clone the repository** (or create as private repo named `ai-cybersecurity-unit1-mini-project-<yourname>`):
   ```bash
   git clone <your-repo-url>
   cd ai-cybersecurity-unit1-mini-project-<yourname>
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows: venv\Scripts\activate
   # On macOS/Linux: source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate the dataset** (optional – the pipeline can regenerate it):
   ```bash
   python src/generate_dataset.py
   ```

5. **Run the full pipeline**:
   ```bash
   python src/threat_detection_pipeline.py
   ```
   This generates plots in `plots/` and prints metrics.

6. **Interactive exploration** (recommended for submission):
   ```bash
   jupyter notebook notebooks/cybersecurity_threat_prototype.ipynb
   ```
   Or open in VS Code / JupyterLab.

## Tasks Covered

| Task | Implementation |
|------|----------------|
| 1. Workspace Bootstrap | This repository structure, venv, requirements |
| 2. Threat Landscape Analysis | EDA + visualizations of normal vs. suspicious patterns |
| 3. Basic ML Classification | Logistic Regression & Decision Tree with full metrics |
| 4. DL Prototype | scikit-learn `MLPClassifier` + comparison |
| 5. Ethical Reflection | Threshold analysis, false positive simulation, bias discussion |

## Key Results (run the code for exact numbers)

- Dataset: ~2,000 synthetic login events with engineered features (failed attempts, unusual hour, new device, location risk, etc.)
- Classical ML models achieve high accuracy on this controlled synthetic data
- Neural network provides comparable performance with slightly higher training time
- Threshold tuning demonstrates clear trade-off between false positives and detection rate

## Ethical Notes

AI in cyber security can block legitimate users (false positives), amplify bias present in training data, and suffer from limited explainability. This prototype deliberately explores these issues. See the notebook and technical report for detailed reflection.

## Academic Integrity

This is an individual assignment. All code and analysis should be understood and, where appropriate, customized. External references (scikit-learn docs, NIST framework concepts) are cited in the report.

## Support Resources Used

- scikit-learn documentation
- Concepts from NIST Cyber Security Framework (high-level)
- Foundational AI/ML lecture notes (Unit-1)

**Faculty:** Dr. Amar Saraswat  
**Deadline:** [To be announced]
