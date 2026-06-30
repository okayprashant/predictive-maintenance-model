# Predictive Maintenance System
## Machine Failure Prediction Using Industrial Sensor Data

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

This project presents an end-to-end Predictive Maintenance System developed using the AI4I 2020 Predictive Maintenance Dataset. The objective is to predict machine failures before they occur by analyzing sensor data collected during machine operation.

The project follows a complete data science workflow including data cleaning, exploratory data analysis (EDA), feature engineering, machine learning model development, evaluation, and explainable AI using SHAP.

The final model enables maintenance teams to identify high-risk machines early, helping reduce unexpected downtime and optimize maintenance schedules.

**🎯 Problem Statement**

Unexpected machine failures can lead to production downtime, increased maintenance costs, and operational inefficiencies.

The objective of this project is to build a machine learning model capable of accurately predicting machine failures based on operational sensor data while providing interpretable explanations for each prediction.


**Business Impact:**
- 🎯 **35.29% failure detection rate** at 90% precision (Random Forest)
- 🔴 **Reduces false alarms** by 22.35 percentage points vs. baseline
- ⚙️ Enables **proactive maintenance planning** instead of reactive repairs

---

## 📊 Project Workflow

This project is organized into **3 modular notebooks** for reproducibility and clarity:

| # | Notebook | Purpose |
|---|----------|---------|
| **1** | `01_Data_Cleaning_EDA.ipynb` | Data loading, quality checks, exploratory analysis, leakage detection |
| **2** | `02_Predictive_Maintenance.ipynb` | Feature engineering, train/test split, model training & evaluation |
| **3** | `03_Model_Interpretation.ipynb` | Feature importance, SHAP analysis, business operating points |

Each notebook is **self-contained** and can be run independently.

---

## 🎯 Key Results

| Model | PR-AUC | ROC-AUC | Recall @ 90% Precision | Threshold |
|-------|--------|---------|------------------------|-----------|
| **Random Forest** | 0.7609 | 0.9664 | **35.29%** | 0.58 |
| **Calibrated Logistic** | 0.3793 | 0.8833 | 12.94% | 0.6475 |

**Top 5 Most Important Features (Random Forest):**
1. **Torque [Nm]** — 31.58%
2. **Rotational speed [rpm]** — 29.31%
3. **Tool wear [min]** — 20.06%
4. **Air temperature [K]** — 9.93%
5. **Process temperature [K]** — 7.00%

## 🔍 Explainable AI

To improve transparency, SHAP (SHapley Additive Explanations) was used to interpret model predictions.

The explainability notebook includes:

SHAP Summary Plot
SHAP Feature Importance
SHAP Waterfall Plot
SHAP Force Plot
Global Model Interpretation
Local Prediction Explanations

## 💡 Key Insights

1. The dataset is highly imbalanced, making PR-AUC a more informative metric than accuracy.
2. Torque and Tool Wear are the most influential predictors of machine failure.
3. Rotational Speed significantly contributes to failure prediction under certain operating conditions.
4. Temperature variables provide supporting information but have comparatively lower predictive importance.
5. Explainability confirms that multiple sensor interactions influence machine failure rather than any single feature.


### Visualizations
![PR Curve Comparison](assets/pr_curve_comparison.png)
![RF Feature Importances](assets/feature_importances_rf.png)

---

## 📂 Project Structure

```
predictive-maintenance-model/
├── notebook/
│   ├── 01_Data_Cleaning_EDA.ipynb
│   ├── 02_Predictive_Maintenance.ipynb
│   └── 03_Model_Interpretation.ipynb
├── data/
│   └── ai4i2020.csv                 # Raw data (download from UCI ML)
│   └── ai4i2020_clean.csv
├── models/
│   ├── random_forest_model.pkl
│   ├── logistic_model.pkl 
├── assets/
│   ├── pr_curve_comparison.png
│   ├── feature_importances_rf.png
│   ├── metrics_summary.csv
│   └── operating_point_uplift.csv
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/okayprashant/predictive-maintenance-model.git
cd predictive-maintenance-model

# (Optional) Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Setup Data

```bash
# Download dataset
# Visit: https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset
# Download ai4i2020.csv and place it in the data/ folder

mkdir -p data
# Place ai4i2020.csv here
```

### Run the Analysis

```bash
# Start Jupyter
jupyter notebook

# Open and run notebooks in order:
# 1. notebook/01_Data_Cleaning_EDA.ipynb
# 2. notebook/02_Predictive_Maintenance.ipynb
# 3. notebook/03_Model_Interpretation.ipynb
```

All outputs are saved to `assets/` automatically.

---

## 📋 Dataset Overview

**Source:** [UCI Machine Learning Repository – AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset)

- **Rows:** ~10,000 machine operations
- **Columns:** 14 features (operational parameters + maintenance flags)
- **Target:** Binary `Machine failure` flag

### Failure Types
| Code | Type | Description |
|------|------|-------------|
| TWF | Tool Wear Failure | Excessive tool wear |
| HDF | Heat Dissipation | Overheating/poor cooling |
| PWF | Power Failure | Electrical/power system |
| OSF | Overstrain Failure | Mechanical stress |
| RNF | Random Failure | Unclassified events |

---

## 🔧 Methodology

### 1. **Data Cleaning & EDA** (`01_Data_Cleaning_EDA.ipynb`)
- ✅ Quality checks and missing value handling
- ✅ Label consistency verification (failure subtypes vs. target)
- ✅ Class imbalance analysis
- ✅ Feature distributions and correlations
- ✅ Leakage detection (subtype flags removed before modeling)

### 2. **Model Development** (`02_Predictive_Maintenance.ipynb`)
- **Feature Engineering:** Extract product category, handle categorical variables
- **Preprocessing:** Stratified train/test split, standardization, one-hot encoding
- **Models Trained:**
  - Random Forest (non-linear, strong baseline)
  - Calibrated Logistic Regression (interpretable, probability-calibrated)
- **Evaluation:** PR-AUC, ROC-AUC, Recall @ Fixed Precision

### 3. **Interpretation & Insights** (`03_Model_Interpretation.ipynb`)
-  SHAP feature importance analysis
-  Operating point selection (precision-recall trade-offs)
-  Business recommendations for maintenance scheduling
-  Model comparison and uplift quantification

---

##  Dependencies

See `requirements.txt` for full list:
- **Data:** pandas, numpy
- **ML:** scikit-learn
- **Visualization:** matplotlib, seaborn
- **Interpretation:** SHAP

```bash
pip install -r requirements.txt
```

---

## 🎯 Future Improvements
1. Hyperparameter Optimization
2. XGBoost and LightGBM Implementation
3. Real-Time Prediction Dashboard
4. Cloud Deployment
5. Live Sensor Integration
6. Automated Maintenance Alerts

---

##  Contributing

Contributions are welcome! For major changes:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add improvement'`)
4. Push and open a pull request

---

##  License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

##  Contact & Questions

For questions or suggestions:
- Open an [Issue](https://github.com/okayprashant/predictive-maintenance-model/issues)
- Reach out via GitHub Discussions

---

##  References

- **Dataset:** [AI4I 2020 Predictive Maintenance Dataset (UCI ML)](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset)
- **Paper:** Matzka, S. (2020). *Explainable Artificial Intelligence for Predictive Maintenance Applications*, 2020.
- **SHAP:** [SHapley Additive exPlanations](https://github.com/shap/shap)
