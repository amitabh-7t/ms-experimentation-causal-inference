# MS Experimentation & Causal Inference

A comprehensive framework for A/B testing, causal inference, and uplift modeling in the context of AI SaaS experimentation. This project demonstrates end-to-end experimentation workflows from synthetic data generation to business insights.

## 📋 Overview

This project provides a complete pipeline for:
- **Synthetic Data Generation**: Realistic AI SaaS user behavior simulation
- **Exploratory Data Analysis**: Understanding user patterns and treatment effects
- **A/B Testing**: Statistical hypothesis testing for treatment effectiveness
- **Causal Inference**: Estimating true causal effects using advanced methods
- **Uplift Modeling**: Predicting individual treatment effects
- **Business Reporting**: Translating statistical findings into actionable insights

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ms-experimentation-causal-inference.git
cd ms-experimentation-causal-inference
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Project Structure

```
├── notebooks/           # Jupyter notebooks for analysis
│   ├── 01_data_generation.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_ab_test_engine.ipynb
│   ├── 04_causal_inference.ipynb
│   ├── 05_uplift_model.ipynb
│   └── 06_business_report.ipynb
├── src/                # Source code modules
│   ├── ab_test.py
│   ├── causal.py
│   ├── uplift_model.py
│   └── utils.py
├── data/               # Generated datasets (gitignored)
├── requirements.txt    # Python dependencies
└── README.md
```

## 🔬 Notebooks

### 01. Data Generation
Generates synthetic AI SaaS experimentation data with:
- 50,000 users across 30 days
- Multiple treatment cohorts (control, adaptive_v1, adaptive_v2)
- Realistic user attributes and behavioral metrics
- Confounding variables for causal analysis

### 02. Exploratory Data Analysis
- Distribution analysis of key metrics
- Treatment group comparisons
- Correlation analysis
- Data quality checks

### 03. A/B Test Engine
Statistical testing framework including:
- T-tests for continuous metrics
- Chi-square tests for categorical metrics
- Multiple testing corrections
- Statistical power analysis

### 04. Causal Inference
Advanced causal methods:
- Propensity Score Matching
- Inverse Probability Weighting
- Doubly Robust Estimation
- Treatment effect heterogeneity

### 05. Uplift Modeling
Machine learning for personalized treatment effects:
- Meta-learner approaches (S-learner, T-learner, X-learner)
- Feature importance for treatment response
- Individual treatment effect predictions

### 06. Business Report
Translating analysis into business insights:
- Executive summaries
- ROI calculations
- Recommendations for product teams

## 🛠️ Key Dependencies

- **pandas** (2.3.3): Data manipulation
- **numpy** (2.3.5): Numerical computing
- **scikit-learn** (1.7.2): Machine learning
- **statsmodels** (0.14.5): Statistical modeling
- **causalml** (0.15.5): Causal inference
- **xgboost** (3.1.2): Gradient boosting
- **lightgbm** (4.6.0): Gradient boosting
- **shap** (0.50.0): Model interpretability
- **matplotlib** (3.10.7): Visualization
- **seaborn** (0.13.2): Statistical visualization
- **pyarrow** (16.1.0): Parquet file support

## 📈 Usage

1. **Generate Data**:
   ```bash
   jupyter notebook notebooks/01_data_generation.ipynb
   ```
   Run all cells to generate synthetic datasets in the `data/` directory.

2. **Run Analysis**:
   Execute notebooks sequentially (02-06) to perform the complete analysis pipeline.

3. **Custom Analysis**:
   Import modules from `src/` for custom experimentation workflows.

## 🔧 Troubleshooting

### ArrowKeyError with Parquet Files
If you encounter `ArrowKeyError: No type extension with name arrow.py_extension_type found`, ensure you have the correct pyarrow version:
```bash
pip install pyarrow==16.1.0
```

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.
