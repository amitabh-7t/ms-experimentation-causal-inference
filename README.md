# AI Experimentation & Causal Uplift Modeling

> A production-ready framework for A/B testing, causal inference, and uplift modeling in AI SaaS experimentation

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview

This project demonstrates a complete end-to-end workflow for evaluating AI-powered features in a SaaS platform using advanced causal inference techniques. It combines classical A/B testing with modern machine learning approaches to estimate heterogeneous treatment effects and enable personalized feature rollouts.

**Key Capabilities**:
- 🎯 **Causal Effect Estimation**: Move beyond average treatment effects to individual-level predictions
- 📊 **Uplift Modeling**: Identify which users benefit most from AI features
- 🎨 **User Segmentation**: Create actionable segments based on treatment response
- 💼 **Business Insights**: Translate statistical findings into strategic recommendations

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/amitabh-7t/ms-experimentation-causal-inference.git
cd ms-experimentation-causal-inference

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Analysis

```bash
# Start Jupyter
jupyter notebook

# Open and run the comprehensive notebook
notebooks/00_complete_causal_inference_analysis.ipynb
```

## 📊 Main Notebook

### [00_complete_causal_inference_analysis.ipynb](notebooks/00_complete_causal_inference_analysis.ipynb)

**A comprehensive, production-ready notebook covering the entire analysis pipeline:**

#### Phase 1: Data Loading & Cleaning
- Load and validate 1.5M daily observations across 50k users
- Data quality checks and summary statistics
- Cohort distribution analysis

#### Phase 2: Classical A/B Testing
- Cohort-level statistical comparisons
- Pairwise t-tests with lift calculations
- Visualization of treatment effects
- **Key Finding**: 15-25% revenue lift with AI features

#### Phase 3: Causal Inference with X-learner
- Conditional Average Treatment Effect (CATE) estimation
- Feature engineering (continuous, binary, categorical)
- Individual-level uplift predictions
- **Key Finding**: Heterogeneous effects - not all users benefit equally

#### Phase 4: Feature Importance Analysis
- Identify drivers of treatment response
- Extract importance from meta-learner models
- **Top Drivers**: baseline_productivity, churn_risk, user_tenure

#### Phase 5: Uplift Segmentation
- Five-tier user segmentation (Very Low → Very High)
- Segment profiling and characterization
- **Business Strategy**: Targeted rollout recommendations per segment

#### Phase 6: Business Recommendations
- Executive summary with key findings
- Immediate actions and long-term strategy
- Expected business impact (revenue, retention, adoption)
- Limitations and next steps

**Output**: Production-ready analysis with visualizations, statistical tests, and actionable insights suitable for both technical and non-technical stakeholders.

## 🗂️ Project Structure

```
ms-experimentation-causal-inference/
├── notebooks/
│   ├── 00_complete_causal_inference_analysis.ipynb  # ⭐ Main comprehensive notebook
│   ├── 01_data_generation.ipynb                     # Synthetic data generation
│   ├── 02_eda.ipynb                                 # Exploratory analysis
│   ├── 03_ab_test_engine.ipynb                      # A/B testing framework
│   ├── 04_causal_inference.ipynb                    # Causal methods
│   ├── 05_uplift_model.ipynb                        # Uplift modeling
│   └── 06_business_report.ipynb                     # Business insights
├── src/
│   ├── ab_test.py                                   # A/B testing utilities
│   ├── causal.py                                    # Causal inference methods
│   ├── uplift_model.py                              # Uplift modeling
│   └── utils.py                                     # Helper functions
├── data/                                            # Generated datasets (gitignored)
├── requirements.txt                                 # Python dependencies
└── README.md
```

## 🔬 Methodology

### Experimental Design

**Treatment Cohorts**:
- **A_control**: Baseline (no AI features)
- **B_adaptive_v1**: First-generation adaptive AI
- **C_adaptive_v2**: Second-generation adaptive AI

**Dataset**:
- 50,000 users
- 30-day observation period
- 1.5M daily observations
- Rich feature set (demographics, behavior, confounders)

### Causal Inference Approach

**X-learner Meta-learner**:
1. Train separate models for treatment and control groups
2. Estimate counterfactual outcomes for each user
3. Compute individual treatment effects (CATE)
4. Combine predictions using propensity weighting

**Advantages**:
- Handles heterogeneous treatment effects
- Efficient with imbalanced groups
- Provides interpretable feature importance
- Enables personalized targeting

### Key Metrics

- **ai_calls**: AI feature usage intensity
- **tasks_completed**: Productivity measure
- **satisfaction_score**: User satisfaction (1-5 scale)
- **retention_7d**: 7-day retention rate
- **revenue**: Revenue per user

## 🛠️ Technology Stack

### Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| **pandas** | 2.3.3 | Data manipulation |
| **numpy** | 2.3.5 | Numerical computing |
| **scikit-learn** | 1.7.2 | Machine learning |
| **econml** | 0.15.5 | Causal inference (X-learner) |
| **statsmodels** | 0.14.5 | Statistical testing |
| **matplotlib** | 3.10.7 | Visualization |
| **seaborn** | 0.13.2 | Statistical plots |

### Additional Tools

- **causalml** (0.15.5): Alternative causal inference methods
- **xgboost** (3.1.2): Gradient boosting
- **lightgbm** (4.6.0): Gradient boosting
- **shap** (0.50.0): Model interpretability
- **pyarrow** (16.1.0): Parquet file support

## 📈 Key Results

### Treatment Effects

- ✅ **Statistically significant** improvements across all metrics (p < 0.001)
- 📈 **Revenue lift**: 15-25% depending on cohort
- 🔄 **Retention lift**: 10-20%
- 🎯 **C_adaptive_v2** outperforms B_adaptive_v1, validating iterative development

### Heterogeneous Effects

- 🎯 **Top 20% of users** show 3-5x higher uplift than average
- 📊 **Uplift range**: Near-zero to 100+ revenue points
- 🔍 **Key drivers**: Baseline productivity, churn risk, user tenure

### Business Impact

**Targeted Rollout Strategy**:
- **50%** of resources → Very High uplift segment (maximum ROI)
- **30%** of resources → High uplift segment
- **15%** of resources → Medium uplift segment
- **5%** of resources → Low/Very Low segments (focus on retention basics)

**Expected Outcomes**:
- 20-30% increase in incremental revenue vs. blanket rollout
- 5-10% reduction in churn among targeted users
- 2-3x higher AI feature adoption

## 🎯 Use Cases

This framework is applicable to:

- **Product Experimentation**: Evaluate new features with heterogeneous user bases
- **Personalization**: Identify which users benefit from specific treatments
- **Resource Allocation**: Optimize rollout strategies based on predicted uplift
- **Retention Programs**: Target at-risk users with high-impact interventions
- **Pricing Optimization**: Estimate willingness to pay across segments

## 📚 Learn More

### Causal Inference Resources

- [EconML Documentation](https://econml.azurewebsites.net/)
- Künzel et al. (2019): "Metalearners for estimating heterogeneous treatment effects"
- Athey & Imbens (2016): "Recursive partitioning for heterogeneous causal effects"

### Related Concepts

- **CATE**: Conditional Average Treatment Effect
- **Uplift Modeling**: Predicting individual treatment response
- **Meta-learners**: S-learner, T-learner, X-learner
- **Propensity Score**: Probability of treatment assignment

## 🔧 Troubleshooting

### Common Issues

**ArrowKeyError with Parquet files**:
```bash
pip install pyarrow==16.1.0
```

**Jupyter kernel issues**:
```bash
python -m ipykernel install --user --name=venv
```

**Missing dependencies**:
```bash
pip install -r requirements.txt --upgrade
```

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:

- [ ] Multi-treatment optimization (beyond binary treatment)
- [ ] Causal forests implementation
- [ ] Real-time uplift scoring API
- [ ] Additional meta-learner approaches
- [ ] Sensitivity analysis tools

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

**Author**: Amitabh  
**GitHub**: [@amitabh-7t](https://github.com/amitabh-7t)  
**Repository**: [ms-experimentation-causal-inference](https://github.com/amitabh-7t/ms-experimentation-causal-inference)

For questions or feedback, please open an issue on GitHub.

---

⭐ **Star this repository** if you find it useful for your experimentation workflows!
