# Titanic Survival Analysis

<img src="https://github.com/user-attachments/assets/c3fc49a4-9966-4384-9a8f-b668c799b292" alt="Dashboard Preview"/>

🔗 [View Interactive Tableau Dashboard](https://public.tableau.com/app/profile/elena.bolotin/viz/Book1_17786281389970/TitanicSurvivalAnalysisDashboard)

Machine learning project focused on predicting passenger survival on the Titanic dataset using exploratory data analysis, feature engineering, and classification models.

---

# Project Overview

This project analyzes passenger data from the Titanic disaster and builds machine learning models to predict survival outcomes.

The analysis explores how demographic characteristics, passenger class, fare price, and family relationships influenced survival probability.

---

# Business Problem

The goal of this project is to identify the key factors that influenced passenger survival and build predictive models capable of classifying survival outcomes.

This type of classification problem is commonly used in machine learning and predictive analytics workflows.

---

# Dataset

Source:
- Kaggle Titanic Dataset

Files used:
- train.csv
- test.csv

---

# Technologies & Tools

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

---

# Project Workflow

1. Data Cleaning
2. Exploratory Data Analysis (EDA)
3. Missing Value Handling
4. Feature Engineering
5. Data Preprocessing Pipeline
6. Model Training
7. Model Evaluation
8. Visualization & Insights

---

# Exploratory Data Analysis

The analysis explored:

- Survival rate by gender
- Survival rate by passenger class
- Age distribution by survival
- Fare distribution by class
- Survival by class and gender

Key findings:
- Female passengers had significantly higher survival rates
- First-class passengers survived more often
- Higher fares were strongly associated with survival
- Passenger class and gender were among the strongest predictors

---

# Models Used

## Logistic Regression
Used as a baseline classification model.

## Random Forest Classifier
Used as the final model for improved predictive performance.

---

# Model Performance

Random Forest achieved:

- Accuracy: ~81%
- ROC AUC: ~0.85

Evaluation methods:
- Classification Report
- Confusion Matrix
- ROC Curve
- Feature Importance Analysis

---

# Feature Importance

The most influential features included:
- Gender
- Fare
- Age
- Passenger Class

---

# Visualizations

The project includes:
- Survival distribution charts
- ROC Curve
- Confusion Matrix
- Feature Importance plots
- Passenger demographic analysis

---

# Future Improvements

Possible next steps:
- Hyperparameter tuning
- Cross-validation optimization
- XGBoost / CatBoost models
- Deployment with Streamlit
- Advanced feature engineering

---

# Key Insights

- Female passengers had a survival rate of ~74%, compared to ~19% for males.
- First-class passengers demonstrated the highest survival probability.
- Fare price strongly correlated with survival outcomes.
- Passenger class and gender were the most influential predictive features.

---
# Repository Structure

```text
Titanic-survival-analysis/
│
├── Titanic_competition_TensorFlow_Decision_Forests.ipynb
├── train.csv
├── test.csv
├── requirements.txt
├── app.py
├── templates/
├── static/
├── feature_importance_analysis.png
└── README.md
