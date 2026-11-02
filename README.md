###Telco Customer Churn Analysis & Predictive Modeling

This project analyzes customer churn using the Telco Customer Churn dataset, performs Exploratory Data Analysis (EDA), addresses class imbalance using SMOTE, and evaluates predictive models (Logistic Regression and XGBoost) to identify high-risk customers.

## Table of Contents
- [Project Overview](#project-overview)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Modeling Workflow](#modeling-workflow)
- [Model Evaluation & Results](#model-evaluation--results)
- [Key Insights & Recommendations](#key-insights--recommendations)

## Project Overview
Customer churn represents a major cost for telecom companies. This project focuses on:
- Cleaning and preparing raw customer demographics, services, and billing data.
- Analyzing key drivers of customer attrition.
- Training machine learning models to predict churn probabilities.

## Data Preprocessing
- **Missing Value Imputation**: Identified 11 missing values in `TotalCharges` corresponding to new customers (`tenure = 0`). Filled these missing values with `0` since they have not been billed yet.
- **Feature Scaling and Encoding**: Continuous variables (`tenure`, `MonthlyCharges`, `TotalCharges`) were standardized using `StandardScaler`. Categorical features were one-hot encoded using `OneHotEncoder`.
- **Class Imbalance**: The target variable `Churn` is imbalanced (~26.5% churn). We balanced the training dataset using **SMOTE** (Synthetic Minority Over-sampling Technique), raising the minority class size from 1,495 to 4,139 samples.

## Exploratory Data Analysis (EDA)
- **High-Risk Profiles**: Customers with month-to-month contracts, fiber-optic internet services, electronic check payment methods, and lack of online security or technical support exhibit significantly higher churn rates (~41% to 45%).
- **Numeric Drivers**: Shorter tenure and higher monthly charges strongly correlate with a higher probability of churn.

## Modeling Workflow
1. Split data into an 80/20 train/test split stratified by the target.
2. Built an end-to-end preprocessing pipeline via `ColumnTransformer`.
3. Applied SMOTE to training data to handle imbalance.
4. Trained and evaluated two classifiers:
   - **Logistic Regression** (baseline)
   - **XGBoost Classifier**

## Model Evaluation & Results

### ROC AUC Comparison
- **Logistic Regression**: **0.840**
- **XGBoost**: **0.821**

### XGBoost Classification Report (0.5 threshold)
- **Accuracy**: 78%
- **No Churn (Precision / Recall / F1)**: 0.85 / 0.85 / 0.85
- **Churn (Precision / Recall / F1)**: 0.59 / 0.59 / 0.59

### XGBoost Decision Threshold Tuning
- **t = 0.3**: Precision = 0.52, Recall = 0.72, F1 = 0.60
- **t = 0.4**: Precision = 0.55, Recall = 0.66, F1 = 0.60
- **t = 0.5**: Precision = 0.59, Recall = 0.59, F1 = 0.59
- **t = 0.6**: Precision = 0.63, Recall = 0.51, F1 = 0.56

*Note: Lowering the threshold to `0.3` or `0.4` increases churn recall (identifying more at-risk customers) at the expense of precision.*

## Key Insights & Recommendations
- **Target Month-to-Month Contracts**: Offer loyalty incentives, discounts, or paperless billing perks to migrate month-to-month customers to 1-year or 2-year contracts.
- **Optimize Support Services**: Promote technical support and online security add-ons to new fiber-optic customers to improve customer satisfaction and retention.
- **Proactive Outreach**: Use threshold-optimized probability predictions to offer proactive incentives to customers predicted as "high-risk" before they initiate cancellation.
