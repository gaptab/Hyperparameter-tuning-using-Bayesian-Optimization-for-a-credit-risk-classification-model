# Hyperparameter-tuning-using-Bayesian-Optimization-for-a-credit-risk-classification-model

The objective of this project is to develop an application scorecard model using XGBoost for Personal Loan (PL) applications, predicting whether a loan applicant is likely to default.

**Step 1: Data Generation**

We create dummy data representing loan applicants with two sets of features:

1️⃣ Bureau-only Data: Includes details from credit bureau reports, such as:

Credit Score (higher is better)

Total Debt (amount of outstanding debt)

Debt-to-Income Ratio (financial health indicator)

Missed Payments (historical defaults)

2️⃣ Banking Data: Captures transaction behavior, including:

Average Account Balance

Number of Transactions

Maximum Withdrawal Amount

Overdraft Usage

A final column, Loan Default (1 = Default, 0 = No Default), is the target variable we want to predict.

**Step 2: Data Preprocessing**

Before training the models, we:

✅ Merge Bureau and Banking data using a unique Customer ID.

✅ Split features (X) and target variable (Loan Default - y).

✅ Scale numeric features using StandardScaler to improve model performance.

**Step 3: Model Training**

We develop two separate models using XGBoost, a popular gradient boosting algorithm:

📍 Bureau-Only Model:

Uses only credit bureau features to predict the likelihood of loan default.

Trained using 80% of the data and tested on 20%.

📍 Bureau + Banking Model:

Uses both credit bureau and banking features for improved risk assessment.

Captures financial transactions in addition to credit history.

More accurate compared to Bureau-Only Model since it has more information.

Both models are trained using XGBoost, which is optimized for classification problems and handles imbalanced data well.

**Step 4: Model Evaluation**

After training, we evaluate model performance using:

✅ AUC Score (Area Under the Curve) → Measures how well the model separates defaulters from non-defaulters.

✅ Classification Report → Shows metrics like precision, recall, and F1-score to assess model accuracy.

**Key Takeaways:**

Bureau + Banking Model performs better because it includes more detailed financial behavior data.

AUC Score helps compare the two models – a higher score means better predictions.
