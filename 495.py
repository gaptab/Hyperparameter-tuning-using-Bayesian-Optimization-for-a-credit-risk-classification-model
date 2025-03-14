import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from bayes_opt import BayesianOptimization
import joblib

# 🚀 Step 1: Generate Dummy Data (Simulating Credit Risk Data)
np.random.seed(42)
n_samples = 5000

df = pd.DataFrame({
    'credit_score': np.random.randint(300, 850, n_samples),
    'income': np.random.randint(20000, 120000, n_samples),
    'loan_amount': np.random.randint(5000, 50000, n_samples),
    'loan_term': np.random.choice([12, 24, 36, 48, 60], n_samples),
    'interest_rate': np.random.uniform(5, 20, n_samples),
    'debt_to_income': np.random.uniform(0.1, 0.9, n_samples),
    'num_accounts': np.random.randint(1, 10, n_samples),
    'missed_payments': np.random.randint(0, 5, n_samples),
    'loan_default': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # 30% default rate
})

# 🚀 Step 2: Data Preprocessing
X = df.drop(columns=['loan_default'])
y = df['loan_default']

# 🚀 Step 3: Train-Test Split (Including OOT Dataset)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_valid, X_oot, y_valid, y_oot = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 🚀 Step 4: Define Bayesian Optimization Function for XGBoost
def xgb_booster(learning_rate, max_depth, subsample, colsample_bytree):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': learning_rate,
        'max_depth': int(max_depth),
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'random_state': 42
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    
    model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dvalid, 'valid')], early_stopping_rounds=10, verbose_eval=False)
    
    preds = model.predict(dvalid)
    auc_score = roc_auc_score(y_valid, preds)
    return auc_score

# 🚀 Step 5: Perform Bayesian Optimization
optimizer = BayesianOptimization(
    f=xgb_booster,
    pbounds={
        'learning_rate': (0.01, 0.3),
        'max_depth': (3, 10),
        'subsample': (0.5, 1.0),
        'colsample_bytree': (0.5, 1.0)
    },
    random_state=42
)
optimizer.maximize(init_points=5, n_iter=15)

# 🚀 Step 6: Train XGBoost Model with Optimized Hyperparameters
best_params = optimizer.max['params']
best_params['max_depth'] = int(best_params['max_depth'])  # Convert max_depth to int

final_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    **best_params,
    'random_state': 42
}

dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)
dtest = xgb.DMatrix(X_oot, label=y_oot)

final_model = xgb.train(final_params, dtrain, num_boost_round=200, evals=[(dvalid, 'valid')], early_stopping_rounds=10, verbose_eval=False)

# 🚀 Step 7: Evaluate Model on OOT Dataset (Calculate Gini Coefficient)
preds_oot = final_model.predict(dtest)
auc_oot = roc_auc_score(y_oot, preds_oot)
gini_oot = 2 * auc_oot - 1  # Gini Coefficient Formula

print(f"AUC on OOT Dataset: {auc_oot:.4f}")
print(f"Gini Coefficient on OOT Dataset: {gini_oot:.4f}")

# 🚀 Step 8: Save Data and Model
df.to_csv("credit_risk_data.csv", index=False)
joblib.dump(final_model, "xgboost_credit_risk_model.pkl")
