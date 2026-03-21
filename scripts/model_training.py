"""
Script to train models with MLflow tracking and register to HF Hub
"""
import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from huggingface_hub import HfApi, hf_hub_download, login
import warnings
warnings.filterwarnings('ignore')

# Login using HF_TOKEN environment variable
login(token=os.environ.get("HF_TOKEN"))

# Configuration
HF_USERNAME = "AiRemastered"  # Change this to your HF username
DATASET_REPO = f"{HF_USERNAME}/tourism-package-prediction"
MODEL_REPO = f"{HF_USERNAME}/tourism-package-model"

# Download data from HF Hub
train_path = hf_hub_download(repo_id=DATASET_REPO, filename="train.csv", repo_type="dataset")
test_path = hf_hub_download(repo_id=DATASET_REPO, filename="test.csv", repo_type="dataset")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# Prepare features
X_train = train_df.drop('ProdTaken', axis=1)
y_train = train_df['ProdTaken']
X_test = test_df.drop('ProdTaken', axis=1)
y_test = test_df['ProdTaken']

# Label encode categorical columns
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([X_train[col], X_test[col]], axis=0)
    le.fit(combined)
    X_train[col] = le.transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    label_encoders[col] = le

# Set MLflow experiment
mlflow.set_experiment("Tourism_Package_Prediction")

# Define models
models = {
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {"n_estimators": [50, 100], "max_depth": [5, 10]}
    },
    "XGBoost": {
        "model": XGBClassifier(random_state=42, eval_metric='logloss'),
        "params": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1], "max_depth": [3, 5]}
    }
}

# Train models
best_model = None
best_model_name = None
best_accuracy = 0

for model_name, config in models.items():
    print(f"\nTraining {model_name}...")
    with mlflow.start_run(run_name=model_name):
        grid_search = GridSearchCV(config["model"], config["params"], cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_estimator = grid_search.best_estimator_
        y_pred = best_estimator.predict(X_test)
        y_pred_proba = best_estimator.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
        mlflow.sklearn.log_model(best_estimator, model_name)
        
        print(f"{model_name} Accuracy: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = best_estimator
            best_model_name = model_name

print(f"\nBest Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")

# Save model locally
os.makedirs("model_building", exist_ok=True)
joblib.dump(best_model, "model_building/best_model.pkl")
joblib.dump(label_encoders, "model_building/label_encoders.pkl")
joblib.dump(X_train.columns.tolist(), "model_building/feature_columns.pkl")

# Upload to HF Model Hub
api = HfApi()
api.create_repo(repo_id=MODEL_REPO, repo_type="model", exist_ok=True)
api.upload_file(path_or_fileobj="model_building/best_model.pkl", path_in_repo="best_model.pkl", repo_id=MODEL_REPO, repo_type="model")
api.upload_file(path_or_fileobj="model_building/label_encoders.pkl", path_in_repo="label_encoders.pkl", repo_id=MODEL_REPO, repo_type="model")
api.upload_file(path_or_fileobj="model_building/feature_columns.pkl", path_in_repo="feature_columns.pkl", repo_id=MODEL_REPO, repo_type="model")

print(f"Model registered at: https://huggingface.co/{MODEL_REPO}")
