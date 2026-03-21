"""
Script to prepare data: clean, split, and upload to HF Hub
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, hf_hub_download, login

# Login using HF_TOKEN environment variable
login(token=os.environ.get("HF_TOKEN"))

# Configuration
HF_USERNAME = "AiRemastered"  # Change this to your HF username
DATASET_REPO = f"{HF_USERNAME}/tourism-package-prediction"

# Download dataset from HF Hub
file_path = hf_hub_download(repo_id=DATASET_REPO, filename="tourism.csv", repo_type="dataset")
df = pd.read_csv(file_path)
print(f"Loaded dataset with shape: {df.shape}")

# Data Cleaning
df_cleaned = df.copy()

# Remove unnecessary columns
columns_to_drop = ['CustomerID']
if 'Unnamed: 0' in df_cleaned.columns:
    columns_to_drop.append('Unnamed: 0')
df_cleaned = df_cleaned.drop(columns=columns_to_drop)

# Fix Gender inconsistencies
df_cleaned['Gender'] = df_cleaned['Gender'].replace('Fe Male', 'Female')

# Handle missing values
numerical_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_cols:
    if df_cleaned[col].isnull().sum() > 0:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df_cleaned[col].isnull().sum() > 0:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])

print(f"Cleaned dataset shape: {df_cleaned.shape}")

# Split data
train_df, test_df = train_test_split(
    df_cleaned, test_size=0.2, random_state=42, stratify=df_cleaned['ProdTaken']
)
print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

# Save locally
os.makedirs("data", exist_ok=True)
train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

# Upload to HF Hub
api = HfApi()
api.upload_file(path_or_fileobj="data/train.csv", path_in_repo="train.csv", repo_id=DATASET_REPO, repo_type="dataset")
api.upload_file(path_or_fileobj="data/test.csv", path_in_repo="test.csv", repo_id=DATASET_REPO, repo_type="dataset")

print("Train and test datasets uploaded to HF Hub!")
