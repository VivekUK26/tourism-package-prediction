"""
Script to register the dataset on Hugging Face Hub
"""
import os
from huggingface_hub import HfApi, login

# Login using HF_TOKEN environment variable
login(token=os.environ.get("HF_TOKEN"))

# Initialize API
api = HfApi()

# Configuration
HF_USERNAME = "AiRemastered"  
DATASET_REPO = f"{HF_USERNAME}/tourism-package-prediction"

# Create dataset repository
api.create_repo(repo_id=DATASET_REPO, repo_type="dataset", exist_ok=True)

# Upload the dataset
api.upload_file(
    path_or_fileobj="data/tourism.csv",
    path_in_repo="tourism.csv",
    repo_id=DATASET_REPO,
    repo_type="dataset"
)

print(f"Dataset uploaded to: https://huggingface.co/datasets/{DATASET_REPO}")
