"""
Script to deploy the Streamlit app to Hugging Face Spaces.
Called by the deploy-hosting job in the GitHub Actions workflow.
"""
import os
from huggingface_hub import HfApi, create_repo, login

login(token=os.environ.get("HF_TOKEN"))

api = HfApi()

HF_USERNAME = "AiRemastered"  # <-- CHANGE THIS to your HF username
SPACE_REPO = f"{HF_USERNAME}/tourism-package-app"

create_repo(
    repo_id=SPACE_REPO,
    repo_type="space",
    space_sdk="docker",
    exist_ok=True,
    private=False
)
print(f"Space ready: https://huggingface.co/spaces/{SPACE_REPO}")

for filename in ["Dockerfile", "app.py", "requirements.txt"]:
    api.upload_file(
        path_or_fileobj=f"deployment/{filename}",
        path_in_repo=filename,
        repo_id=SPACE_REPO,
        repo_type="space"
    )
    print(f"Uploaded: {filename}")

print(f"\nApp deployed at: https://huggingface.co/spaces/{SPACE_REPO}")
print("Note: Space may take 2-3 minutes to build.")
