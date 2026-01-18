#!/usr/bin/env python3
"""
Deployment Script for Engine Predictive Maintenance
"""

from huggingface_hub import HfApi
import os

def deploy_to_huggingface_space():
    """Deploy application to HuggingFace Spaces"""
    print("Deploying to HuggingFace Spaces...")

    try:
        HF_TOKEN = os.getenv('HF_TOKEN')
        HF_USERNAME_OR_ORG = os.getenv('HF_USERNAME_OR_ORG')

        if not HF_TOKEN or not HF_USERNAME_OR_ORG:
            raise ValueError("HF_TOKEN and HF_USERNAME_OR_ORG environment variables required")

        api = HfApi()
        space_id = f"{HF_USERNAME_OR_ORG}/engine-predictive-maintenance-app"

        # Create space
        try:
            api.create_repo(
                repo_id=space_id,
                repo_type="space",
                space_sdk="streamlit",
                exist_ok=True,
                private=False,
                token=HF_TOKEN
            )
            print(f"✅ Space created/verified: {space_id}")
        except Exception as e:
            print(f"Space creation info: {e}")

        # Upload files
        files_to_upload = [
            ("app.py", "app.py"),
            ("requirements.txt", "requirements.txt"),
            ("Dockerfile", "Dockerfile")
        ]

        for local_path, repo_path in files_to_upload:
            if os.path.exists(local_path):
                print(f"Uploading {local_path}...")
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=repo_path,
                    repo_id=space_id,
                    repo_type="space",
                    token=HF_TOKEN
                )
                print(f"✅ {local_path} uploaded")
            else:
                print(f"⚠️  {local_path} not found")

        print(f"\n✅ Deployment completed!")
        print(f"🔗 App URL: https://huggingface.co/spaces/{space_id}")
        return True

    except Exception as e:
        print(f"❌ Deployment error: {e}")
        return False

if __name__ == "__main__":
    deploy_to_huggingface_space()
