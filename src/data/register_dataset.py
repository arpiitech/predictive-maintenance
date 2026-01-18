"""
Data Registration Script for GitHub Actions
Uploads raw dataset to HuggingFace Datasets
"""
import os
import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import create_repo, login

# Configuration from environment
HF_USERNAME_OR_ORG = os.getenv("HF_USERNAME_OR_ORG")
DATASET_NAME = os.getenv("HF_DATASET_NAME", "engine-predictive-maintenance")
DATASET_REPO_ID = f"{HF_USERNAME_OR_ORG}/{DATASET_NAME}"
LOCAL_CSV = "data/engine_data.csv"

# HuggingFace authentication
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
    print("✅ Logged in to HuggingFace")

# Load and clean data
print(f"Loading data from {LOCAL_CSV}...")
df = pd.read_csv(LOCAL_CSV)
print(f"✅ Dataset loaded: {len(df)} rows")

# Normalize columns
df.columns = [c.strip().lower().replace(" ", "_").replace("/", "_") for c in df.columns]

# Convert to numeric and clean
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna()

# Rename columns
column_mapping = {
    "lub_oil_temp": "lub_oil_temperature",
    "coolant_temp": "coolant_temperature"
}
for old, new in column_mapping.items():
    if old in df.columns:
        df = df.rename(columns={old: new})

print(f"✅ Data cleaned: {df.shape}")

# Create and upload dataset
ds = Dataset.from_pandas(df, preserve_index=False)
dd = DatasetDict({"full": ds})

create_repo(repo_id=DATASET_REPO_ID, repo_type="dataset", exist_ok=True)
dd.push_to_hub(DATASET_REPO_ID)

print(f"✅ Dataset uploaded to https://huggingface.co/datasets/{DATASET_REPO_ID}")
