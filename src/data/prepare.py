"""
Data Preparation Script for GitHub Actions
Creates train/test splits and uploads to HuggingFace
"""
import os
from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split

# Configuration
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO")

# Load dataset
print("Loading dataset from HuggingFace...")
dataset = load_dataset(HF_DATASET_REPO, split="full")
df = dataset.to_pandas()
print(f"✅ Dataset loaded: {df.shape}")

# Split data
X = df.drop(columns=["engine_condition"])
y = df["engine_condition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create dataframes
train_df = X_train.copy()
train_df["engine_condition"] = y_train
test_df = X_test.copy()
test_df["engine_condition"] = y_test

print(f"✅ Train set: {len(train_df)} rows")
print(f"✅ Test set: {len(test_df)} rows")

# Upload splits to HuggingFace
from datasets import Dataset
train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

dd = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

dd.push_to_hub(HF_DATASET_REPO)
print(f"✅ Train/test splits uploaded to {HF_DATASET_REPO}")
