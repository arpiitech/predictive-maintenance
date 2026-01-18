"""
Exploratory Data Analysis Script for GitHub Actions
Generates visualizations and saves them
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

# Configuration
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO")
OUTPUT_DIR = "outputs/eda_plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_style("whitegrid")

# Load data from HuggingFace
print("Loading dataset from HuggingFace...")
dataset = load_dataset(HF_DATASET_REPO, split="full")
df = dataset.to_pandas()
print(f"✅ Data loaded: {df.shape}")

# Target distribution
plt.figure(figsize=(8, 5))
df["engine_condition"].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title("Engine Condition Distribution")
plt.xlabel("Engine Condition (0=Normal, 1=Maintenance)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/target_distribution.png", dpi=150)
plt.close()

# Histograms
num_cols = [c for c in df.columns if c != "engine_condition"]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for idx, col in enumerate(num_cols):
    axes[idx].hist(df[col], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[idx].set_title(f"Distribution of {col}")
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/histograms.png", dpi=150)
plt.close()

# Correlation heatmap
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png", dpi=150)
plt.close()

# Boxplots by target
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for idx, col in enumerate(num_cols):
    sns.boxplot(data=df, x="engine_condition", y=col, ax=axes[idx],
                hue="engine_condition", palette="Set2", legend=False)
    axes[idx].set_title(f"{col} vs Engine Condition")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/boxplots_by_target.png", dpi=150)
plt.close()

print(f"✅ EDA plots saved to {OUTPUT_DIR}/")
