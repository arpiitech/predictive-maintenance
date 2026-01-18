"""
Model Training Script for GitHub Actions
Trains multiple models, selects best, uploads to HuggingFace
"""
import os
import json
import joblib
from datasets import load_dataset
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from huggingface_hub import HfApi, create_repo

# Configuration
HF_USERNAME_OR_ORG = os.getenv("HF_USERNAME_OR_ORG")
MODEL_NAME = os.getenv("MODEL_NAME", "engine-predictive-maintenance-sklearn")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO")
MODEL_REPO_ID = f"{HF_USERNAME_OR_ORG}/{MODEL_NAME}"

# Load data
print("Loading training data...")
train_df = load_dataset(HF_DATASET_REPO, split="train").to_pandas()
test_df = load_dataset(HF_DATASET_REPO, split="test").to_pandas()

y_train = train_df["engine_condition"].astype(int)
X_train = train_df.drop(columns=["engine_condition"])
y_test = test_df["engine_condition"].astype(int)
X_test = test_df.drop(columns=["engine_condition"])

print(f"✅ Data loaded: Train={X_train.shape}, Test={X_test.shape}")

# Preprocessing
num_cols = X_train.columns.tolist()
preprocessor = ColumnTransformer([("num", StandardScaler(), num_cols)])

# Model configurations
candidates = {
    "logreg": (
        LogisticRegression(max_iter=1000, random_state=42),
        {"clf__C": [0.1, 1.0, 3.0]}
    ),
    "rf": (
        RandomForestClassifier(random_state=42),
        {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 10, 20]}
    ),
    "gb": (
        GradientBoostingClassifier(random_state=42),
        {"clf__n_estimators": [200, 400], "clf__learning_rate": [0.05, 0.1], "clf__max_depth": [2, 3]}
    ),
}

# Train models
results = []
best = {"name": None, "estimator": None, "f1": -1}

for name, (clf, grid) in candidates.items():
    print(f"\nTraining {name.upper()}...")
    pipe = Pipeline([("pre", preprocessor), ("clf", clf)])
    gs = GridSearchCV(pipe, grid, scoring="f1", cv=5, n_jobs=-1)
    gs.fit(X_train, y_train)

    y_pred = gs.best_estimator_.predict(X_test)
    y_proba = gs.best_estimator_.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "best_params": gs.best_params_,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }
    results.append(metrics)
    print(f"✅ {name} - F1: {metrics['f1']:.4f}")

    if metrics["f1"] > best["f1"]:
        best = {"name": name, "estimator": gs.best_estimator_, "f1": metrics["f1"]}

# Save results
os.makedirs("outputs", exist_ok=True)
with open("outputs/experiments.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

# Save best model
joblib.dump(best["estimator"], "outputs/best_model.joblib")
print(f"\n✅ Best model: {best['name']} (F1: {best['f1']:.4f})")

# Create model card
model_card = f"""# Engine Predictive Maintenance Model

## Best Model: {best['name'].upper()}
- F1 Score: {best['f1']:.4f}

## Features
- engine_rpm, lub_oil_pressure, fuel_pressure
- coolant_pressure, lub_oil_temperature, coolant_temperature

## Target
- engine_condition (0=Normal, 1=Maintenance Needed)
"""

with open("outputs/README.md", "w") as f:
    f.write(model_card)

# Upload to HuggingFace
create_repo(MODEL_REPO_ID, repo_type="model", exist_ok=True)
api = HfApi()
api.upload_file(
    path_or_fileobj="outputs/best_model.joblib",
    path_in_repo="model.joblib",
    repo_id=MODEL_REPO_ID,
)
api.upload_file(
    path_or_fileobj="outputs/README.md",
    path_in_repo="README.md",
    repo_id=MODEL_REPO_ID,
)

print(f"✅ Model uploaded to https://huggingface.co/{MODEL_REPO_ID}")
