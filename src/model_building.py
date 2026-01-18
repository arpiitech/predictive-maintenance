import os
import json
import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from huggingface_hub import HfApi, create_repo, upload_file
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded")
except ImportError:
    print("⚠️  python-dotenv not installed")

# Configuration
HF_USERNAME_OR_ORG = os.getenv("HF_USERNAME_OR_ORG")
MODEL_NAME = os.getenv("MODEL_NAME", "engine-predictive-maintenance-sklearn")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO")

if not HF_USERNAME_OR_ORG or not HF_DATASET_REPO:
    print("❌ Required environment variables not set")
    print("Using local files instead")
    USE_LOCAL = True
else:
    USE_LOCAL = False
    MODEL_REPO_ID = f"{HF_USERNAME_OR_ORG}/{MODEL_NAME}"

# Load data
print("Loading training data...")
data_loaded = False

if not USE_LOCAL:
    try:
        train_df = load_dataset(HF_DATASET_REPO, split="train").to_pandas()
        test_df = load_dataset(HF_DATASET_REPO, split="test").to_pandas()
        print(f"✅ Data loaded from HuggingFace")
        data_loaded = True
    except Exception as e:
        print(f"Failed to load from HuggingFace: {e}")

if not data_loaded:
    try:
        train_df = pd.read_csv("data/train_data.csv")
        test_df = pd.read_csv("data/test_data.csv")
        print(f"✅ Data loaded locally")
        data_loaded = True
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        raise

# Prepare features
y_train = train_df["engine_condition"].astype(int)
X_train = train_df.drop(columns=["engine_condition"])
y_test = test_df["engine_condition"].astype(int)
X_test = test_df.drop(columns=["engine_condition"])

print(f"Training features shape: {X_train.shape}")
print(f"Test features shape: {X_test.shape}")

# Preprocessing pipeline
num_cols = X_train.columns.tolist()
preprocessor = ColumnTransformer([("num", StandardScaler(), num_cols)])

# Model configurations with hyperparameters
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

# Train and evaluate models
results = []
best = {"name": None, "estimator": None, "f1": -1}

print("\n" + "="*60)
print("STARTING MODEL TRAINING")
print("="*60)

for name, (clf, grid) in candidates.items():
    print(f"\nTraining {name.upper()}...")

    try:
        pipe = Pipeline([("pre", preprocessor), ("clf", clf)])
        gs = GridSearchCV(pipe, grid, scoring="f1", cv=5, n_jobs=-1, verbose=1)
        gs.fit(X_train, y_train)

        # Predict
        y_pred = gs.best_estimator_.predict(X_test)
        y_proba = gs.best_estimator_.predict_proba(X_test)[:, 1]

        # Calculate metrics
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

        print(f"✅ {name.upper()} Results:")
        print(f"   Best params: {gs.best_params_}")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   F1 Score: {metrics['f1']:.4f}")
        print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")

        # Track best model
        if metrics["f1"] > best["f1"]:
            best["name"] = name
            best["estimator"] = gs.best_estimator_
            best["f1"] = metrics["f1"]

    except Exception as e:
        print(f"❌ Error training {name}: {e}")

# Save results
os.makedirs("src/model_building", exist_ok=True)

# Save experiment logs
with open("src/model_building/experiments.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\n✅ Experiment results saved to model_building/experiments.json")

# Save best model info
with open("src/model_building/best_model.json", "w") as f:
    json.dump({"best_model": best["name"], "f1_score": best["f1"]}, f, indent=2)

# Save best model
if best["estimator"] is not None:
    joblib.dump(best["estimator"], "src/model_building/best_model.joblib")
    print(f"✅ Best model ({best['name']}) saved locally")

    # Create model card
    model_card = f"""# Engine Predictive Maintenance Model

## Model Description
Sklearn pipeline (StandardScaler + {best['name'].upper()}) for engine predictive maintenance.

## Performance
- F1 Score: {best['f1']:.4f}

## Features
- engine_rpm
- lub_oil_pressure
- fuel_pressure
- coolant_pressure
- lub_oil_temperature
- coolant_temperature

## Target
- engine_condition (0=Normal, 1=Needs Maintenance)

## Usage
```python
import joblib
model = joblib.load('best_model.joblib')
prediction = model.predict(X)
```
"""

    with open("src/model_building/README.md", "w") as f:
        f.write(model_card)

    # Upload to HuggingFace
    if not USE_LOCAL:
        try:
            HF_TOKEN = os.getenv("HF_TOKEN")
            create_repo(MODEL_REPO_ID, repo_type="model", exist_ok=True)
            api = HfApi()

            api.upload_file(
                path_or_fileobj="src/model_building/best_model.joblib",
                path_in_repo="model.joblib",
                repo_id=MODEL_REPO_ID,
            )

            api.upload_file(
                path_or_fileobj="src/model_building/README.md",
                path_in_repo="README.md",
                repo_id=MODEL_REPO_ID,
            )

            print(f"✅ Model uploaded to HuggingFace: https://huggingface.co/{MODEL_REPO_ID}")
        except Exception as e:
            print(f"❌ Error uploading to HuggingFace: {e}")

    # Print summary
    print("\n" + "="*60)
    print("MODEL TRAINING SUMMARY")
    print("="*60)
    print(f"Models trained: {len(results)}")
    print(f"Best model: {best['name'].upper()}")
    print(f"Best F1 Score: {best['f1']:.4f}")
    print("="*60)
else:
    print("❌ No models were successfully trained")
