# Engine Predictive Maintenance - MLOps Project

Machine learning pipeline for predicting engine maintenance needs in the automotive industry.

## Features
- Data registration on HuggingFace Datasets
- Comprehensive EDA with visualizations
- Multiple model training with hyperparameter tuning
- Automated CI/CD with GitHub Actions
- Streamlit app deployment on HuggingFace Spaces

## Project Structure
```
predictive-maintenance/
├── .github/workflows/pipeline.yml    # CI/CD automation
├── data/engine_data.csv              # Raw dataset
├── src/
│   ├── data/                         # Data processing scripts
│   └── models/                       # Model training scripts
├── space/                            # Streamlit app
└── requirements.txt                  # Dependencies
```

## Setup
1. Add dataset to `data/engine_data.csv`
2. Configure GitHub Secrets: `HF_TOKEN`
3. Configure GitHub Variables: usernames, repo paths
4. Push to GitHub - Actions will run automatically

## Links
- Dataset: https://huggingface.co/datasets/your-username/engine-predictive-maintenance
- Model: https://huggingface.co/your-username/engine-predictive-maintenance-sklearn
- App: https://huggingface.co/spaces/your-username/engine-predictive-maintenance-app
