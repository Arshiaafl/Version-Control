# Model Training and Evaluation with MLflow

This repository contains code for training and evaluating machine learning models using `MLflow`. The primary focus is on data preprocessing, model training, and evaluation, along with integration of `MLflow` for tracking and managing models.

## Project Overview

The project involves training various machine learning models, including Logistic Regression, Random Forest, and XGBoost, on a dataset. The models are evaluated for performance, and results are logged using `MLflow`. The repository includes scripts for model training, evaluation, and deployment.

## Features

- **Data Preprocessing:** Preparation of datasets with scaling and splitting.
- **Model Training:** Training of Logistic Regression, Random Forest, and XGBoost models.
- **Model Evaluation:** Evaluation of model performance using metrics such as accuracy and classification report.
- **MLflow Integration:** Logging models and evaluation metrics with `MLflow` for tracking and management.

## Getting Started

### Prerequisites

- Python 3.x
- `pip` (Python package installer)

### Installation

1. Clone the repository:

   ```bash
   git clone github.com/Arshiaafl/Version-Control.git
   cd Version-Control
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `.\venv\Scripts\activate`
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Prepare the Dataset**: Ensure that the dataset file (`creditcard_2023.csv`) is in the correct directory. You can download it from: https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023

2. **Train and Log Models**: Run the `version1.py` script to train and log models using `MLflow`. Don't forget to run "mlflow ui" first.

   ```bash
   python version1.py
   ```

3. **Register Models**: Customize and run the `model_registry.py` script to deploy the model.

   ```bash
   python model_registry.py
   ```

4. **Evaluate Models**: Use the `test.py` script to evaluate model performance.

   ```bash
   python evaluate_model.py
   ```

### .gitignore

The `.gitignore` file is configured to exclude MLflow logs and model files, along with other unnecessary files. Key entries include:

```plaintext
# MLflow files
mlruns/
*.mlmodel
*.pkl

# Python bytecode
__pycache__/
*.pyc

# Virtual environment
venv/
.env

# Jupyter Notebook checkpoints
.ipynb_checkpoints/

# Log files
*.log

# Deployment files
*.zip
*.tar.gz

# Mac system files
.DS_Store
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes. Make sure to follow the coding standards and include appropriate tests.


## Acknowledgments

- [MLflow](https://mlflow.org/) for model tracking and management.
- [Scikit-learn](https://scikit-learn.org/) for machine learning algorithms and metrics.
- [XGBoost](https://xgboost.readthedocs.io/en/latest/) for gradient boosting.

