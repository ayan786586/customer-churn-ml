# Customer Churn Prediction

An end-to-end Machine Learning project that predicts whether a telecommunications customer is likely to leave a company. The project includes data ingestion, validation, preprocessing, model comparison, hyperparameter tuning, prediction, and an interactive Streamlit application.

## Features

- Data ingestion and logging
- Dataset validation and missing-value handling
- Feature preprocessing with scikit-learn pipelines
- Comparison of Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting
- Hyperparameter tuning using GridSearchCV and stratified 5-fold cross-validation
- Customer churn predictions with estimated probabilities
- Interactive Streamlit web application
- Automated tests using pytest

## Dataset

The project uses the IBM Telco Customer Churn dataset.

- Rows: 7,043
- Columns: 21
- Target: `Churn` (`Yes` or `No`)
- Missing values: 11 in `TotalCharges`

Customer IDs are excluded from model features. Missing numeric values are handled through an imputation pipeline.

## Model Results

### Initial model comparison

| Model | Accuracy | F1-score | ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression | 80.55% | 60.40% | 84.19% |
| Gradient Boosting | 80.62% | 58.95% | 84.34% |
| Random Forest | 77.79% | 53.21% | 81.62% |
| Decision Tree | 72.89% | 49.74% | 65.73% |

### Tuned Logistic Regression

The best configuration found by GridSearchCV was:

- `C`: 0.1
- `class_weight`: `balanced`
- Best cross-validation F1-score: 0.6295

Held-out test results:

- Accuracy: 74.24%
- Precision: 50.95%
- Recall: 78.61%
- F1-score: 61.83%
- ROC-AUC: 84.09%

The tuned model identifies more churners at the default classification threshold, with a trade-off in precision and accuracy.

## Tech Stack

- Python
- pandas
- scikit-learn
- NumPy
- joblib
- Streamlit
- pytest
- Git and GitHub

## Project Structure

```text
customer-churn-ml/
├── app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── logs/
├── src/
│   ├── data_ingestion.py
│   ├── data_validation.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_tuning.py
│   ├── predict.py
│   ├── logger.py
│   └── exception.py
├── tests/
│   └── test_predict.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

1. Clone the repository:

   ```bash
   git clone YOUR_GITHUB_REPOSITORY_URL
   cd customer-churn-ml
   ```

2. Create and activate a virtual environment.

   ```bash
   python -m venv .venv
   ```

   Windows:

   ```bash
   .venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Place the Telco Customer Churn CSV at:

   ```text
   data/raw/Telco-Customer-Churn.csv
   ```

5. Ensure the trained pipeline exists at:

   ```text
   models/best_logistic_pipeline.joblib
   ```

   If the model artifact is not included in the repository, generate it by running:

   ```bash
   python -m src.model_tuning
   ```

## Run the Application

```bash
streamlit run app.py
```

## Run Tests

```bash
pytest -v
```

## Important Notes

- Model predictions are estimates, not guarantees of future customer behavior.
- Performance may vary on new data.
- The test set was used during the initial candidate-model comparison. For a more rigorous final evaluation, model selection should be performed using training-set cross-validation, followed by one final evaluation on an untouched test set.

## Future Improvements

- Compare more classification algorithms.
- Tune additional models and decision thresholds.
- Add visualizations for churn patterns and model performance.
- Deploy the Streamlit application.
- Add continuous integration for automated tests.
