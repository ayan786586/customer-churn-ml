
from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

from src.data_ingestion import load_data
from src.logger import logger
from src.exception import CustomerChurnException


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def tune_model():
    logger.info("Starting Logistic Regression hyperparameter tuning.")

    try:
        # 1. Load the dataset
        df = load_data().copy()

        # 2. Convert TotalCharges into numeric data
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce",
        )

        # 3. Separate input features and target
        X = df.drop(columns=["Churn", "customerID"])
        y = df["Churn"].map({"No": 0, "Yes": 1})

        if y.isna().any():
            raise ValueError("Unexpected or missing Churn labels.")

        # 4. Identify numeric and categorical columns
        numeric_features = X.select_dtypes(
            include=["number"]
        ).columns.tolist()

        categorical_features = X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        # 5. Split data before fitting preprocessing
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        # 6. Create the numeric preprocessing pipeline
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        # 7. Create the categorical preprocessing pipeline
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])

        # 8. Combine numeric and categorical preprocessing
        preprocessor = ColumnTransformer([
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ])

        # 9. Combine preprocessing and Logistic Regression
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ])

        # 10. Define the hyperparameters to test
        parameter_grid = {
            "model__C": [0.01, 0.1, 1, 10],
            "model__class_weight": [None, "balanced"],
        }

        # 11. Configure five-fold cross-validation
        cross_validation = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42,
        )

        # 12. Search for the best configuration
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring="f1",
            cv=cross_validation,
            n_jobs=-1,
            refit=True,
        )

        logger.info("Running GridSearchCV.")
        search.fit(X_train, y_train)

        best_model = search.best_estimator_

        print("\n========== TUNING RESULTS ==========")
        print("Best parameters:", search.best_params_)
        print(f"Best CV F1-score: {search.best_score_:.4f}")

        # 13. Evaluate the selected model on the test set
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        print("\n========== FINAL TEST RESULTS ==========")
        print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
        print(
            f"Precision: "
            f"{precision_score(y_test, y_pred, zero_division=0):.4f}"
        )
        print(
            f"Recall:    "
            f"{recall_score(y_test, y_pred, zero_division=0):.4f}"
        )
        print(f"F1-score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
        print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")

        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                y_pred,
                target_names=["No Churn", "Churn"],
                zero_division=0,
            )
        )

        # 14. Save the complete fitted pipeline
        model_path = MODEL_DIR / "best_logistic_pipeline.joblib"
        joblib.dump(best_model, model_path)

        logger.info("Best pipeline saved to %s", model_path)
        print("\nComplete model pipeline saved to:", model_path)

        return best_model

    except Exception as e:
        logger.exception("Model tuning failed.")
        raise CustomerChurnException(e) from e


if __name__ == "__main__":
    tune_model()
