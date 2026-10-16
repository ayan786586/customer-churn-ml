from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import(
  accuracy_score,
  precision_score,
  recall_score,f1_score,
  roc_auc_score
)
from src.logger import logger
from src.exception import CustomerChurnException
from src.data_ingestion import load_data
from src.data_preprocessing import preprocess_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT/"models"
MODEL_DIR.mkdir(parents=True,exist_ok=True)

def train_models():
  """Train and compare multiple classification models."""
  logger.info("Starting multi-model training.")
  try:
    df = load_data()
    (
      X_train,
      X_test,
      y_train,
      y_test,
      preprocessor,
    ) = preprocess_data(df)

    models = {
      "Logistic Regression":LogisticRegression(
        max_iter=1000,random_state=42,
      ),
      "Decision Tree":DecisionTreeClassifier(
        random_state=42,
      ),
      "Random Forest":RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
      ),
      "Gradient Boosting":GradientBoostingClassifier(
        random_state=42,
      ),

    }
    results = []
    train_models = {}

    for name,model in models.items():
      logger.info("Training model : %s",name)
      model.fit(X_train,y_train)

      y_pred = model.predict(X_test)
      y_prob = model.predict_proba(X_test)[:,1]

      accuracy = accuracy_score(y_test,y_pred)
      precision = precision_score(
        y_test,y_pred,zero_division=0
      )
      recall = recall_score(
        y_test,y_pred,zero_division=0
      )
      f1 = f1_score(
        y_test,y_pred,zero_division=0
      )
      roc_auc = roc_auc_score(y_test,y_prob)
      results.append(
        {
          "Model":name,
          "Accuracy":round(accuracy,4),
          "Precision":round(precision,4),
          "Recall":round(recall,4),
          "F1 Score":round(f1,4),
          "Roc-Auc":round(roc_auc,4)
        }
      )
      train_models[name] = model
      logger.info(
        "%s | Accuracy: %.4f | F1:%.4f | ROC-AUC:%.4f",
        name,accuracy,f1,roc_auc
      )
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
      by="F1 Score",
      ascending=False,
    )
    print("\n========== MODEL COMPARISON ==========")
    print(results_df.to_string(index=False))
    comparison_path = MODEL_DIR / "model_comparision.csv"
    results_df.to_csv(comparison_path,index=False)
    joblib.dump(
      train_models,
      MODEL_DIR / "candidate_models.joblib",
    )
    joblib.dump(
      train_models,
      MODEL_DIR/"preprocessor.joblib",
    )
    logger.info("Comparison saved to %s", comparison_path)
    logger.info("All candidate models saved successfully.")

    print("\nComparison saved to:", comparison_path)
    print("Trained models saved in models/candidate_models.joblib")
    print("Preprocessor saved in models/preprocessor.joblib")

    return results_df, train_models, preprocessor

  except Exception as e:
        logger.exception("Multi-model training failed.")
        raise CustomerChurnException(e) from e


if __name__ == "__main__":
    train_models()
