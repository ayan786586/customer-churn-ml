from pathlib import Path
import joblib
import pandas as pd
from src.logger import logger
from src.exception import CustomerChurnException


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_logistic_pipeline.joblib"

def predict_churn(customer_data:dict)->dict:
  """
  predict churn for one customer
  """
  try:
    if not MODEL_PATH.is_file():
      raise FileNotFoundError(
        f"Trained pipeline not found: {MODEL_PATH}"
      )
    model = joblib.load(MODEL_PATH)

    customer_df = pd.DataFrame([customer_data])
    prediction = int(model.predict(customer_df)[0])
    probability = float(model.predict_proba(customer_df)[0,1])

    result = {
    "prediction": "Churn" if prediction == 1 else "No Churn",
    "churn_probability": round(probability * 100, 2),
    }
    logger.info("Prediction Complete : %s",result)
    return result
  except Exception as e:
      logger.exception("Customer churn prediction failed.")
      raise CustomerChurnException(e) from e


if __name__ == "__main__":
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.70,
        "TotalCharges": 171.40,
    }

    result = predict_churn(sample_customer)

    print("\n========== CUSTOMER CHURN PREDICTION ==========")
    print("Prediction:", result["prediction"])
    print("Churn probability:", f'{result["churn_probability"]}%')
