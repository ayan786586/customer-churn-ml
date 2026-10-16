from pathlib import Path
from src.logger import logger
from src.exception import CustomerChurnException
from src.data_ingestion import load_data
import pandas as pd

EXPECTED_COLUMNS = {
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
}
def validate_data(df):
  """ validate the cutomer churn dataset"""
  logger.info("start data validation")
  try:
    if df.empty:
      raise ValueError("Dataset is Empty.")
    missing_columns = EXPECTED_COLUMNS - set(df.columns)
    if missing_columns:
      raise ValueError(
        f"Missing required columns : {sorted(missing_columns)}"
      )
    if df["customerID"].duplicated().any():
      raise ValueError("Duplicate customer IDs found.")
    valid_churn_values = {"Yes","No"}
    actual_churn_values = set(df["Churn"].dropna().unique())
    if valid_churn_values != actual_churn_values:
      raise ValueError(
        f"Unexpected Churn Value : {actual_churn_values}"
      )
    missing_value = df.isna().sum()
    colums_with_missing_value= missing_value[missing_value>0]
    if colums_with_missing_value.empty:
      logger.info("No missing values found")
    else:
      logger.warning(
        "missing values found:\m%s",
        colums_with_missing_value.to_string(),
      )
    logger.info("Data validation completed successfully.")
    return True
  except Exception as e:
    logger.exception("Data validation failed")
    raise CustomerChurnException(e) from e
if __name__ == "__main__":
    df = load_data()
    is_valid = validate_data(df)

    print("Dataset validation successful:", is_valid)
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("Missing TotalCharges:", df["TotalCharges"].isna().sum())
print("Rows with zero tenure:", (df["tenure"] == 0).sum())
print(df.loc[df["TotalCharges"].isna(), ["tenure", "Churn"]])