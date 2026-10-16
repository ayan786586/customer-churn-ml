import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.logger import logger
from src.exception import CustomerChurnException
from src.data_ingestion import load_data

def preprocess_data(df:pd.DataFrame):
  """
  prepare customer churn data for machine learning
  """
  logger.info("Starting data preprocessing")

  try:
    df= df.copy()
    df["TotalCharges"] = pd.to_numeric(
      df["TotalCharges"],
      errors="coerce"
    )
    X = df.drop(columns=["Churn","customerID"])
    y = df["Churn"].map({"No":0,"Yes":1})

    if y.isna().any():
      raise ValueError("Unexpected or missing values in churn")
    numeric_features = X.select_dtypes(
      include=["int64","float64"]
    ).columns.tolist()
    categorical_features = X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()
    logger.info("Numeric fetures : %s",numeric_features)
    logger.info("categorical features: %s",categorical_features)
    X_train,X_test,y_train,y_test = train_test_split(
      X,y,test_size=0.2,
      random_state=42,
      stratify=y
    )
    numerc_pipeline = Pipeline(
      steps=[
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())
      ]
    )
    categorical_pipeline = Pipeline(
      steps=[
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("encode",OneHotEncoder(handle_unknown="ignore"))
      ]
    )
    preprocessor = ColumnTransformer(
      transformers=[
        ("numeric",numerc_pipeline,numeric_features),
        ("categorical",categorical_pipeline,categorical_features)
      ]
    )
    x_train_processed = preprocessor.fit_transform(X_train)
    x_test_processed = preprocessor.transform(X_test)
    logger.info(
      "Preprocessing Completed,Training Shape:%s,Test shape:%s",
      x_train_processed.shape,
      x_test_processed.shape
    )
    return (
            x_train_processed,
            x_test_processed,
            y_train,
            y_test,
            preprocessor
        )
  except Exception as e:
    logger.exception("Data preprocessing failed")
    raise CustomerChurnException(e) from e

if __name__ == "__main__":
  df = load_data()
  X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

  print("\nPreprocessing successful!")
  print("Training features:", X_train.shape)
  print("Testing features:", X_test.shape)
  print("Training target:", y_train.shape)
  print("Testing target:", y_test.shape)
  print("Missing values handled by the pipeline.")