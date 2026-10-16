from pathlib import Path
import pandas as pd
from src.logger import logger
from src.exception import CustomerChurnException
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (
  PROJECT_ROOT
  /"data"
  /"raw"
  /"Telco-Customer-Churn.csv"
)

def load_data(file_path:Path=DATA_PATH)->pd.DataFrame:
  '''
  Load the raw customer churn dataset.
  '''
  logger.info("Starting data ingestion.")
  logger.info("DataSet path: %s",file_path)

  try:
    if not file_path.is_file():
       raise FileNotFoundError(
          f"Dataset not found : {file_path}"
       )
    df = pd.read_csv(
       file_path,
       na_values=[""," "],
       skipinitialspace=True,
    )
    if df.empty:
       raise ValueError("Dataset Contain no rows.")
    logger.info(
      "Dataset Loaded Successfully : %s rows,%s columns",
      df.shape[0],
      df.shape[1],
    )
    logger.info(
       "missing values by column:\n%s",
        df.isna().sum().to_string(),
    )
    logger.info("Data Ingestion Completed Successfully")
    return df
  except Exception as e:
    raise CustomerChurnException(e) from e

if __name__ == "__main__":
    print("Data ingestion module started.")

    df = load_data()

    print("Dataset shape:", df.shape)
    print(df.head())