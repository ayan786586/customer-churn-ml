import  logging
from pathlib import Path
from logging.handlers import RotatingFileHandler
PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOG_DIR = PROJECT_ROOT/"logs"
LOG_DIR.mkdir(parents=True,exist_ok=True)
LOG_FILE = LOG_DIR/"app.log"
file_handler = RotatingFileHandler(
  LOG_FILE,
  maxBytes = 5*1024*1024,
  backupCount=3,
  encoding="utf-8",
)
console_handler = logging.StreamHandler()
logging.basicConfig(
 level=logging.INFO,
 format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
 handlers=[
   file_handler,console_handler
 ],
 force=True
)
logger  = logging.getLogger("Customer_churn")