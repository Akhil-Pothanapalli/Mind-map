import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

# Check if directory exists and create it if necessary
if not os.path.exists(logs_path):
    os.makedirs(logs_path)
    print(f"Created log directory: {logs_path}")

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

print(f"Log file path: {LOG_FILE_PATH}")
print(f"Current working directory: {os.getcwd()}")

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,  # Try a lower level for testing
)

