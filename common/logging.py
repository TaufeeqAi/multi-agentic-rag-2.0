import os
import logging
from logging.handlers import TimedRotatingFileHandler

# get the project root directory 
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Create logs directory in project root
LOG_DIR = os.path.join(ROOT_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Base log file path (active log)
BASE_LOG_FILE = os.path.join(LOG_DIR, 'app.log')

# Log formatter with timestamp
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# File handler: rotate logs daily at midnight, keep 7 days of logs
file_handler = TimedRotatingFileHandler(
    BASE_LOG_FILE,
    when='midnight',
    interval=1,
    backupCount=7,
    encoding='utf-8'
)
file_handler.suffix = "%Y-%m-%d"
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Console (stream) handler for development
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Configure the root logger explicitly
root_logger = logging.getLogger()
# Prevent duplicate handlers if re-imported
if not root_logger.handlers:
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)
