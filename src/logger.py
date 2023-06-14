import os
import sys
from datetime import datetime
import logging

# Creating log file with name as current datetime
LOG_FILE = f"{datetime.now().strftime('%m-%d-%Y-%H-%M-%S')}.log"
# Creating Logs folder int current dir under that we store our log file
log_path = os.path.join(os.getcwd(),"logs",LOG_FILE)
# Making dir
os.makedirs(log_path,exist_ok=True)
# joining our logs folder with log files
LOG_FILE_PATH = os.path.join(log_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format= "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

if __name__ == "__main__":
    logging.info('Testing of logging')