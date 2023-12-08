import os
import sys
import logging

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging_dir = os.path.join(project_root, 'logs')
logfile_path = os.path.join(logging_dir, 'logfile.log')

# Gets or creates a logger
logger = logging.getLogger(__name__)

# set log level
logger.setLevel(logging.DEBUG)

# define file handler and set formatter
file_handler = logging.FileHandler(logfile_path, mode='a')
formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)

# add file handler to logger
logger.addHandler(file_handler)
# also log to stdout
logger.addHandler(logging.StreamHandler(sys.stdout))