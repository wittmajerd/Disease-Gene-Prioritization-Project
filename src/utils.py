import logging
import sys

def setup_logger():
    """
    Sets up a logger that outputs to the console (stdout).
    """
    logger = logging.getLogger("DL_Project")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
