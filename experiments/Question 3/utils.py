import os
import logging

def setup_logger(log_file_path):
    '''
    A helper function to setup the logger - allowing it to access files, read & write them.
    '''
    logger = logging.getLogger('custom_logger')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        fh = logging.FileHandler(log_file_path)
        formatter = logging.Formatter('%(asctime)s - %(message)s')

        fh.setLevel(logging.INFO)    
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger



def log_text(log_file_path, text):
    '''
    A helper function to log the text to the file path specified. 
    '''
    logger = setup_logger(log_file_path)
    print(text)
    logger.info(text)

