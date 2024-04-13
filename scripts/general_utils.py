# General file handling utilities

import os, sys, logging
###################################################
def create_path(path: str) -> None:
    '''
    Create a path if non-existent.

    Parameters:
    -----------------------------
    path: str
        Path to be created if non-existent
    '''
    if not os.path.isdir(path):
        print('%s does not exist.'% (path))
        print('Creating %s'% (path))
        os.makedirs(path)
###################################################
# Set up logger to output to stdout.
def setup_logger_stdout(level=logging.DEBUG, format: str ='%(asctime)s - %(levelname)s - %(message)s') -> logging.getLoggerClass:
    '''
    Set up logger to output messages to standard output. Returns object of logging.getLoggerClass

    Parameters:
    -----------------------------
    level: Logging severity threshold (logging.NOTSET, logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL)
        Sets the threshold for both the logger and the handler. Logging messages which are less severe than "level" will be ignored.
        Logging messages with severity equal to higher than "level" will be emitted by whichever handler services the logger.
    
    format: str
        Message format include date-time string
    '''
    root = logging.getLogger('main')
    root.propagate = False
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p')
    handler.setFormatter(formatter)
    root.addHandler(handler)

    return root
###################################################