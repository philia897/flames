import logging
import os
import sys
import datetime

DEFAULT_FORMAT = "%(levelname)s %(name)s %(asctime)s | %(filename)s:%(lineno)d | %(message)s"
DEFAULT_LOG_FILE = "flames.log"
DEFAULT_LOG_LEVEL = "debug"
DEFAULT_LOG_NAME = "flames"

def create_id_by_timestamp(include_ms=True):
    if include_ms:
        tstamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    else:
        tstamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    return tstamp

def getLogger(name=DEFAULT_LOG_NAME, loglevel=DEFAULT_LOG_LEVEL, logfile=DEFAULT_LOG_FILE):
    logger = logging.getLogger(name=name)

    if logger.handlers:
        return logger # return the logger if it already exists
    else:
        loglevel = getattr(logging, loglevel.upper(), logging.DEBUG)
        logger.setLevel(loglevel)
        fmt = DEFAULT_FORMAT
        # fmt_date = '%Y-%m-%dT%T%Z'
        fmt_date = None
        formatter = logging.Formatter(fmt, fmt_date)
        if isinstance(logfile, str):
            handler2 = logging.FileHandler(logfile) # save logs to file
            handler2.setFormatter(formatter)
            logger.addHandler(handler2)
        handler = logging.StreamHandler() # print logs
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        if logger.name == 'root':
            logger.warning('Running: %s %s',
                        os.path.basename(sys.argv[0]),
                        ' '.join(sys.argv[1:]))
        return logger