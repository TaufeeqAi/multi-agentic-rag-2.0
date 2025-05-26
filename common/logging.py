import logging
import sys
from logging import Logger

def setup_logging(level: str = "INFO") -> Logger:
    """
    Configures root logger:
      - StreamHandler to stdout
      - Consistent format: timestamp • level • module • message
    """
    logger = logging.getLogger()  # root logger
    if logger.handlers:
        return logger

    # Create format
    fmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)8s | %(name)s | %(message)s',
        datefmt=fmt
    )

    # Handler to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.setLevel(level.upper())
    return logger