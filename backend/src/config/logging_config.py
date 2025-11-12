# loggingconfig.py
# src/config/logging_config.py
import logging
import logging.config
import sys
from pathlib import Path

# It's better to get the root directory from a central place like settings
# to avoid relative path issues. We'll assume settings.py has it.
from .settings import ROOT_DIR

LOGS_DIR = ROOT_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)  # Create logs directory if it doesn't exist

def setup_base_logging():
    """
    Configures the root logger and console handler.
    This should be called only ONCE at the start of the application.
    """
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {
                "format": "%(levelname)s - %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "simple",
                "stream": sys.stdout,
            },
        },
        "root": {
            "level": "DEBUG",
            "handlers": ["console"], 
        },
    }
    logging.config.dictConfig(logging_config)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    """
    Gets a logger and adds a unique, rotating file handler to it.
    This function can be called from any module to get a logger that
    writes to its own specific file (e.g., 'data_processor.log').
    """
    logger = logging.getLogger(name)

    if not any(isinstance(h, logging.FileHandler) and h.baseFilename.endswith(f"{name}.log") for h in logger.handlers):
        log_file = LOGS_DIR / f"{name}.log"

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=1024 * 1024 * 15,  # 15 MB
            backupCount=5,
            encoding="utf8",
        )

        detailed_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(detailed_formatter)
        file_handler.setLevel(logging.DEBUG) 

        logger.addHandler(file_handler)
        logger.propagate = True 

    return logger