"""
logging_setup.py

Sets up logging for the whole application.

Usage in any other module:

    from src.logging_setup import get_logger
    logger = get_logger(__name__)
    logger.info("Loaded image %s", filename)

Messages are written to the console AND to logs/app.log. The log file
rotates: when it reaches about 1 MB, a new one is started, so logs can
never fill your disk.
"""

import logging
from logging.handlers import RotatingFileHandler

from src import config

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Name shared by all of our loggers, so we can configure them in one place
# without changing the logging of third-party libraries.
ROOT_LOGGER_NAME = "photo_restorer"

_is_configured = False


def setup_logging() -> None:
    """Configure console and file logging. Safe to call more than once."""
    global _is_configured
    if _is_configured:
        return  # avoid adding duplicate handlers (duplicate log lines)

    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # An invalid level name in the environment falls back to INFO.
    level = getattr(logging, config.LOG_LEVEL, logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    app_logger = logging.getLogger(ROOT_LOGGER_NAME)
    app_logger.setLevel(level)
    app_logger.propagate = False  # do not pass messages to the root logger

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    app_logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        config.LOG_FILE,
        maxBytes=config.LOG_FILE_MAX_BYTES,
        backupCount=config.LOG_FILE_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    app_logger.addHandler(file_handler)

    _is_configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger for a module. Pass __name__ as the name.

    Example: get_logger("src.image_io") returns a logger whose messages
    show "photo_restorer.src.image_io" in the log.
    """
    setup_logging()
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


if __name__ == "__main__":
    # Quick self-test: run  python -m src.logging_setup
    test_logger = get_logger("self_test")
    test_logger.debug("DEBUG message (hidden unless PHOTO_LOG_LEVEL=DEBUG)")
    test_logger.info("INFO message: logging works")
    test_logger.warning("WARNING message: something to be aware of")
    test_logger.error("ERROR message: something went wrong (this is only a test)")
    print(f"\nLog file written to: {config.LOG_FILE}")