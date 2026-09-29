import logging
import os


def setup_logging():
    """
    Configure application logging.

    Logs are written to both:
    - console
    - logs/constitution_ai.log
    """

    os.makedirs("logs", exist_ok=True)

    logger = logging.getLogger()

    if logger.handlers:
        return

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(
        "logs/constitution_ai.log",
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)