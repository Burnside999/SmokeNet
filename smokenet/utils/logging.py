import logging
from pathlib import Path

PathLike = str | Path


def setup_logging(output_root: PathLike, level: int = logging.INFO) -> logging.Logger:
    """Configure a shared logger that writes to output_root/training.log.

    The logger is named "smokenet" and will output to both the console and
    a log file inside ``output_root``. Subsequent calls reuse the same
    handlers to avoid duplicate log entries.
    """

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "training.log"

    logger = logging.getLogger("smokenet")
    logger.setLevel(level)
    logger.propagate = False

    # Remove pre-existing handlers to avoid duplicate logs when reconfiguring.
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
