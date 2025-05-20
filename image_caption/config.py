import logging
import logging.handlers
from pathlib import Path

from rich.logging import RichHandler

BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = Path(BASE_DIR, "logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Get root logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Set lower logging level for other libraries
logging.getLogger("PIL").setLevel(logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.INFO)

# Create handlers
console_handler = RichHandler(markup=True)
console_handler.setLevel(logging.DEBUG)
info_handler = logging.handlers.RotatingFileHandler(
    filename=Path(LOGS_DIR, "info.log"),
    maxBytes=10485760,  # 1 MB
    backupCount=10,
)
info_handler.setLevel(logging.INFO)
error_handler = logging.handlers.RotatingFileHandler(
    filename=Path(LOGS_DIR, "error.log"),
    maxBytes=10485760,  # 1 MB
    backupCount=10,
)
error_handler.setLevel(logging.ERROR)

# Create formatters
minimal_formatter = logging.Formatter(fmt="%(message)s")
detailed_formatter = logging.Formatter(
    fmt="%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
)

# Hook it all up
console_handler.setFormatter(fmt=minimal_formatter)
info_handler.setFormatter(fmt=detailed_formatter)
error_handler.setFormatter(fmt=detailed_formatter)
logger.addHandler(hdlr=console_handler)
logger.addHandler(hdlr=info_handler)
logger.addHandler(hdlr=error_handler)
