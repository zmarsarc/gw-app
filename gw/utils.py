import random
import sys

from loguru import logger

from .settings import get_app_settings

random.seed()


def generate_a_random_hex_str(length: int) -> str:
    seq = random.choices("1234567890abcdef", k=length)
    return "".join(seq)


def initlize_logger(name: str):
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green>|"
        "<level>{level}</level>|"
        f"<cyan><bold>{name}</bold></cyan>|"
        "<magenta>{process}</magenta>|"
        "<yellow>{file}</yellow>:<yellow>{line}</yellow> - "
        "<level>{message}</level>"
    )

    logger.configure(handlers=[
        {
            "sink": sys.stdout,
            "format": log_format,
            "level": get_app_settings().log_level
        }
    ])
