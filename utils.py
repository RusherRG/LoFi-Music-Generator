import colorlog
import logging

LOG_LEVEL = logging.DEBUG


def get_logger(name):
    bold_seq = "\033[1m"
    colorlog_format = (
        f"{bold_seq}"
        "%(log_color)s"
        "%(asctime)s | %(name)s.%(funcName)s | "
        "%(levelname)s:%(reset)s %(message)s"
    )
    colorlog.basicConfig(
        format=colorlog_format, level=logging.DEBUG, datefmt="%d/%m/%Y %H:%M:%S"
    )

    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    return logger
