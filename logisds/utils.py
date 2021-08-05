import logging


def get_console_logger(name=__file__, level=logging.INFO):
    """
    This is a console logger getter.
    Parameters
    ----------
    name: name of the logger
    level: logging level

    Returns
    logging.logger
    -------

    """
    sh_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s- %(funcName)s -%(lineno)d-"
        " %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level)
    sh = logging.StreamHandler()
    sh.setFormatter(sh_formatter)
    logger.addHandler(sh)
    return logger
