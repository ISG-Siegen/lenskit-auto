import logging


def get_logger(name: str = 'lenskit-auto', level: str = logging.INFO) -> logging.Logger:
    """
    Returns a logger with the given name and level.

    Parameters
    ----------
    name : str
        name of the logger
    level : int
        level of the logger

    Returns
    -------
    logging.Logger
        logger with the given name and level
    """
    # logging.basicConfig(
    #     level=level,
    #     format="%(asctime)s %(levelname)s %(message)s",
    #     datefmt="%Y-%m-%d %H:%M:%S",
    # )

    logger = logging.getLogger(name=name)
    logger.setLevel(level=level)

    fh = logging.StreamHandler()
    fh_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(fh_formatter)

    logger.addHandler(fh)

    return logger

