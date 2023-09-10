import sys
import logging
from copy import deepcopy


_registry = []
_format = "[%(asctime)s] [%(name)s] [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s"


class ColorFormatter(logging.Formatter):

    grey = "\x1b[37m"
    cyan = "\x1b[36m"
    green = "\x1b[32m"
    yellow = "\x1b[33m"
    red = "\x1b[31m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    LEVEL_COLORS = {
        logging.DEBUG: "grey",
        logging.INFO: "green",
        logging.WARNING: "yellow",
        logging.ERROR: "red",
        logging.CRITICAL: "bold_red",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._color_styles = {
            "grey": self.duplicate_style(self._style, self.grey),
            "cyan": self.duplicate_style(self._style, self.cyan),
            "green": self.duplicate_style(self._style, self.green),
            "yellow": self.duplicate_style(self._style, self.yellow),
            "red": self.duplicate_style(self._style, self.red),
            "bold_red": self.duplicate_style(self._style, self.bold_red)
        }

    @staticmethod
    def duplicate_style(style, ansi_color_str):
        style = deepcopy(style)
        style._fmt = ansi_color_str + style._fmt + ColorFormatter.reset
        return style

    def formatMessage(self, record):
        # Special color for timer info logs
        if "timer" in record.name.lower() and record.levelno == logging.INFO:
            return self._color_styles["cyan"].format(record)

        return self._color_styles[self.LEVEL_COLORS.get(record.levelno)].format(record)


def get_logger(name=None, fmt=_format, datefmt=None, with_stream=True, stdout=False,
               log_file=None, log_level=logging.INFO, log_file_level=logging.NOTSET):
    """Initialize a logger by name and add to registry.

    If logger is in _registry, that logger is directly returned
    If logger is a child of a logger in _registry, its kwargs are ignored and
        will use its parent's kwargs

    Args:
        name (str | None): Logger name. If not specified, get the root logger
        fmt (str): logging format
        datefmt (str): logging date format
        with_stream (bool): whether to add StreamHandler
        stdout (bool): StreamHandler outputs to stdout or stderr
        log_file (str | None): log filename. If specified, a FileHandler will be added.
        log_level (int): The logger level. Note that only the process of rank 0
                         is affected, and other processes will set the level to
                         "Error" thus be silent most of the time.
    Returns:
        logging.Logger: The expected logger.
    """
    if len(_registry) == 0:
        logging.basicConfig(format=_format, level=logging.NOTSET, handlers=[])

    logger = logging.getLogger(name)
    # e.g., logger "a" is initialized, then logger "a.b" will skip the initialization
    #   since it is a child of "a".
    for _logger in _registry:
        if logger is _logger or name is not None and name.startswith(_logger.name+'.'):
            return logger

    logger.propagate = True  # allow propergate to root logger
    handlers = []

    if with_stream and name is not None:  # no stream for root logger
        handlers.append(logging.StreamHandler(sys.stdout if stdout else sys.stderr))
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))

    color_formatter = ColorFormatter(fmt=fmt, datefmt=datefmt)
    file_formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    logger.handlers = []

    for handler in handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setFormatter(file_formatter)
            handler.setLevel(log_file_level)
            logger.addHandler(handler)
        else:
            handler.setFormatter(color_formatter)
            handler.setLevel(log_level)
            logger.addHandler(handler)

    # logger.setLevel(log_level)
    _registry.append(logger)
    return logger


logger = get_logger("real_robot")
