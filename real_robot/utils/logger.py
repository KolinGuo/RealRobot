import os
import sys
import logging
from copy import deepcopy
from pathlib import Path
from datetime import datetime
import multiprocessing as mp


_registry = []
_format = "[%(asctime)s] [%(name)s] [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s"
_log_dir = Path(os.getenv(
    "REAL_ROBOT_LOG_DIR",
    Path.home() / f"real_robot_logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
))
_current_pid = -1
_default_file_handler = None


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


def get_logger(name=None, *, fmt=_format, datefmt=None,
               with_stream=True, stdout=False, log_file=None,
               log_level=logging.INFO, log_file_level=logging.NOTSET) -> logging.Logger:
    """Initialize a logger by name and add to registry.
    By default, it will add a FileHandler to
        _log_dir / "master.log" for main process
        _log_dir / "<proc_name>_<proc_pid>.log" for child processes

    If logger is in _registry, that logger is directly returned
    If logger is a child of a logger in _registry, its kwargs are ignored and
        will use its parent's kwargs

    :param name: Logger name. If not specified, get the root logger
    :param fmt: stream logging format, default is logger._format
    :param date_fmt: date (asctime) logging format, default is '%Y-%m-%d %H:%M:%S,uuu'
    :param with_stream: whether to add StreamHandler for terminal output
    :param stdout: StreamHandler outputs to sys.stdout or sys.stderr
    :param log_file: log filename. If specified, a FileHandler will be added.
    :param log_level: logger StreamHandler logging level.
    :param log_file_level: logger FileHandler logging level.
    :return logger: logging.Logger, the expected logger.
    """
    # Initialize for a new process
    global _registry, _log_dir, _current_pid, _default_file_handler
    if _current_pid != os.getpid():  # this is a new process
        _registry = []
        _current_pid = os.getpid()
        _log_dir = Path(os.getenv("REAL_ROBOT_LOG_DIR", _log_dir))
        _log_dir.mkdir(parents=True, exist_ok=True)
        # Create default FileHandler
        if mp.parent_process() is None:  # main process
            _default_log_file = _log_dir / "master.log"
        else:  # child process
            cur_proc = mp.current_process()
            _default_log_file = _log_dir / f"{cur_proc.name}_{cur_proc.pid}.log"
        _default_file_handler = logging.FileHandler(_default_log_file)
        _default_file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        _default_file_handler.setLevel(log_file_level)

    if len(_registry) == 0:
        logging.basicConfig(format=_format, level=logging.NOTSET, handlers=[])

    logger = logging.getLogger(name)
    # e.g., logger "a" is initialized, then logger "a.b" will skip the initialization
    #   since it is a child of "a".
    for _logger in _registry:
        if logger is _logger or name is not None and name.startswith(_logger.name+'.'):
            return logger

    logger.propagate = False  # allow propergate to root logger
    handlers = []

    if with_stream and name is not None:  # no stream for root logger
        handlers.append(logging.StreamHandler(sys.stdout if stdout else sys.stderr))
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))

    color_formatter = ColorFormatter(fmt=fmt, datefmt=datefmt)
    file_formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    logger.handlers = []

    logger.addHandler(_default_file_handler)
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
