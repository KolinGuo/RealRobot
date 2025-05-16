"""
Loguru-based project-specific logger.
Consolidates multiprocessing/multithreading logs into a single log file.

Customize its behaviors with these environment variables:
  * logging file path: <PACKAGE_NAME>_LOG_PATH="/tmp/logs/your_log.log"
  * logging level (for stderr only, default=INFO): <PACKAGE_NAME>_LOG="TRACE"
    * If <PACKAGE_NAME>_LOG="TRACE", set <PACKAGE_NAME>_FILE_LOG="TRACE" as well
      unless <PACKAGE_NAME>_FILE_LOG is explicitly set.
  * logging level (for file sink only, default=DEBUG): <PACKAGE_NAME>_FILE_LOG="TRACE"

Usage:
  * Copy this file into your package (e.g., <package_name>/utils/logger.py)
  * Add the following lines to your __init__.py:
        ```python3
        from <package_name>.utils.logger import Logger

        LOGGER = Logger()
        ```
  * In anywhere of your package, import and use as follows:
        ```python3
        from <package_name> import LOGGER

        LOGGER.info("Hello, world!")
        ```

Quick start reference:
  * Logging methods:
        LOGGER.trace("This is a {} from class {!r} __repr__()", "trace message", Class)
              .debug()
              .info()
              .success()
              .warning()
              .error()
              .critical()
              .log(level, "This is {:.4f}", np.pi)
  * Log ERROR while also capturing exception:
        try:
            ...
        except Exception:
            LOGGER.exception("This is an {}", "exception message")

  * Available logging levels = [
        "TRACE" (5), "DEBUG" (10), "INFO" (20), "SUCCESS" (25), "WARNING" (30),
        "ERROR" (40), "CRITICAL" (50)
    ]

version 0.1.3

Written by Kolin Guo
"""

import functools
import os
import sys
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import loguru
from loguru import logger

# Default _log_path is /tmp/logs/<package_name>/<date_time>.log
# You can overwrite with '<PACKAGE_NAME>_LOG_PATH' environment variable
_package = __package__.partition(".")[0]  # type: ignore
_env_name = f"{_package.upper()}_LOG_PATH"
_log_path = Path(
    os.getenv(
        _env_name,
        f"/tmp/logs/{_package}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
)
os.environ[_env_name] = str(_log_path)  # set env variable to consolidates logs
_log_path.parent.mkdir(parents=True, exist_ok=True)


def _get_logging_format(
    detailed: bool = False, process: bool = True, thread: bool = False
) -> str:
    """
    Returns the desired logging format.

    :param detailed: Include more detailed information for file logging.
    :param process: Include the process info in the log output.
    :param thread: Include the thread info in the log output.
    :return: The desired logging format string.
    """
    if detailed:
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
        )
    else:
        log_format = (
            "<green>{time:MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
        )

    if process:
        log_format += (
            "<yellow>{process.name} (pid={process.id})</yellow> | "
            if detailed
            else "<yellow>{process.name}</yellow> | "
        )

    if thread:
        log_format += (
            "<yellow>{thread.name} (tid={thread.id})</yellow> | "
            if detailed
            else "<yellow>{thread.name}</yellow> | "
        )

    log_format += (
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    return log_format


class Logger:
    """
    A logger for packages that need to use loguru without disrupting the global config.
    All logs of this package should use an instance of this Logger.
    """

    def __init__(self):
        self.new_handler_ids = []

        # ----- Filter this project's logs from existing stdout/stderr handlers ----- #
        self.std_handler_orig_filter = {}  # {handler_id: filter}
        # For all handlers of the singleton logger
        for handler_id, handler in logger._core.handlers.items():  # type: ignore
            # If the handler is a stdout/stderr stream sink
            if isinstance(handler._sink, loguru._simple_sinks.StreamSink) and (  # type: ignore
                handler._sink._stream is sys.stderr
                or handler._sink._stream is sys.stdout
            ):
                # Save its original filter
                self.std_handler_orig_filter[handler_id] = handler._filter

                handler._filter = functools.partial(
                    self.new_filter_fn, old_filter_fn=handler._filter
                )

        # ----- Add handlers for this project's logs ----- #
        log_level = os.getenv(f"{_package.upper()}_LOG", "INFO").upper()
        file_log_level = os.getenv(
            f"{_package.upper()}_FILE_LOG", "TRACE" if log_level == "TRACE" else "DEBUG"
        ).upper()
        # Add a file sink for this library
        self.new_handler_ids.append(
            logger.add(
                _log_path,
                level=file_log_level,
                format=_get_logging_format(detailed=True, process=True, thread=False),
                filter=lambda record: record["extra"].get("__package__") == _package,
                backtrace=True,
                diagnose=True,
                enqueue=True,  # This is crucial for multiprocessing!
            )
        )

        # Add a stderr sink for this library
        self.new_handler_ids.append(
            logger.add(
                sys.stderr,
                level=log_level,
                format=_get_logging_format(detailed=False, process=True, thread=False),
                filter=lambda record: record["extra"].get("__package__") == _package,
                backtrace=True,
                diagnose=True,
                enqueue=True,  # This is crucial for multiprocessing!
            )
        )

        # Create a contextualized logger for this library
        self._logger = logger.bind(__package__=_package)

        # Log a debug message of current package version
        self.debug('Package "{}" version "{}"', _package, version(_package))

    @staticmethod
    def new_filter_fn(record: dict, old_filter_fn: Callable | None) -> bool:
        """New log filter function for existing handlers"""
        # Skip our package's logs in the handler
        if record["extra"].get("__package__") == _package:
            return False
        # Apply the original filter for other logs (if there was one)
        if callable(old_filter_fn):
            return old_filter_fn(record)
        return True

    def __getattr__(self, name):
        """Delegate all logging methods to the bound logger."""
        return getattr(self._logger, name)

    def cleanup(self):
        """
        Remove all handlers added by this library.
        Restore previous filter functions for stdout/stderr handlers.
        """
        for handler_id in self.new_handler_ids:
            logger.remove(handler_id)
        self.new_handler_ids = []

        # ----- Restore previous filter functions for stdout/stderr handlers ----- #
        for handler_id, handler in logger._core.handlers.items():  # type: ignore
            # If the handler is a stdout/stderr stream sink
            if isinstance(handler._sink, loguru._simple_sinks.StreamSink) and (  # type: ignore
                handler._sink._stream is sys.stderr
                or handler._sink._stream is sys.stdout
            ):
                handler._filter = self.std_handler_orig_filter[handler_id]
        self.std_handler_orig_filter = {}  # {handler_id: filter}

    @contextmanager
    def catch(self, *args, **kwargs):
        """Delegate the catch context manager to the internal logger."""
        with self._logger.catch(*args, **kwargs):
            yield

    def __del__(self):
        """On deletion, remove all handlers and wait for the logger to finish dumping"""
        try:
            self.cleanup()
            self._logger.complete()
        except Exception:
            # Avoid exceptions during garbage collection
            pass
