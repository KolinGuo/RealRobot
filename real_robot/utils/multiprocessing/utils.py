import os
import time
import multiprocessing as mp

from .shared_object import SharedObject


class SharedObjectDefaultDict(dict):
    """This defaultdict helps to store SharedObject by name (only known at runtime)
    so we don't need to frequently create SharedObject
    """

    def __missing__(self, so_name: str) -> SharedObject:
        so = self[so_name] = SharedObject(so_name)
        return so


def start_and_wait_for_process(process: mp.Process, *, timeout: float = None) -> None:
    """Start and wait for process to be ready (finishes initialization)
    When the waiting process is ready, it should trigger SharedObject "proc_<pid>_ready"

    :param process: mp.Process
    :param timeout: If process is not ready after timeout seconds, raise a TimeoutError
                    If timeout is None, wait indefinitely
    """
    # TODO: should this be written using mp.Pipe?
    process.start()
    so_ready = SharedObject(f"proc_{process.pid}_ready")

    start_time = time.time()
    while not so_ready.triggered:
        if timeout is not None and time.time() - start_time > timeout:
            raise TimeoutError(
                f"Process {process.name} did not become ready within {timeout=} seconds"
            )


def signal_process_ready() -> None:
    """When called, signals that the current process is ready
    by triggering SharedObject "proc_<pid>_ready"
    """
    SharedObject(f"proc_{os.getpid()}_ready").trigger().unlink()
