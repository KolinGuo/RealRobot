import multiprocessing as mp

from .shared_object import SharedObject
from .utils import (
    SharedObjectDefaultDict,
    signal_process_ready,
    start_and_wait_for_process,
)

ctx = mp.get_context(
    "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
)
