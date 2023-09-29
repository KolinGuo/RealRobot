import multiprocessing as mp
from .shared_object import SharedObject
from .utils import SharedObjectDefaultDict

ctx = mp.get_context("forkserver" if "forkserver" in mp.get_all_start_methods()
                     else "spawn")
