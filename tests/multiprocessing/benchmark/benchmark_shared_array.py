import ctypes
import multiprocessing as mp
import os
import tempfile
import time
from contextlib import AbstractContextManager
from multiprocessing.shared_memory import SharedMemory
from time import perf_counter

import numpy as np

from real_robot.utils.logger import get_logger

os.environ["REAL_ROBOT_LOG_DIR"] = tempfile.TemporaryDirectory().name
_logger = get_logger(
    "Timer", fmt="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
)


class RuntimeTimer(AbstractContextManager):
    def __init__(self, description, enabled=True):
        self.description = description
        self.enabled = enabled
        self.elapsed_time = 0.0

    def __enter__(self):
        if self.enabled:
            self.start_time = perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.enabled:
            self.elapsed_time = perf_counter() - self.start_time
            _logger.info(f"{self.description}: Took {self.elapsed_time:.3f} seconds")


"""1. Shared via manager_dict"""


def read_forever(data_dict, lock):
    while True:
        with lock:
            if data_dict["join"]:
                break
            data = data_dict["array"]
        print(type(data_dict), type(data), len(data), data[0, 0], flush=True)
        time.sleep(0.01)


def test_shared_via_manager_dict(ctx, data, n_iters=3, n_updates=5):
    total_time = 0.0
    update_times = []

    input_data = data.copy()
    with RuntimeTimer("manager_dict") as t_all:
        for _ in range(n_iters):
            manager = ctx.Manager()
            lock = ctx.Lock()

            data_dict = manager.dict()
            data_dict["array"] = input_data
            data_dict["join"] = False

            p = ctx.Process(target=read_forever, args=(data_dict, lock))
            p.start()
            with RuntimeTimer("manager_dict: read_forever") as t:
                for _ in range(n_updates):
                    with lock:
                        data_dict["array"] += 1
                    time.sleep(0.01)
            update_times.append(t.elapsed_time)
            print("End:", data_dict["array"][0, 0], flush=True)
            with lock:
                data_dict["join"] = True
            p.join()
            manager.shutdown()
    total_time = t_all.elapsed_time

    update_time = np.mean(update_times)
    setup_time = total_time - np.sum(update_times)
    _logger.info(
        f"manager_dict: {total_time = :.6f} {update_time = :.6f} {setup_time = :.6f}"
    )


"""2. Shared via ctype_array"""


def read_forever_ctype(array, shape, joined):
    while True:
        with joined.get_lock():
            if joined.value:
                break
        with array.get_lock():
            data = np.ndarray(shape, np.double, array.get_obj())
        print(type(array), type(joined), len(data), data[0, 0], flush=True)
        time.sleep(0.01)


def test_shared_via_ctype_array(ctx, data: np.ndarray, n_iters=3, n_updates=5):
    total_time = 0.0
    update_times = []

    input_data = data.copy()
    with RuntimeTimer("ctype_array") as t_all:
        for _ in range(n_iters):
            ctype = np.ctypeslib.as_ctypes(input_data[0])._type_

            array = ctx.Array(ctype, input_data.flatten(), lock=True)
            shape = ctx.Array("i", input_data.shape, lock=array.get_lock())
            joined = ctx.Value(ctypes.c_bool, False, lock=True)

            data = np.ndarray(data.shape, data.dtype, array.get_obj())

            p = ctx.Process(target=read_forever_ctype, args=(array, shape, joined))
            p.start()
            with RuntimeTimer("ctype_array: read_forever_ctype") as t:
                for _ in range(n_updates):
                    with array.get_lock():
                        data += 1
                    time.sleep(0.01)
            update_times.append(t.elapsed_time)
            print("End:", data[0, 0], flush=True)
            with joined.get_lock():
                joined.value = True
            p.join()
    total_time = t_all.elapsed_time

    update_time = np.mean(update_times)
    setup_time = total_time - np.sum(update_times)
    _logger.info(
        f"ctype_array: {total_time = :.6f} {update_time = :.6f} {setup_time = :.6f}"
    )


"""3. Shared via ctype_rawarray"""


def read_forever_ctype_raw(array, shape, joined, array_lock, joined_lock):
    while True:
        with joined_lock:
            if joined.value:
                break
        with array_lock:
            data = np.ndarray(shape, np.double, array)
        print(type(array), type(joined), len(data), data[0, 0], flush=True)
        time.sleep(0.01)


def test_shared_via_ctype_rawarray(ctx, data: np.ndarray, n_iters=3, n_updates=5):
    total_time = 0.0
    update_times = []

    input_data = data.copy()
    with RuntimeTimer("ctype_rawarray") as t_all:
        for _ in range(n_iters):
            ctype = np.ctypeslib.as_ctypes(input_data[0])._type_

            array_lock = ctx.Lock()
            array = ctx.RawArray(ctype, input_data.flatten())
            shape = ctx.RawArray("i", input_data.shape)
            joined_lock = ctx.Lock()
            joined = ctx.RawValue(ctypes.c_bool, False)

            data = np.ndarray(data.shape, data.dtype, array)

            p = ctx.Process(
                target=read_forever_ctype_raw,
                args=(array, shape, joined, array_lock, joined_lock),
            )
            p.start()
            with RuntimeTimer("ctype_rawarray: read_forever_ctype_raw") as t:
                for _ in range(n_updates):
                    with array_lock:
                        data += 1
                    time.sleep(0.01)
            update_times.append(t.elapsed_time)
            print("End:", data[0, 0], flush=True)
            with joined_lock:
                joined.value = True
            p.join()
    total_time = t_all.elapsed_time

    update_time = np.mean(update_times)
    setup_time = total_time - np.sum(update_times)
    _logger.info(
        f"ctype_rawarray: {total_time = :.6f} {update_time = :.6f} {setup_time = :.6f}"
    )


"""4. Shared via SharedMemory"""


def read_forever_sharedmemory(array, shape, joined, array_lock, joined_lock):
    while True:
        with joined_lock:
            if joined.value:
                break
        with array_lock:
            data = np.ndarray(shape, np.double, array.buf)
        print(type(array), type(joined), len(data), data[0, 0], flush=True)
        time.sleep(0.01)


def test_shared_via_sharedmemory(ctx, data: np.ndarray, n_iters=3, n_updates=5):
    total_time = 0.0
    update_times = []

    input_data = data.copy()
    with RuntimeTimer("SharedMemory") as t_all:
        for _ in range(n_iters):
            array_lock = ctx.Lock()
            array = SharedMemory("array", create=True, size=data.nbytes)
            shape = ctx.RawArray("i", input_data.shape)
            joined_lock = ctx.Lock()
            joined = ctx.RawValue(ctypes.c_bool, False)

            data = np.ndarray(data.shape, dtype=data.dtype, buffer=array.buf)
            data[:] = input_data[:]

            p = ctx.Process(
                target=read_forever_sharedmemory,
                args=(array, shape, joined, array_lock, joined_lock),
            )
            p.start()
            with RuntimeTimer("SharedMemory: read_forever_sharedmemory") as t:
                for _ in range(n_updates):
                    with array_lock:
                        data += 1
                    time.sleep(0.01)
            update_times.append(t.elapsed_time)
            print("End:", data[0, 0], flush=True)
            with joined_lock:
                joined.value = True
            p.join()
            array.unlink()
    total_time = t_all.elapsed_time

    update_time = np.mean(update_times)
    setup_time = total_time - np.sum(update_times)
    _logger.info(
        f"SharedMemory: {total_time = :.6f} {update_time = :.6f} {setup_time = :.6f}"
    )


"""5. Shared via MemoryMappedFile"""


def read_forever_memmap_file(shape, joined, array_lock, joined_lock):
    while True:
        with joined_lock:
            if joined.value:
                break
        with array_lock:
            data = np.memmap(
                "/tmp/data.np", dtype=np.double, mode="r+", shape=tuple(shape)
            )
        print(type(joined), len(data), data[0, 0], flush=True)
        time.sleep(0.01)


def test_shared_via_memmap_file(ctx, data: np.ndarray, n_iters=3, n_updates=5):
    total_time = 0.0
    update_times = []

    input_data = data.copy()
    with RuntimeTimer("MemoryMappedFile") as t_all:
        for _ in range(n_iters):
            array_lock = ctx.Lock()
            shape = ctx.RawArray("i", input_data.shape)
            joined_lock = ctx.Lock()
            joined = ctx.RawValue(ctypes.c_bool, False)

            data = np.memmap(
                "/tmp/data.np", dtype=data.dtype, mode="w+", shape=data.shape
            )
            data[:] = input_data[:]

            p = ctx.Process(
                target=read_forever_memmap_file,
                args=(shape, joined, array_lock, joined_lock),
            )
            p.start()
            with RuntimeTimer("MemoryMappedFile: read_forever_memmap_file") as t:
                for _ in range(n_updates):
                    with array_lock:
                        data += 1
                        # data.flush()
                    time.sleep(0.01)
            update_times.append(t.elapsed_time)
            print("End:", data[0, 0], flush=True)
            with joined_lock:
                joined.value = True
            p.join()
    total_time = t_all.elapsed_time

    update_time = np.mean(update_times)
    setup_time = total_time - np.sum(update_times)
    _logger.info(
        f"MemoryMappedFile: {total_time = :.6f} {update_time = :.6f} "
        f"{setup_time = :.6f}"
    )


if __name__ == "__main__":
    ctx = mp.get_context(
        "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
    )

    n = 10000
    data = np.ones((n, n))

    # 1. Shared via manager_dict
    # [Timer] [INFO] manager_dict: total_time = 57.666792 update_time = 16.248225 setup_time = 8.922118
    test_shared_via_manager_dict(ctx, data, n_iters=3, n_updates=5)

    # 2. Shared via ctype_array (allocated on heap)
    # [Timer] [INFO] ctype_array: total_time = 18.985839 update_time = 0.297639 setup_time = 18.092923
    test_shared_via_ctype_array(ctx, data, n_iters=3, n_updates=5)

    # 3. Shared via ctype_rawarray (allocated on heap)
    # [Timer] [INFO] ctype_rawarray: total_time = 19.101692 update_time = 0.299847 setup_time = 18.202151
    test_shared_via_ctype_rawarray(ctx, data, n_iters=3, n_updates=5)

    # 4. Shared via SharedMemory
    # [Timer] [INFO] SharedMemory: total_time = 1.751088 update_time = 0.302887 setup_time = 0.842429
    test_shared_via_sharedmemory(ctx, data, n_iters=3, n_updates=5)

    # 5. Shared via MemoryMappedFile
    # [Timer] [INFO] MemoryMappedFile: total_time = 5.738195 update_time = 1.565792 setup_time = 1.040819
    # No flush()
    # [Timer] [INFO] MemoryMappedFile: total_time = 2.200087 update_time = 0.378451 setup_time = 1.064734
    test_shared_via_memmap_file(ctx, data, n_iters=3, n_updates=5)
