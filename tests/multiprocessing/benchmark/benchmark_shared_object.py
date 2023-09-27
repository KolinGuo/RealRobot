import uuid
import random
import string
import multiprocessing as mp
from time import sleep, perf_counter_ns
from collections import defaultdict
from pathlib import Path
from typing import Union, Tuple

import numpy as np
from sapien.core import Pose
from transforms3d.euler import euler2quat
import scipy
from prettytable import PrettyTable

from real_robot.utils.multiprocessing import SharedObject


def create_random_ndarray(dtype: Union[SharedObject._np_dtypes], shape: Tuple[int]):
    if np.issubdtype(dtype, np.bool_):
        data = np.random.randint(2, size=shape, dtype=dtype)
    elif np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        min, max = info.min, info.max
        data = np.random.randint(min, max+1, size=shape, dtype=dtype)
    elif np.issubdtype(dtype, np.inexact):
        info = np.finfo(np.float32)  # cannot sample uniform for float128
        min, max = info.min, info.max
        data = np.random.uniform(min, max, size=shape).astype(dtype)
    else:
        raise TypeError(f"Unknown numpy {dtype = }")
    return data


def create_random_object(
    object_type_idx: int, bytes_len=50, dtype=np.uint8, shape=(480, 848, 3)
) -> Union[SharedObject._object_types]:
    if object_type_idx == 0:  # None.__class__
        return None
    elif object_type_idx == 1:  # bool
        return bool(random.randrange(2))
    elif object_type_idx == 2:  # int
        return random.randint(-9223372036854775808, 9223372036854775807)
    elif object_type_idx == 3:  # float
        return (random.uniform(-100, 100) if bool(random.randrange(2))
                else random.uniform(-1e307, 1e308))
    elif object_type_idx == 4:  # sapien.core.Pose
        return Pose(p=np.random.uniform(-10, 10, size=3),
                    q=euler2quat(*np.random.uniform([0, 0, 0],
                                                    [np.pi*2, np.pi, np.pi*2])))
    elif object_type_idx == 5:  # str
        return ''.join(random.choices(string.printable, k=bytes_len))
    elif object_type_idx == 6:  # bytes
        return random.randbytes(bytes_len)
    elif object_type_idx == 7:  # np.ndarray
        return create_random_ndarray(dtype, shape)
    else:
        raise ValueError(f"Unknown {object_type_idx = }")


def benchmark_object_create(object_type_idx: int, n_iters=100, bytes_len=50,
                            dtype=np.uint8, shape=(480, 848, 3)) -> Tuple[float, float]:
    times_ns = []
    for _ in range(n_iters):
        data = create_random_object(object_type_idx, bytes_len=bytes_len,
                                    dtype=dtype, shape=shape)

        start_time = perf_counter_ns()
        so_data = SharedObject(uuid.uuid4().hex, data=data)
        so_data.unlink()
        times_ns.append(perf_counter_ns() - start_time)

    total_time_ns = sum(times_ns)
    if object_type_idx in [5, 6]:  # str, bytes
        out_str = f"{SharedObject._object_types[object_type_idx]} (len={bytes_len}): "
    elif object_type_idx == 7:  # np.ndarray
        out_str = f"{SharedObject._object_types[object_type_idx]} ({dtype=} {shape=}): "
    else:
        out_str = f"{SharedObject._object_types[object_type_idx]}: "
    print(out_str + f"{n_iters} iterations take {total_time_ns / 1e9 :.4g} seconds")

    return np.mean(times_ns), np.std(times_ns)


def benchmark_object_fetch(object_type_idx: int, n_iters=100, bytes_len=50,
                           dtype=np.uint8, shape=(480, 848, 3)) -> Tuple[float, float]:
    times_ns = []
    for _ in range(n_iters):
        data = create_random_object(object_type_idx, bytes_len=bytes_len,
                                    dtype=dtype, shape=shape)
        so_data = SharedObject(uuid.uuid4().hex, data=data)

        start_time = perf_counter_ns()
        so_data.fetch()
        times_ns.append(perf_counter_ns() - start_time)

        so_data.unlink()

    total_time_ns = sum(times_ns)
    if object_type_idx in [5, 6]:  # str, bytes
        out_str = f"{SharedObject._object_types[object_type_idx]} (len={bytes_len}): "
    elif object_type_idx == 7:  # np.ndarray
        out_str = f"{SharedObject._object_types[object_type_idx]} ({dtype=} {shape=}): "
    else:
        out_str = f"{SharedObject._object_types[object_type_idx]}: "
    print(out_str + f"{n_iters} iterations take {total_time_ns / 1e9 :.4g} seconds")

    return np.mean(times_ns), np.std(times_ns)


def benchmark_object_assign(object_type_idx: int, n_iters=100, bytes_len=50,
                            dtype=np.uint8, shape=(480, 848, 3)) -> Tuple[float, float]:
    times_ns = []
    for _ in range(n_iters):
        data = create_random_object(object_type_idx, bytes_len=bytes_len,
                                    dtype=dtype, shape=shape)
        new_data = create_random_object(object_type_idx, bytes_len=bytes_len,
                                        dtype=dtype, shape=shape)
        so_data = SharedObject(uuid.uuid4().hex, data=data)

        start_time = perf_counter_ns()
        so_data.assign(new_data)
        times_ns.append(perf_counter_ns() - start_time)

        so_data.unlink()

    total_time_ns = sum(times_ns)
    if object_type_idx in [5, 6]:  # str, bytes
        out_str = f"{SharedObject._object_types[object_type_idx]} (len={bytes_len}): "
    elif object_type_idx == 7:  # np.ndarray
        out_str = f"{SharedObject._object_types[object_type_idx]} ({dtype=} {shape=}): "
    else:
        out_str = f"{SharedObject._object_types[object_type_idx]}: "
    print(out_str + f"{n_iters} iterations take {total_time_ns / 1e9 :.4g} seconds")

    return np.mean(times_ns), np.std(times_ns)


def benchmark_object_create_ref(
    object_type_idx: int, n_iters=100, bytes_len=50, dtype=np.uint8, shape=(480, 848, 3)
) -> Tuple[float, float]:
    times_ns = []
    for _ in range(n_iters):
        data = create_random_object(object_type_idx, bytes_len=bytes_len,
                                    dtype=dtype, shape=shape)
        so_data = SharedObject(uuid.uuid4().hex, data=data)

        start_time = perf_counter_ns()
        so_data_ref = SharedObject(so_data.name)
        times_ns.append(perf_counter_ns() - start_time)

        so_data.close()
        so_data_ref.unlink()

    total_time_ns = sum(times_ns)
    if object_type_idx in [5, 6]:  # str, bytes
        out_str = f"{SharedObject._object_types[object_type_idx]} (len={bytes_len}): "
    elif object_type_idx == 7:  # np.ndarray
        out_str = f"{SharedObject._object_types[object_type_idx]} ({dtype=} {shape=}): "
    else:
        out_str = f"{SharedObject._object_types[object_type_idx]}: "
    print(out_str + f"{n_iters} iterations take {total_time_ns / 1e9 :.4g} seconds")

    return np.mean(times_ns), np.std(times_ns)


def benchmark_object_modified(
    object_type_idx: int, n_iters=100, bytes_len=50, dtype=np.uint8, shape=(480, 848, 3)
) -> Tuple[float, float]:
    times_ns = []
    for _ in range(n_iters):
        data = create_random_object(object_type_idx, bytes_len=bytes_len,
                                    dtype=dtype, shape=shape)
        so_data = SharedObject(uuid.uuid4().hex, data=data)

        start_time = perf_counter_ns()
        _ = so_data.modified
        times_ns.append(perf_counter_ns() - start_time)

        so_data.unlink()

    total_time_ns = sum(times_ns)
    if object_type_idx in [5, 6]:  # str, bytes
        out_str = f"{SharedObject._object_types[object_type_idx]} (len={bytes_len}): "
    elif object_type_idx == 7:  # np.ndarray
        out_str = f"{SharedObject._object_types[object_type_idx]} ({dtype=} {shape=}): "
    else:
        out_str = f"{SharedObject._object_types[object_type_idx]}: "
    print(out_str + f"{n_iters} iterations take {total_time_ns / 1e9 :.4g} seconds")

    return np.mean(times_ns), np.std(times_ns)


def benchmark_object_triggered(
    object_type_idx: int, n_iters=100, bytes_len=50, dtype=np.uint8, shape=(480, 848, 3)
) -> Tuple[float, float]:
    times_ns = []
    for _ in range(n_iters):
        data = create_random_object(object_type_idx, bytes_len=bytes_len,
                                    dtype=dtype, shape=shape)
        so_data = SharedObject(uuid.uuid4().hex, data=data)

        start_time = perf_counter_ns()
        _ = so_data.triggered
        times_ns.append(perf_counter_ns() - start_time)

        so_data.unlink()

    total_time_ns = sum(times_ns)
    if object_type_idx in [5, 6]:  # str, bytes
        out_str = f"{SharedObject._object_types[object_type_idx]} (len={bytes_len}): "
    elif object_type_idx == 7:  # np.ndarray
        out_str = f"{SharedObject._object_types[object_type_idx]} ({dtype=} {shape=}): "
    else:
        out_str = f"{SharedObject._object_types[object_type_idx]}: "
    print(out_str + f"{n_iters} iterations take {total_time_ns / 1e9 :.4g} seconds")

    return np.mean(times_ns), np.std(times_ns)


def benchmark_object_trigger(
    object_type_idx: int, n_iters=100, bytes_len=50, dtype=np.uint8, shape=(480, 848, 3)
) -> Tuple[float, float]:
    times_ns = []
    for _ in range(n_iters):
        data = create_random_object(object_type_idx, bytes_len=bytes_len,
                                    dtype=dtype, shape=shape)
        so_data = SharedObject(uuid.uuid4().hex, data=data)

        start_time = perf_counter_ns()
        so_data.trigger()
        times_ns.append(perf_counter_ns() - start_time)

        so_data.unlink()

    total_time_ns = sum(times_ns)
    if object_type_idx in [5, 6]:  # str, bytes
        out_str = f"{SharedObject._object_types[object_type_idx]} (len={bytes_len}): "
    elif object_type_idx == 7:  # np.ndarray
        out_str = f"{SharedObject._object_types[object_type_idx]} ({dtype=} {shape=}): "
    else:
        out_str = f"{SharedObject._object_types[object_type_idx]}: "
    print(out_str + f"{n_iters} iterations take {total_time_ns / 1e9 :.4g} seconds")

    return np.mean(times_ns), np.std(times_ns)


def child_benchmark_object_fetch_assign(object_type_idx: int, p_idx: int):
    so_data = SharedObject("data")
    so_joined = SharedObject("joined")
    so_ready = SharedObject(f"ready_{p_idx}")

    # Create fetch function
    if object_type_idx == 0:
        fn = lambda x: type(x)
    elif object_type_idx in [1, 2, 3]:
        fn = lambda x: x + 1
    elif object_type_idx == 4:  # sapien.core.Pose
        fn = None
    elif object_type_idx in [5, 6]:
        fn = lambda x: len(x)
    elif object_type_idx == 7:
        fn = lambda x: x.sum()
    else:
        raise ValueError(f"Unknown {object_type_idx = }")

    so_ready.trigger()
    while not so_joined.triggered:
        if so_data.modified:
            _ = so_data.fetch(fn)


def benchmark_object_2_proc_fetch_assign(
    object_type_idx: int, n_iters=100, bytes_len=50, dtype=np.uint8, shape=(480, 848, 3)
) -> Tuple[float, float]:
    data = create_random_object(object_type_idx, bytes_len=bytes_len,
                                dtype=dtype, shape=shape)
    so_data = SharedObject("data", data=data)
    so_joined = SharedObject("joined")
    so_ready = SharedObject("ready_0")

    times_ns = []
    procs = [ctx.Process(target=child_benchmark_object_fetch_assign,
                         args=(object_type_idx, 0)) for _ in range(n_iters)]
    for i in range(n_iters):
        data = create_random_object(object_type_idx, bytes_len=bytes_len,
                                    dtype=dtype, shape=shape)
        procs[i].start()
        while not so_ready.triggered:
            pass

        start_time = perf_counter_ns()
        for _ in range(5):
            if object_type_idx in [0, 1, 4, 5, 6]:
                sleep(1e-6)  # sleep a while to simulate gaps between writes
            else:
                data += 1
            so_data.assign(data)

        so_joined.trigger()
        procs[i].join()
        times_ns.append(perf_counter_ns() - start_time)

    so_data.unlink()
    so_joined.unlink()
    so_ready.unlink()

    total_time_ns = sum(times_ns)
    if object_type_idx in [5, 6]:  # str, bytes
        out_str = f"{SharedObject._object_types[object_type_idx]} (len={bytes_len}): "
    elif object_type_idx == 7:  # np.ndarray
        out_str = f"{SharedObject._object_types[object_type_idx]} ({dtype=} {shape=}): "
    else:
        out_str = f"{SharedObject._object_types[object_type_idx]}: "
    print(out_str + f"{n_iters} iterations take {total_time_ns / 1e9 :.4g} seconds")

    return np.mean(times_ns), np.std(times_ns)


def benchmark_object_5_proc_fetch_assign(
    object_type_idx: int, n_iters=100, bytes_len=50, dtype=np.uint8, shape=(480, 848, 3)
) -> Tuple[float, float]:
    data = create_random_object(object_type_idx, bytes_len=bytes_len,
                                dtype=dtype, shape=shape)
    so_data = SharedObject("data", data=data)
    so_joined = SharedObject("joined")
    so_readys = [SharedObject(f"ready_{i}") for i in range(5)]

    times_ns = []
    procs = [ctx.Process(target=child_benchmark_object_fetch_assign,
                         args=(object_type_idx, i % 5)) for i in range(n_iters*5)]
    for i in range(n_iters):
        data = create_random_object(object_type_idx, bytes_len=bytes_len,
                                    dtype=dtype, shape=shape)
        are_ready = [False] * 5
        [proc.start() for proc in procs[5*i:5*(i+1)]]
        while not all(are_ready):
            for j, so_ready in enumerate(so_readys):
                if so_ready.triggered:
                    are_ready[j] = True

        start_time = perf_counter_ns()
        for _ in range(5):
            if object_type_idx in [0, 1, 4, 5, 6]:
                sleep(1e-6)  # sleep a while to simulate gaps between writes
            else:
                data += 1
            so_data.assign(data)

        so_joined.trigger()
        [proc.join() for proc in procs[5*i:5*(i+1)]]
        times_ns.append(perf_counter_ns() - start_time)

    so_data.unlink()
    so_joined.unlink()
    [so_ready.unlink() for so_ready in so_readys]

    total_time_ns = sum(times_ns)
    if object_type_idx in [5, 6]:  # str, bytes
        out_str = f"{SharedObject._object_types[object_type_idx]} (len={bytes_len}): "
    elif object_type_idx == 7:  # np.ndarray
        out_str = f"{SharedObject._object_types[object_type_idx]} ({dtype=} {shape=}): "
    else:
        out_str = f"{SharedObject._object_types[object_type_idx]}: "
    print(out_str + f"{n_iters} iterations take {total_time_ns / 1e9 :.4g} seconds")

    return np.mean(times_ns), np.std(times_ns)


if __name__ == '__main__':
    ctx = mp.get_context("forkserver" if "forkserver" in mp.get_all_start_methods()
                         else "spawn")

    results = defaultdict(dict)

    # ----- create ----- #
    n_iters = 1000
    print('-' * 10 + " Benchmark creating SharedObject " + '-' * 10)
    for object_type_idx in range(len(SharedObject._object_types)-1):
        mean_ns, std_ns = benchmark_object_create(object_type_idx, n_iters=n_iters)
        results[object_type_idx]["create"] = (mean_ns, std_ns)

    dtype, shape = np.uint8, (480, 848, 3)
    mean_ns, std_ns = benchmark_object_create(7, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["create"] = (mean_ns, std_ns)
    n_iters = 500
    dtype, shape = np.float64, (720, 1280, 3)
    mean_ns, std_ns = benchmark_object_create(7, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["create"] = (mean_ns, std_ns)
    n_iters = 10
    dtype, shape = np.float64, (10000, 10000)
    mean_ns, std_ns = benchmark_object_create(7, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["create"] = (mean_ns, std_ns)

    # ----- fetch ----- #
    n_iters = 1000
    print('\n' + '-' * 10 + " Benchmark fetching SharedObject " + '-' * 10)
    for object_type_idx in range(len(SharedObject._object_types)-1):
        mean_ns, std_ns = benchmark_object_fetch(object_type_idx, n_iters=n_iters)
        results[object_type_idx]["fetch"] = (mean_ns, std_ns)

    dtype, shape = np.uint8, (480, 848, 3)
    mean_ns, std_ns = benchmark_object_fetch(7, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["fetch"] = (mean_ns, std_ns)
    dtype, shape = np.float64, (720, 1280, 3)
    mean_ns, std_ns = benchmark_object_fetch(7, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["fetch"] = (mean_ns, std_ns)
    n_iters = 20
    dtype, shape = np.float64, (10000, 10000)
    mean_ns, std_ns = benchmark_object_fetch(7, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["fetch"] = (mean_ns, std_ns)

    # ----- assign ----- #
    n_iters = 1000
    print('\n' + '-' * 10 + " Benchmark assigning SharedObject " + '-' * 10)
    for object_type_idx in range(len(SharedObject._object_types)-1):
        mean_ns, std_ns = benchmark_object_assign(object_type_idx, n_iters=n_iters)
        results[object_type_idx]["assign"] = (mean_ns, std_ns)

    dtype, shape = np.uint8, (480, 848, 3)
    mean_ns, std_ns = benchmark_object_assign(7, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["assign"] = (mean_ns, std_ns)
    dtype, shape = np.float64, (720, 1280, 3)
    mean_ns, std_ns = benchmark_object_assign(7, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["assign"] = (mean_ns, std_ns)
    n_iters = 50
    dtype, shape = np.float64, (10000, 10000)
    mean_ns, std_ns = benchmark_object_assign(7, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["assign"] = (mean_ns, std_ns)

    # ----- create_ref ----- #
    n_iters = 1000
    print('\n' + '-' * 10 + " Benchmark creating from existing SharedObject " + '-' * 10)
    for object_type_idx in range(len(SharedObject._object_types)-1):
        mean_ns, std_ns = benchmark_object_create_ref(object_type_idx, n_iters=n_iters)
        results[object_type_idx]["create_ref"] = (mean_ns, std_ns)

    dtype, shape = np.uint8, (480, 848, 3)
    mean_ns, std_ns = benchmark_object_create_ref(7, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["create_ref"] = (mean_ns, std_ns)
    dtype, shape = np.float64, (720, 1280, 3)
    mean_ns, std_ns = benchmark_object_create_ref(7, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["create_ref"] = (mean_ns, std_ns)
    n_iters = 50
    dtype, shape = np.float64, (10000, 10000)
    mean_ns, std_ns = benchmark_object_create_ref(7, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["create_ref"] = (mean_ns, std_ns)

    # ----- modified ----- #
    n_iters = 1000
    print('\n' + '-' * 10 + " Benchmark checking SharedObject modified " + '-' * 10)
    for object_type_idx in range(len(SharedObject._object_types)-1):
        mean_ns, std_ns = benchmark_object_modified(object_type_idx, n_iters=n_iters)
        results[object_type_idx]["modified"] = (mean_ns, std_ns)

    dtype, shape = np.uint8, (480, 848, 3)
    mean_ns, std_ns = benchmark_object_modified(7, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["modified"] = (mean_ns, std_ns)
    dtype, shape = np.float64, (720, 1280, 3)
    mean_ns, std_ns = benchmark_object_modified(7, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["modified"] = (mean_ns, std_ns)
    n_iters = 50
    dtype, shape = np.float64, (10000, 10000)
    mean_ns, std_ns = benchmark_object_modified(7, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["modified"] = (mean_ns, std_ns)

    # ----- triggered ----- #
    n_iters = 1000
    print('\n' + '-' * 10 + " Benchmark checking SharedObject triggered " + '-' * 10)
    for object_type_idx in range(len(SharedObject._object_types)-1):
        mean_ns, std_ns = benchmark_object_triggered(object_type_idx, n_iters=n_iters)
        results[object_type_idx]["triggered"] = (mean_ns, std_ns)

    dtype, shape = np.uint8, (480, 848, 3)
    mean_ns, std_ns = benchmark_object_triggered(7, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["triggered"] = (mean_ns, std_ns)
    dtype, shape = np.float64, (720, 1280, 3)
    mean_ns, std_ns = benchmark_object_triggered(7, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["triggered"] = (mean_ns, std_ns)
    n_iters = 50
    dtype, shape = np.float64, (10000, 10000)
    mean_ns, std_ns = benchmark_object_triggered(7, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["triggered"] = (mean_ns, std_ns)

    # ----- trigger ----- #
    n_iters = 1000
    print('\n' + '-' * 10 + " Benchmark triggering SharedObject " + '-' * 10)
    for object_type_idx in range(len(SharedObject._object_types)-1):
        mean_ns, std_ns = benchmark_object_trigger(object_type_idx, n_iters=n_iters)
        results[object_type_idx]["trigger"] = (mean_ns, std_ns)

    dtype, shape = np.uint8, (480, 848, 3)
    mean_ns, std_ns = benchmark_object_trigger(7, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["trigger"] = (mean_ns, std_ns)
    dtype, shape = np.float64, (720, 1280, 3)
    mean_ns, std_ns = benchmark_object_trigger(7, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["trigger"] = (mean_ns, std_ns)
    n_iters = 50
    dtype, shape = np.float64, (10000, 10000)
    mean_ns, std_ns = benchmark_object_trigger(7, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["trigger"] = (mean_ns, std_ns)

    # ----- 2 processes fetch/assign ----- #
    n_iters = 100
    print('\n' + '-' * 10 + " Benchmark fetching/assigning SharedObject with 2 processes " + '-' * 10)
    for object_type_idx in range(len(SharedObject._object_types)-1):
        mean_ns, std_ns = benchmark_object_2_proc_fetch_assign(object_type_idx, n_iters=n_iters)
        results[object_type_idx]["2 proc fetch_assign"] = (mean_ns, std_ns)

    dtype, shape = np.uint8, (480, 848, 3)
    mean_ns, std_ns = benchmark_object_2_proc_fetch_assign(7, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["2 proc fetch_assign"] = (mean_ns, std_ns)
    dtype, shape = np.float64, (720, 1280, 3)
    mean_ns, std_ns = benchmark_object_2_proc_fetch_assign(7, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["2 proc fetch_assign"] = (mean_ns, std_ns)
    n_iters = 10
    dtype, shape = np.float64, (10000, 10000)
    mean_ns, std_ns = benchmark_object_2_proc_fetch_assign(7, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["2 proc fetch_assign"] = (mean_ns, std_ns)

    # ----- 5 processes fetch/assign ----- #
    n_iters = 100
    print('\n' + '-' * 10 + " Benchmark fetching/assigning SharedObject with 5 processes " + '-' * 10)
    for object_type_idx in range(len(SharedObject._object_types)-1):
        mean_ns, std_ns = benchmark_object_5_proc_fetch_assign(object_type_idx, n_iters=n_iters)
        results[object_type_idx]["5 proc fetch_assign"] = (mean_ns, std_ns)

    dtype, shape = np.uint8, (480, 848, 3)
    mean_ns, std_ns = benchmark_object_5_proc_fetch_assign(7, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["5 proc fetch_assign"] = (mean_ns, std_ns)
    dtype, shape = np.float64, (720, 1280, 3)
    mean_ns, std_ns = benchmark_object_5_proc_fetch_assign(7, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["5 proc fetch_assign"] = (mean_ns, std_ns)
    n_iters = 10
    dtype, shape = np.float64, (10000, 10000)
    mean_ns, std_ns = benchmark_object_5_proc_fetch_assign(7, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["5 proc fetch_assign"] = (mean_ns, std_ns)

    # ----- print results table ----- #
    results_table = PrettyTable()
    results_table.field_names = (["Object Type / Duration for 1 obj (mean \xb1 std)"]
                                 + list(results[0].keys()))
    for idx, res_dict in results.items():
        results_table.add_row(
            [SharedObject._object_types[idx] if isinstance(idx, int) else idx]
            + [f"{mean_ns/1e9:.6g} \xb1 {std_ns/1e9:.2g}"
               for mean_ns, std_ns in res_dict.values()]
        )
    print(results_table)

    # ----- compute difference ----- #
    # Get latest benchmark results npz
    cur_dir = Path(__file__).resolve().parent
    prev_npz = sorted(cur_dir.glob("*.npz"), key=lambda p: p.stat().st_mtime_ns)[-1]
    prev_results = np.load(prev_npz, allow_pickle=True)["arr_0"].tolist()

    down_arrow_str = b"\xef\xbf\xac".decode("utf8")
    up_arrow_str = b"\xef\xbf\xaa".decode("utf8")
    sigma_str = b"\xcf\x83".decode("utf8")
    print(f"\nComparing with {str(prev_npz.name)}\n"
          f"\tshow if mean has increased ({up_arrow_str}) or decreased ({down_arrow_str}) "
          f"with (mean - prev_mean) and its confidence level")

    def compute_conf_level(mean_ns, std_ns, prev_mean_ns, prev_std_ns, N) -> float:
        """Compute confidence level for mean_ns > prev_mean_ns or mean_ns < prev_mean_ns
        to be statistically significant.

        Reference:
        https://sphweb.bumc.bu.edu/otlt/mph-modules/bs/bs704_confidence_intervals/bs704_confidence_intervals5.html
        Here, N1 = N2 = N
        """
        # convert uncorrected sample std to corrected sample std
        std_ns = np.sqrt(std_ns ** 2 * N / (N-1))
        prev_std_ns = np.sqrt(prev_std_ns ** 2 * N / (N-1))

        # assert 0.5 <= (ratio := std_ns**2/prev_std_ns**2) <= 2, ratio

        sp = np.sqrt((N-1) * (std_ns**2 + prev_std_ns**2) / (2*N - 2))
        t = np.abs(mean_ns - prev_mean_ns) / (sp * np.sqrt(2/N))
        one_tail_p = scipy.stats.t.cdf(t, 2*N - 2)
        return 1 - (1 - one_tail_p)*2

    N_iters = np.array(
        [[1000]*7 + [100, 100]]*8  # NoneType, bool, int, float, Pose, str, bytes, ndarray
        + [[500] + [1000] * 6 + [100, 100],  # ndarray (720, 1280, 3)
           [10, 20, 50, 50, 50, 50, 50, 10, 10]]  # ndarray (10000, 10000)
    )

    results_diff_table = PrettyTable()
    results_diff_table.field_names = (
        ["Object Type / Duration for 1 obj (mean, conf lvl)"]
        + list(prev_results[0].keys())
    )
    for i, (idx, res_dict) in enumerate(results.items()):
        row = [SharedObject._object_types[idx] if isinstance(idx, int) else idx]

        prev_res_dict = prev_results[idx]
        for j, res_key in enumerate(res_dict):
            if res_key not in prev_res_dict:
                continue
            mean_ns, std_ns = res_dict[res_key]
            prev_mean_ns, prev_std_ns = prev_res_dict[res_key]

            compare_str = ""
            # Mean comparison
            compare_str += f"{down_arrow_str} " if mean_ns < prev_mean_ns else f" {up_arrow_str} "
            compare_str += f"({(mean_ns - prev_mean_ns)/1e9:.4g}), "

            # Confidence level
            conf_level = compute_conf_level(mean_ns, std_ns, prev_mean_ns, prev_std_ns,
                                            N_iters[i, j])
            compare_str += f"{conf_level:.2f}"

            row.append(compare_str)
        results_diff_table.add_row(row)
    print(results_diff_table)

    # ----- generate results npz ----- #
    np.savez(cur_dir / "benchmark_shared_object_result.npz", results)
