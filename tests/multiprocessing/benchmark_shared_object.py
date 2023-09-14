import uuid
import random
import string
import multiprocessing as mp
from time import sleep, perf_counter_ns
from collections import defaultdict
from typing import Union, Tuple

import numpy as np
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
    elif object_type_idx == 4:  # str
        return ''.join(random.choices(string.printable, k=bytes_len))
    elif object_type_idx == 5:  # bytes
        return random.randbytes(bytes_len)
    elif object_type_idx == 6:  # np.ndarray
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
    if object_type_idx in [4, 5]:  # str, bytes
        out_str = f"{SharedObject._object_types[object_type_idx]} (len={bytes_len}): "
    elif object_type_idx == 6:  # np.ndarray
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
    if object_type_idx in [4, 5]:  # str, bytes
        out_str = f"{SharedObject._object_types[object_type_idx]} (len={bytes_len}): "
    elif object_type_idx == 6:  # np.ndarray
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
    if object_type_idx in [4, 5]:  # str, bytes
        out_str = f"{SharedObject._object_types[object_type_idx]} (len={bytes_len}): "
    elif object_type_idx == 6:  # np.ndarray
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
    if object_type_idx in [4, 5]:  # str, bytes
        out_str = f"{SharedObject._object_types[object_type_idx]} (len={bytes_len}): "
    elif object_type_idx == 6:  # np.ndarray
        out_str = f"{SharedObject._object_types[object_type_idx]} ({dtype=} {shape=}): "
    else:
        out_str = f"{SharedObject._object_types[object_type_idx]}: "
    print(out_str + f"{n_iters} iterations take {total_time_ns / 1e9 :.4g} seconds")

    return np.mean(times_ns), np.std(times_ns)


def child_benchmark_object_fetch_assign(object_type_idx):
    # TODO: switch to timestamp
    so_data = SharedObject("data")
    so_data_updated = SharedObject("data_updated")
    so_joined = SharedObject("joined")

    # Create fetch function
    if object_type_idx == 0:
        fn = lambda x: type(x)
    elif object_type_idx in [1, 2, 3]:
        fn = lambda x: x + 1
    elif object_type_idx in [4, 5]:
        fn = lambda x: len(x)
    elif object_type_idx == 6:
        fn = lambda x: x.sum()
    else:
        raise ValueError(f"Unknown {object_type_idx = }")

    while True:
        if so_joined.fetch():
            break
        if so_data_updated.fetch():
            _ = so_data.fetch(fn)
            so_data_updated.assign(False)


def benchmark_object_2_proc_fetch_assign(
    object_type_idx: int, n_iters=100, bytes_len=50, dtype=np.uint8, shape=(480, 848, 3)
) -> Tuple[float, float]:
    # TODO: switch to timestamp
    data = create_random_object(object_type_idx, bytes_len=bytes_len,
                                dtype=dtype, shape=shape)
    so_data = SharedObject("data", data=data)
    so_data_updated = SharedObject("data_updated", data=False)
    so_joined = SharedObject("joined", data=False)

    times_ns = []
    procs = [ctx.Process(target=child_benchmark_object_fetch_assign,
                         args=(object_type_idx,)) for _ in range(n_iters)]
    for i in range(n_iters):
        data = create_random_object(object_type_idx, bytes_len=bytes_len,
                                    dtype=dtype, shape=shape)
        so_joined.assign(False)
        so_data_updated.assign(False)

        start_time = perf_counter_ns()
        procs[i].start()
        for _ in range(5):
            if object_type_idx in [0, 1, 4, 5]:
                sleep(1e-3)  # sleep a while to simulate gaps between writes
            else:
                data += 1
            so_data.assign(data)
            so_data_updated.assign(True)

        so_joined.assign(True)
        procs[i].join()
        times_ns.append(perf_counter_ns() - start_time)

    so_data.unlink()
    so_data_updated.unlink()
    so_joined.unlink()

    total_time_ns = sum(times_ns)
    if object_type_idx in [4, 5]:  # str, bytes
        out_str = f"{SharedObject._object_types[object_type_idx]} (len={bytes_len}): "
    elif object_type_idx == 6:  # np.ndarray
        out_str = f"{SharedObject._object_types[object_type_idx]} ({dtype=} {shape=}): "
    else:
        out_str = f"{SharedObject._object_types[object_type_idx]}: "
    print(out_str + f"{n_iters} iterations take {total_time_ns / 1e9 :.4g} seconds")

    return np.mean(times_ns), np.std(times_ns)


def benchmark_object_5_proc_fetch_assign(
    object_type_idx: int, n_iters=100, bytes_len=50, dtype=np.uint8, shape=(480, 848, 3)
) -> Tuple[float, float]:
    # TODO: switch to timestamp
    data = create_random_object(object_type_idx, bytes_len=bytes_len,
                                dtype=dtype, shape=shape)
    so_data = SharedObject("data", data=data)
    so_data_updated = SharedObject("data_updated", data=False)
    so_joined = SharedObject("joined", data=False)

    times_ns = []
    procs = [ctx.Process(target=child_benchmark_object_fetch_assign,
                         args=(object_type_idx,)) for _ in range(n_iters*5)]
    for i in range(n_iters):
        data = create_random_object(object_type_idx, bytes_len=bytes_len,
                                    dtype=dtype, shape=shape)
        so_joined.assign(False)
        so_data_updated.assign(False)

        start_time = perf_counter_ns()
        [proc.start() for proc in procs[5*i:5*(i+1)]]
        for _ in range(5):
            if object_type_idx in [0, 1, 4, 5]:
                sleep(1e-3)  # sleep a while to simulate gaps between writes
            else:
                data += 1
            so_data.assign(data)
            so_data_updated.assign(True)

        so_joined.assign(True)
        [proc.join() for proc in procs[5*i:5*(i+1)]]
        times_ns.append(perf_counter_ns() - start_time)

    so_data.unlink()
    so_data_updated.unlink()
    so_joined.unlink()

    total_time_ns = sum(times_ns)
    if object_type_idx in [4, 5]:  # str, bytes
        out_str = f"{SharedObject._object_types[object_type_idx]} (len={bytes_len}): "
    elif object_type_idx == 6:  # np.ndarray
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
        results[object_type_idx]["create"] = f"{mean_ns/1e9:.6g} \xb1 {std_ns/1e9:.2g}"

    dtype, shape = np.uint8, (480, 848, 3)
    mean_ns, std_ns = benchmark_object_create(6, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["create"] = f"{mean_ns/1e9:.6g} \xb1 {std_ns/1e9:.2g}"
    n_iters = 500
    dtype, shape = np.float64, (720, 1280, 3)
    mean_ns, std_ns = benchmark_object_create(6, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["create"] = f"{mean_ns/1e9:.6g} \xb1 {std_ns/1e9:.2g}"
    n_iters = 10
    dtype, shape = np.float64, (10000, 10000)
    mean_ns, std_ns = benchmark_object_create(6, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["create"] = f"{mean_ns/1e9:.6g} \xb1 {std_ns/1e9:.2g}"

    # ----- fetch ----- #
    n_iters = 1000
    print('\n' + '-' * 10 + " Benchmark fetching SharedObject " + '-' * 10)
    for object_type_idx in range(len(SharedObject._object_types)-1):
        mean_ns, std_ns = benchmark_object_fetch(object_type_idx, n_iters=n_iters)
        results[object_type_idx]["fetch"] = f"{mean_ns/1e9:.6g} \xb1 {std_ns/1e9:.2g}"

    dtype, shape = np.uint8, (480, 848, 3)
    mean_ns, std_ns = benchmark_object_fetch(6, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["fetch"] = f"{mean_ns/1e9:.6g} \xb1 {std_ns/1e9:.2g}"
    dtype, shape = np.float64, (720, 1280, 3)
    mean_ns, std_ns = benchmark_object_fetch(6, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["fetch"] = f"{mean_ns/1e9:.6g} \xb1 {std_ns/1e9:.2g}"
    n_iters = 20
    dtype, shape = np.float64, (10000, 10000)
    mean_ns, std_ns = benchmark_object_fetch(6, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["fetch"] = f"{mean_ns/1e9:.6g} \xb1 {std_ns/1e9:.2g}"

    # ----- assign ----- #
    n_iters = 1000
    print('\n' + '-' * 10 + " Benchmark assigning SharedObject " + '-' * 10)
    for object_type_idx in range(len(SharedObject._object_types)-1):
        mean_ns, std_ns = benchmark_object_assign(object_type_idx, n_iters=n_iters)
        results[object_type_idx]["assign"] = f"{mean_ns/1e9:.6g} \xb1 {std_ns/1e9:.2g}"

    dtype, shape = np.uint8, (480, 848, 3)
    mean_ns, std_ns = benchmark_object_assign(6, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["assign"] = f"{mean_ns/1e9:.6g} \xb1 {std_ns/1e9:.2g}"
    dtype, shape = np.float64, (720, 1280, 3)
    mean_ns, std_ns = benchmark_object_assign(6, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["assign"] = f"{mean_ns/1e9:.6g} \xb1 {std_ns/1e9:.2g}"
    n_iters = 50
    dtype, shape = np.float64, (10000, 10000)
    mean_ns, std_ns = benchmark_object_assign(6, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["assign"] = f"{mean_ns/1e9:.6g} \xb1 {std_ns/1e9:.2g}"

    # ----- create_ref ----- #
    n_iters = 1000
    print('\n' + '-' * 10 + " Benchmark creating from existing SharedObject " + '-' * 10)
    for object_type_idx in range(len(SharedObject._object_types)-1):
        mean_ns, std_ns = benchmark_object_create_ref(object_type_idx, n_iters=n_iters)
        results[object_type_idx]["create_ref"] = f"{mean_ns/1e9:.6g} \xb1 {std_ns/1e9:.2g}"

    dtype, shape = np.uint8, (480, 848, 3)
    mean_ns, std_ns = benchmark_object_create_ref(6, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["create_ref"] = f"{mean_ns/1e9:.6g} \xb1 {std_ns/1e9:.2g}"
    dtype, shape = np.float64, (720, 1280, 3)
    mean_ns, std_ns = benchmark_object_create_ref(6, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["create_ref"] = f"{mean_ns/1e9:.6g} \xb1 {std_ns/1e9:.2g}"
    n_iters = 50
    dtype, shape = np.float64, (10000, 10000)
    mean_ns, std_ns = benchmark_object_create_ref(6, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["create_ref"] = f"{mean_ns/1e9:.6g} \xb1 {std_ns/1e9:.2g}"

    # ----- 2 processes fetch/assign ----- #
    n_iters = 100
    print('\n' + '-' * 10 + " Benchmark fetching/assigning SharedObject with 2 processes " + '-' * 10)
    for object_type_idx in range(len(SharedObject._object_types)-1):
        mean_ns, std_ns = benchmark_object_2_proc_fetch_assign(object_type_idx, n_iters=n_iters)
        results[object_type_idx]["2 proc fetch_assign"] = f"{mean_ns/1e9:.6g} \xb1 {std_ns/1e9:.2g}"

    dtype, shape = np.uint8, (480, 848, 3)
    mean_ns, std_ns = benchmark_object_2_proc_fetch_assign(6, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["2 proc fetch_assign"] = f"{mean_ns/1e9:.6g} \xb1 {std_ns/1e9:.2g}"
    dtype, shape = np.float64, (720, 1280, 3)
    mean_ns, std_ns = benchmark_object_2_proc_fetch_assign(6, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["2 proc fetch_assign"] = f"{mean_ns/1e9:.6g} \xb1 {std_ns/1e9:.2g}"
    n_iters = 10
    dtype, shape = np.float64, (10000, 10000)
    mean_ns, std_ns = benchmark_object_2_proc_fetch_assign(6, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["2 proc fetch_assign"] = f"{mean_ns/1e9:.6g} \xb1 {std_ns/1e9:.2g}"

    # ----- 5 processes fetch/assign ----- #
    n_iters = 50
    print('\n' + '-' * 10 + " Benchmark fetching/assigning SharedObject with 5 processes " + '-' * 10)
    for object_type_idx in range(len(SharedObject._object_types)-1):
        mean_ns, std_ns = benchmark_object_5_proc_fetch_assign(object_type_idx, n_iters=n_iters)
        results[object_type_idx]["5 proc fetch_assign"] = f"{mean_ns/1e9:.6g} \xb1 {std_ns/1e9:.2g}"

    dtype, shape = np.uint8, (480, 848, 3)
    mean_ns, std_ns = benchmark_object_5_proc_fetch_assign(6, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["5 proc fetch_assign"] = f"{mean_ns/1e9:.6g} \xb1 {std_ns/1e9:.2g}"
    dtype, shape = np.float64, (720, 1280, 3)
    mean_ns, std_ns = benchmark_object_5_proc_fetch_assign(6, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["5 proc fetch_assign"] = f"{mean_ns/1e9:.6g} \xb1 {std_ns/1e9:.2g}"
    n_iters = 10
    dtype, shape = np.float64, (10000, 10000)
    mean_ns, std_ns = benchmark_object_5_proc_fetch_assign(6, n_iters=n_iters, dtype=dtype, shape=shape)
    results[f"ndarray {dtype} {shape}"]["5 proc fetch_assign"] = f"{mean_ns/1e9:.6g} \xb1 {std_ns/1e9:.2g}"

    # ----- print results table ----- #
    results_table = PrettyTable()
    results_table.field_names = (["Object Type / Duration for 1 obj (mean \xb1 std)"]
                                 + list(results[0].keys()))
    for idx, res_dict in results.items():
        results_table.add_row(
            [SharedObject._object_types[idx] if isinstance(idx, int) else idx]
            + [v_str for v_str in res_dict.values()]
        )
    print(results_table)
