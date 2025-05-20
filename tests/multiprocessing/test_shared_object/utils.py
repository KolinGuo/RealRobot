import random
import string
from typing import Union

import numpy as np
import pytest
from transforms3d.euler import euler2quat

from real_robot.utils.multiprocessing import SharedObject

NDARRAY_NBYTES_LIMIT = 20 * 1024**2  # 20 MiB


def create_random_ndarray(
    dtype: Union[SharedObject._np_dtypes],  # type: ignore
    shape: tuple[int, ...],
) -> np.ndarray:
    rng = np.random.default_rng()
    if np.issubdtype(dtype, np.bool_):
        data = rng.integers(2, size=shape, dtype=dtype)
    elif np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        min, max = info.min, info.max
        data = rng.integers(min, max + 1, size=shape, dtype=dtype)
    elif np.issubdtype(dtype, np.inexact):
        info = np.finfo(np.float32)  # cannot sample uniform for float128
        min, max = info.min, info.max
        data = rng.uniform(min, max, size=shape).astype(dtype)
    else:
        raise TypeError(f"Unknown numpy {dtype = }")
    return data


def create_random_object(object_type_idx: int) -> Union[SharedObject._object_types]:  # type: ignore
    rng = np.random.default_rng()

    if object_type_idx == SharedObject._object_types.index(None.__class__):
        return None
    elif object_type_idx == SharedObject._object_types.index(bool):
        return bool(random.randrange(2))
    elif object_type_idx == SharedObject._object_types.index(int):
        return random.randint(-9223372036854775808, 9223372036854775807)
    elif object_type_idx == SharedObject._object_types.index(float):
        return (
            random.uniform(-100, 100)
            if bool(random.randrange(2))
            else random.uniform(-1e307, 1e308)
        )
    elif object_type_idx == SharedObject._object_types.index(complex):
        return complex(
            random.uniform(-100, 100)
            if bool(random.randrange(2))
            else random.uniform(-1e307, 1e308),
            random.uniform(-100, 100)
            if bool(random.randrange(2))
            else random.uniform(-1e307, 1e308),
        )
    elif object_type_idx == 5:  # sapien.Pose
        pytest.importorskip("sapien")
        from sapien import Pose

        return Pose(
            p=rng.uniform(-10, 10, size=3),
            q=euler2quat(*rng.uniform([0, 0, 0], [np.pi * 2, np.pi, np.pi * 2])),
        )
    elif object_type_idx == SharedObject._object_types.index(str):  # str
        str_len = random.randrange(51)
        return "".join(random.choices(string.printable, k=str_len))
    elif object_type_idx == SharedObject._object_types.index(bytes):  # bytes
        bytes_len = random.randrange(51)
        return random.randbytes(bytes_len)
    elif object_type_idx == SharedObject._object_types.index(bytearray):  # bytearray
        bytes_len = random.randrange(51)
        return bytearray(random.randbytes(bytes_len))
    elif object_type_idx == SharedObject._object_types.index(np.ndarray):  # np.ndarray
        size = NDARRAY_NBYTES_LIMIT + 1
        while size > NDARRAY_NBYTES_LIMIT:
            ndim = random.randint(1, 5)
            shape = tuple(random.randint(1, 1000) for _ in range(ndim))
            dtype = random.choice(SharedObject._np_dtypes)
            size = dtype().itemsize * np.prod(shape, dtype=np.uint64)
        return create_random_ndarray(dtype, shape)  # type: ignore
    else:
        raise ValueError(f"Unknown {object_type_idx = }")


def check_object_equal(obj1: SharedObject, obj2: SharedObject, data=None):
    assert obj1.object_type_idx == obj2.object_type_idx
    if data is not None:
        assert type(data) is SharedObject._object_types[obj1.object_type_idx]

    if obj1.object_type_idx == SharedObject._object_types.index(None.__class__):
        assert obj1.fetch() is None and obj2.fetch() is None
    elif obj1.object_type_idx in [
        SharedObject._object_types.index(bool),
        SharedObject._object_types.index(int),
        SharedObject._object_types.index(float),
        SharedObject._object_types.index(complex),
        SharedObject._object_types.index(str),
        SharedObject._object_types.index(bytes),
        SharedObject._object_types.index(bytearray),
    ]:
        assert obj1.fetch() == obj2.fetch()
        if data is not None:
            assert obj1.fetch() == data
    elif obj1.object_type_idx == 5:  # sapien.Pose
        np.testing.assert_equal(
            obj1.fetch().__getstate__(), obj2.fetch().__getstate__()
        )
        if data is not None:
            np.testing.assert_equal(obj1.fetch().__getstate__(), data.__getstate__())
    elif obj1.object_type_idx == SharedObject._object_types.index(np.ndarray):
        np.testing.assert_equal(obj1.fetch(), obj2.fetch())
        if data is not None:
            np.testing.assert_equal(obj1.fetch(), data)
    else:
        raise ValueError(f"Unknown {obj1.object_type_idx = }")
