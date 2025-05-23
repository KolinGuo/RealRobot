import random
import string
from typing import Union

import numpy as np
import pytest
from transforms3d.euler import euler2quat

from real_robot.utils.multiprocessing import SharedObject
from real_robot.utils.multiprocessing.shared_object_metas import NP_DTYPES

NDARRAY_NBYTES_LIMIT = 20 * 1024**2  # 20 MiB


def create_random_ndarray(
    dtype: Union[NP_DTYPES],  # type: ignore
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
            p=rng.uniform(-10, 10, size=3),  # type: ignore
            q=euler2quat(*rng.uniform([0, 0, 0], [np.pi * 2, np.pi, np.pi * 2])),  # type: ignore
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
            dtype = random.choice(NP_DTYPES)
            size = dtype().itemsize * np.prod(shape, dtype=np.uint64)  # type: ignore
        return create_random_ndarray(dtype, shape)  # type: ignore
    else:
        raise ValueError(f"Unknown {object_type_idx = }")


def check_object_equal(
    obj1: SharedObject | Union[SharedObject._object_types],  # type: ignore
    obj2: SharedObject | Union[SharedObject._object_types],  # type: ignore
    data: Union[SharedObject._object_types] = None,  # type: ignore
):
    if isinstance(obj1, SharedObject):
        obj1 = obj1.fetch()
    if isinstance(obj2, SharedObject):
        obj2 = obj2.fetch()

    # Check object type
    assert type(obj1) is type(obj2)
    if data is not None:
        assert type(data) is type(obj1)

    # Check object value
    if isinstance(
        obj1, (None.__class__, bool, int, float, complex, str, bytes, bytearray)
    ):
        assert obj1 == obj2
        if data is not None:
            assert obj1 == data
    elif type(obj1).__name__ == "Pose" and type(obj1).__module__.startswith(
        "sapien"
    ):  # sapien.Pose
        np.testing.assert_equal(obj1.__getstate__(), obj2.__getstate__())
        if data is not None:
            np.testing.assert_equal(obj1.__getstate__(), data.__getstate__())
    elif isinstance(obj1, np.ndarray):
        np.testing.assert_equal(obj1, obj2)
        if data is not None:
            np.testing.assert_equal(obj1, data)
    elif isinstance(obj1, dict):
        for (key1, value1), (key2, value2) in zip(obj1.items(), obj2.items()):  # type: ignore
            check_object_equal(key1, key2)
            check_object_equal(value1, value2)
        if data is not None:
            for (key1, value1), (key2, value2) in zip(obj1.items(), data.items()):  # type: ignore
                check_object_equal(key1, key2)
                check_object_equal(value1, value2)
    else:
        raise ValueError(f"Unknown {obj1.object_type_idx = }")
