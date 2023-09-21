"""Unittests for real_robot.utils.multiprocessing.shared_object"""
import uuid
import random
import string
from time import perf_counter
import multiprocessing as mp
from typing import Union, Tuple

import numpy as np

from real_robot.utils.multiprocessing import SharedObject
from real_robot.utils.logger import get_logger

_ctx = mp.get_context("forkserver" if "forkserver" in mp.get_all_start_methods()
                      else "spawn")
_logger = get_logger("Timer", fmt="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
NDARRAY_NBYTES_LIMIT = 20 * 1024**2  # 20 MiB


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


def create_random_object(object_type_idx: int) -> Union[SharedObject._object_types]:
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
        str_len = random.randrange(51)
        return ''.join(random.choices(string.printable, k=str_len))
    elif object_type_idx == 5:  # bytes
        bytes_len = random.randrange(51)
        return random.randbytes(bytes_len)
    elif object_type_idx == 6:  # np.ndarray
        size = NDARRAY_NBYTES_LIMIT + 1
        while size > NDARRAY_NBYTES_LIMIT:
            ndim = random.randint(1, 5)
            shape = tuple(random.randint(1, 1000) for _ in range(ndim))
            dtype = random.choice(SharedObject._np_dtypes)
            size = dtype().itemsize * np.prod(shape, dtype=np.uint64)
        return create_random_ndarray(dtype, shape)
    else:
        raise ValueError(f"Unknown {object_type_idx = }")


def check_object_equal(obj1: SharedObject, obj2: SharedObject, data=None):
    assert obj1.object_type_idx == obj2.object_type_idx
    if data is not None:
        assert type(data) == SharedObject._object_types[obj1.object_type_idx]

    if obj1.object_type_idx == 0:  # None.__class__
        assert obj1.fetch() is None and obj2.fetch() is None
    elif obj1.object_type_idx in [1, 2, 3, 4, 5]:  # bool, int, float, str, bytes
        assert obj1.fetch() == obj2.fetch()
        if data is not None:
            assert obj1.fetch() == data
    elif obj1.object_type_idx == 6:  # np.ndarray
        np.testing.assert_equal(obj1.fetch(), obj2.fetch())
        if data is not None:
            np.testing.assert_equal(obj1.fetch(), data)
    else:
        raise ValueError(f"Unknown {obj1.object_type_idx = }")


class TestCreate:
    """Test creating SharedObject"""

    def test_None(self):
        so = SharedObject(uuid.uuid4().hex, data=None)
        assert so.fetch() is None
        assert not so.modified

    def test_bool(self):
        so = SharedObject(uuid.uuid4().hex, data=False)
        assert so.fetch() is False
        assert not so.modified
        so = SharedObject(uuid.uuid4().hex, data=True)
        assert so.fetch() is True
        assert not so.modified

    def test_int(self):
        for _ in range(500):
            data = random.randint(-9223372036854775808, 9223372036854775807)
            so = SharedObject(uuid.uuid4().hex, data=data)
            assert so.fetch() == data
            assert not so.modified

    def test_float(self):
        for _ in range(500):
            data = random.uniform(-100, 100)
            so = SharedObject(uuid.uuid4().hex, data=data)
            assert so.fetch() == data
            assert not so.modified

        for _ in range(500):
            data = random.uniform(-1e307, 1e308)
            so = SharedObject(uuid.uuid4().hex, data=data)
            assert so.fetch() == data
            assert not so.modified

    def test_str(self):
        data = ""
        so = SharedObject(uuid.uuid4().hex, data=data)
        assert so.fetch() == data
        assert not so.modified

        for _ in range(500):
            str_len = random.randrange(100)
            data = ''.join(random.choices(string.printable, k=str_len))
            so = SharedObject(uuid.uuid4().hex, data=data)
            assert so.fetch() == data
            assert not so.modified

    def test_bytes(self):
        data = b""
        so = SharedObject(uuid.uuid4().hex, data=data)
        assert so.fetch() == data
        assert not so.modified

        data = b"\x00"
        so = SharedObject(uuid.uuid4().hex, data=data)
        assert so.fetch() == data
        assert not so.modified

        data = b"asdlkj123\x01asd\x00\x00"
        so = SharedObject(uuid.uuid4().hex, data=data)
        assert so.fetch() == data
        assert not so.modified

        for _ in range(500):
            bytes_len = random.randrange(100)
            data = random.randbytes(bytes_len)
            so = SharedObject(uuid.uuid4().hex, data=data)
            assert so.fetch() == data
            assert not so.modified

    def test_ndarray(self):
        data = np.ones(1)
        so = SharedObject(uuid.uuid4().hex, data=data)
        data_fetched = so.fetch()
        np.testing.assert_equal(data_fetched, data)
        assert data_fetched.flags.owndata
        assert data_fetched.flags.writeable
        assert not so.modified

        for _ in range(500):
            size = NDARRAY_NBYTES_LIMIT + 1
            while size > NDARRAY_NBYTES_LIMIT:
                ndim = random.randint(1, 5)
                shape = tuple(random.randint(1, 1000) for _ in range(ndim))
                dtype = random.choice(SharedObject._np_dtypes)
                size = dtype().itemsize * np.prod(shape, dtype=np.uint64)

            data = create_random_ndarray(dtype, shape)

            so = SharedObject(uuid.uuid4().hex, data=data)
            data_fetched = so.fetch()
            np.testing.assert_equal(data_fetched, data)
            assert data_fetched.flags.owndata
            assert data_fetched.flags.writeable
            assert not so.modified


class TestFetch:
    """Test fetching from SharedObject returns a copy"""

    def test_None(self):
        so = SharedObject(uuid.uuid4().hex, data=None)
        assert so.fetch(lambda x: type(x)) is None.__class__
        assert so.fetch(lambda x: 10) == 10
        assert so.fetch() is None

    def test_bool(self):
        so = SharedObject(uuid.uuid4().hex, data=False)
        v = random.uniform(-100, 100)
        assert so.fetch(lambda x: x+v) == v
        assert so.fetch(lambda x: x*v) == 0.0
        assert so.fetch(lambda x: not x) is True
        assert so.fetch() is False

        so = SharedObject(uuid.uuid4().hex, data=True)
        v = random.uniform(-100, 100)
        assert so.fetch(lambda x: x+v) == v+1
        assert so.fetch(lambda x: x*v) == v
        assert so.fetch(lambda x: not x) is False
        assert so.fetch() is True

    def test_int(self):
        data = random.randint(-999999, 999999)
        so = SharedObject(uuid.uuid4().hex, data=data)

        v = random.randint(-100, 100)
        assert so.fetch(lambda x: x+v) == data+v
        assert so.fetch(lambda x: x*v) == data*v
        v = random.uniform(-100, 100)
        assert so.fetch(lambda x: x+v) == data+v
        assert so.fetch(lambda x: x*v) == data*v
        assert so.fetch() == data

    def test_float(self):
        data = random.uniform(-100000, 100000)
        so = SharedObject(uuid.uuid4().hex, data=data)

        v = random.randint(-100, 100)
        assert so.fetch(lambda x: x+v) == data+v
        assert so.fetch(lambda x: x*v) == data*v
        v = random.uniform(-100, 100)
        assert so.fetch(lambda x: x+v) == data+v
        assert so.fetch(lambda x: x*v) == data*v
        assert so.fetch() == data

    def test_str(self):
        data = ""
        so = SharedObject(uuid.uuid4().hex, data=data)
        assert so.fetch(lambda x: len(x)) == len(data)
        assert so.fetch() == data

        str_len = random.randrange(100)
        data = ''.join(random.choices(string.printable, k=str_len))
        so = SharedObject(uuid.uuid4().hex, data=data)

        assert so.fetch(lambda x: len(x)) == len(data)
        assert so.fetch(lambda x: x[0]) == data[0]
        assert so.fetch(lambda x: x[:20]) == data[:20]
        assert so.fetch() == data

    def test_bytes(self):
        data = b""
        so = SharedObject(uuid.uuid4().hex, data=data)
        assert so.fetch(lambda x: len(x)) == len(data)
        assert so.fetch() == data

        data = b"\x00"
        so = SharedObject(uuid.uuid4().hex, data=data)
        assert so.fetch(lambda x: len(x)) == len(data)
        assert so.fetch() == data

        data = b"asdlkj123\x01asd\x00\x00"
        so = SharedObject(uuid.uuid4().hex, data=data)
        assert so.fetch(lambda x: len(x)) == len(data)
        assert so.fetch(lambda x: x[0]) == data[0]
        assert so.fetch(lambda x: x[:20]) == data[:20]
        assert so.fetch() == data

        bytes_len = random.randrange(100)
        data = random.randbytes(bytes_len)
        so = SharedObject(uuid.uuid4().hex, data=data)
        assert so.fetch(lambda x: len(x)) == len(data)
        assert so.fetch(lambda x: x[0]) == data[0]
        assert so.fetch(lambda x: x[:20]) == data[:20]
        assert so.fetch() == data

    def test_ndarray_fn_None(self):
        data = np.random.rand(480, 848, 3)
        so = SharedObject(uuid.uuid4().hex, data=data)
        data_fetched = so.fetch()
        data_fetched.fill(123)
        np.testing.assert_equal(so.fetch(), data)

    def test_ndarray_fn_easy_op(self):
        data = np.random.rand(480, 848, 3)
        so = SharedObject(uuid.uuid4().hex, data=data)

        # add scalar
        v = random.uniform(-100, 100)
        data_fetched = so.fetch(lambda x: x + v)
        np.testing.assert_equal(data_fetched, data + v)
        assert data_fetched.flags.owndata
        assert data_fetched.flags.writeable
        data_fetched.fill(123)
        np.testing.assert_equal(so.fetch(), data)

        # add vector
        v = [random.uniform(-100, 100) for _ in range(3)]
        data_fetched = so.fetch(lambda x: x + v)
        np.testing.assert_equal(data_fetched, data + v)
        assert data_fetched.flags.owndata
        assert data_fetched.flags.writeable
        data_fetched.fill(123)
        np.testing.assert_equal(so.fetch(), data)

        # matmul vector
        v = [random.uniform(-100, 100) for _ in range(3)]
        data_fetched = so.fetch(lambda x: x @ v)
        np.testing.assert_equal(data_fetched, data @ v)
        assert data_fetched.flags.owndata
        assert data_fetched.flags.writeable
        data_fetched.fill(123)
        np.testing.assert_equal(so.fetch(), data)

        # power
        v = random.randint(2, 5)
        data_fetched = so.fetch(lambda x: x ** v)
        np.testing.assert_equal(data_fetched, data ** v)
        assert data_fetched.flags.owndata
        assert data_fetched.flags.writeable
        data_fetched.fill(123)
        np.testing.assert_equal(so.fetch(), data)

        # sort
        data_fetched = so.fetch(lambda x: np.sort(x))
        np.testing.assert_equal(data_fetched, np.sort(data))
        assert data_fetched.flags.owndata
        assert data_fetched.flags.writeable
        data_fetched.fill(123)
        np.testing.assert_equal(so.fetch(), data)

        # sum
        data_fetched = so.fetch(lambda x: x.sum())
        np.testing.assert_equal(data_fetched, data.sum())
        assert data_fetched.flags.owndata

    def test_ndarray_fn_slice(self):
        data = np.random.rand(480, 848, 3)
        so = SharedObject(uuid.uuid4().hex, data=data)

        # slice
        data_fetched = so.fetch(lambda x: x[..., 0])
        np.testing.assert_equal(data_fetched, data[..., 0])
        assert data_fetched.flags.owndata
        assert data_fetched.flags.writeable
        data_fetched.fill(123)
        np.testing.assert_equal(so.fetch(), data)

        # mask
        data_fetched = so.fetch(lambda x: x[..., [True, False, True]])
        np.testing.assert_equal(data_fetched, data[..., [True, False, True]])
        assert data_fetched.flags.owndata
        assert data_fetched.flags.writeable
        data_fetched.fill(123)
        np.testing.assert_equal(so.fetch(), data)

        # mask
        mask = np.random.randint(2, size=data.shape, dtype=bool)
        data_fetched = so.fetch(lambda x: x[mask])
        np.testing.assert_equal(data_fetched, data[mask])
        assert data_fetched.flags.owndata
        assert data_fetched.flags.writeable
        data_fetched.fill(123)
        np.testing.assert_equal(so.fetch(), data)

    def test_ndarray_fn_slice_and_op(self):
        data = np.random.rand(480, 848, 3)
        so = SharedObject(uuid.uuid4().hex, data=data)

        # slice
        v = random.uniform(-100, 100)
        data_fetched = so.fetch(lambda x: x[..., 0] + v)
        np.testing.assert_equal(data_fetched, data[..., 0] + v)
        assert data_fetched.flags.owndata
        assert data_fetched.flags.writeable
        data_fetched.fill(123)
        np.testing.assert_equal(so.fetch(), data)

        # mask
        v = random.uniform(-100, 100)
        data_fetched = so.fetch(lambda x: x[..., [True, False, True]] * v)
        np.testing.assert_equal(data_fetched, data[..., [True, False, True]] * v)
        assert data_fetched.flags.owndata
        assert data_fetched.flags.writeable
        data_fetched.fill(123)
        np.testing.assert_equal(so.fetch(), data)

        # mask
        v = random.uniform(-100, 100)
        mask = np.random.randint(2, size=data.shape, dtype=bool)
        data_fetched = so.fetch(lambda x: x[mask] + v)
        np.testing.assert_equal(data_fetched, data[mask] + v)
        assert data_fetched.flags.owndata
        assert data_fetched.flags.writeable
        data_fetched.fill(123)
        np.testing.assert_equal(so.fetch(), data)

    def test_ndarray_fn_modify_inplace(self):
        data = np.random.rand(480, 848, 3)
        so = SharedObject(uuid.uuid4().hex, data=data)

        # modify
        def inplace_add(x):
            x[..., 0] += 1
            return x
        try:
            _ = so.fetch(inplace_add)
        except ValueError as e:
            print(e)
        else:
            assert False, "Should raise ValueError when attempting to modify in fetch"
        np.testing.assert_equal(so.fetch(), data)

        # inplace sort
        try:
            _ = so.fetch(lambda x: x.sort())
        except ValueError as e:
            print(e)
        else:
            assert False, "Should raise ValueError when attempting to modify in fetch"
        np.testing.assert_equal(so.fetch(), data)


class TestAssign:
    """Test assigning SharedObject"""

    def test_None(self):
        so = SharedObject(uuid.uuid4().hex, data=None)
        assert so.fetch() is None
        assert not so.modified

        so.assign(None)
        assert not so.modified
        assert so.fetch() is None

    def test_bool(self):
        so = SharedObject(uuid.uuid4().hex, data=False)
        assert so.fetch() is False
        assert not so.modified
        so.assign(True)
        assert not so.modified
        assert so.fetch() is True
        so.assign(False)
        assert not so.modified
        assert so.fetch() is False

        so = SharedObject(uuid.uuid4().hex, data=True)
        assert so.fetch() is True
        assert not so.modified
        so.assign(False)
        assert not so.modified
        assert so.fetch() is False

    def test_int(self):
        data = random.randint(-999999, 999999)
        so = SharedObject(uuid.uuid4().hex, data=data)
        assert so.fetch() == data
        assert not so.modified

        for _ in range(500):
            data = random.randint(-9223372036854775808, 9223372036854775807)
            so.assign(data)
            assert not so.modified
            assert so.fetch() == data

    def test_float(self):
        data = random.uniform(-100, 100)
        so = SharedObject(uuid.uuid4().hex, data=data)
        assert so.fetch() == data
        assert not so.modified

        for _ in range(500):
            data = random.uniform(-100, 100)
            so.assign(data)
            assert not so.modified
            assert so.fetch() == data

        for _ in range(500):
            data = random.uniform(-1e307, 1e308)
            so.assign(data)
            assert not so.modified
            assert so.fetch() == data

    def test_str(self):
        data = ""
        so = SharedObject(uuid.uuid4().hex, data=data, init_size=200)
        assert so.fetch() == data
        assert not so.modified

        for _ in range(500):
            str_len = random.randrange(100)
            data = ''.join(random.choices(string.printable, k=str_len))
            so.assign(data)
            assert not so.modified
            assert so.fetch() == data

    def test_str_overflow(self):
        data = ""
        so = SharedObject(uuid.uuid4().hex, data=data)
        assert so.fetch() == data
        assert not so.modified

        str_len = 50
        data = ''.join(random.choices(string.printable, k=str_len))
        so.assign(data)
        assert so.fetch() == data
        assert not so.modified

        str_len = 51
        data = ''.join(random.choices(string.printable, k=str_len))
        try:
            so.assign(data)
        except BufferError as e:
            print(e)
        else:
            assert False, "Should raise BufferError when assigning causes overflow"

        for _ in range(10):
            str_len = random.randrange(51, 100)
            data = ''.join(random.choices(string.printable, k=str_len))
            try:
                so.assign(data)
            except BufferError as e:
                print(e)
            else:
                assert False, "Should raise BufferError when assigning causes overflow"

    def test_bytes(self):
        data = b""
        so = SharedObject(uuid.uuid4().hex, data=data, init_size=200)
        assert so.fetch() == data
        assert not so.modified

        data = b"\x00"
        so.assign(data)
        assert so.fetch() == data
        assert not so.modified

        data = b"asdlkj123\x01asd\x00\x00"
        so.assign(data)
        assert so.fetch() == data
        assert not so.modified

        for _ in range(500):
            bytes_len = random.randrange(100)
            data = random.randbytes(bytes_len)
            so.assign(data)
            assert not so.modified
            assert so.fetch() == data

    def test_bytes_overflow(self):
        data = b""
        so = SharedObject(uuid.uuid4().hex, data=data)
        assert so.fetch() == data
        assert not so.modified

        bytes_len = 50
        data = random.randbytes(bytes_len)
        so.assign(data)
        assert so.fetch() == data
        assert not so.modified

        bytes_len = 51
        data = random.randbytes(bytes_len)
        try:
            so.assign(data)
        except BufferError as e:
            print(e)
        else:
            assert False, "Should raise BufferError when assigning causes overflow"

        for _ in range(10):
            bytes_len = random.randrange(51, 100)
            data = random.randbytes(bytes_len)
            try:
                so.assign(data)
            except BufferError as e:
                print(e)
            else:
                assert False, "Should raise BufferError when assigning causes overflow"

    def test_ndarray(self):
        data = np.ones(1)
        so = SharedObject(uuid.uuid4().hex, data=data)
        data_fetched = so.fetch()
        np.testing.assert_equal(data_fetched, data)
        assert data_fetched.flags.owndata
        assert data_fetched.flags.writeable
        assert not so.modified

        data = np.ones(1) * random.uniform(-100, 100)
        so.assign(data)
        assert not so.modified
        data_fetched = so.fetch()
        np.testing.assert_equal(data_fetched, data)
        assert data_fetched.flags.owndata
        assert data_fetched.flags.writeable

        for _ in range(100):
            size = NDARRAY_NBYTES_LIMIT + 1
            while size > NDARRAY_NBYTES_LIMIT:
                ndim = random.randint(1, 5)
                shape = tuple(random.randint(1, 1000) for _ in range(ndim))
                dtype = random.choice(SharedObject._np_dtypes)
                size = dtype().itemsize * np.prod(shape, dtype=np.uint64)

            for i in range(10):
                data = create_random_ndarray(dtype, shape)

                if i == 0:
                    so = SharedObject(uuid.uuid4().hex, data=data)
                else:
                    so.assign(data)
                assert not so.modified
                data_fetched = so.fetch()
                np.testing.assert_equal(data_fetched, data)
                assert data_fetched.flags.owndata
                assert data_fetched.flags.writeable

    def test_ndarray_changed_meta(self):
        data = np.random.randint(256, size=(480, 848, 3), dtype=np.uint8)
        so = SharedObject(uuid.uuid4().hex, data=data)
        data_fetched = so.fetch()
        np.testing.assert_equal(data_fetched, data)
        data = np.ones((480, 848, 3))
        try:
            so.assign(data)
        except BufferError as e:
            print(e)
        else:
            assert False, "Should raise BufferError when dtype is changed"

        for _ in range(10):
            size = NDARRAY_NBYTES_LIMIT + 1
            while size > NDARRAY_NBYTES_LIMIT:
                ndim = random.randint(1, 5)
                shape = tuple(random.randint(1, 1000) for _ in range(ndim))
                dtype = random.choice(SharedObject._np_dtypes)
                size = dtype().itemsize * np.prod(shape, dtype=np.uint64)

            data = create_random_ndarray(dtype, shape)

            so = SharedObject(uuid.uuid4().hex, data=data)

            # Changed np dtype
            for new_dtype in SharedObject._np_dtypes:
                if new_dtype != dtype:
                    data = create_random_ndarray(new_dtype, shape)
                    try:
                        so.assign(data)
                    except BufferError as e:
                        print(e)
                    else:
                        assert False, "Should raise BufferError when dtype is changed"

            # Changed ndim
            for i in range(5):
                new_shape = shape + (1,) * (i + 1)
                data = create_random_ndarray(dtype, new_shape)
                try:
                    so.assign(data)
                except BufferError as e:
                    print(e)
                else:
                    assert False, "Should raise BufferError when ndim is changed"

            # Changed shape
            for i in range(5):
                new_shape = shape[:-1] + (shape[-1] + i+1,)
                data = create_random_ndarray(dtype, new_shape)
                try:
                    so.assign(data)
                except BufferError as e:
                    print(e)
                else:
                    assert False, "Should raise BufferError when shape is changed"

    def test_changed_object_type(self):
        for object_type_idx in range(len(SharedObject._object_types)):
            data = create_random_object(object_type_idx)
            so = SharedObject(uuid.uuid4().hex, data=data)

            for new_object_type_idx in range(len(SharedObject._object_types)):
                if object_type_idx != new_object_type_idx:
                    new_data = create_random_object(new_object_type_idx)
                    try:
                        so.assign(new_data)
                    except BufferError as e:
                        print(e)
                    else:
                        assert False, "Should raise BufferError on changed object type"


class TestMultiSharedObject:
    """Test multiple SharedObject"""

    def test_two_instances(self):
        for object_type_idx in range(len(SharedObject._object_types)):
            data = create_random_object(object_type_idx)
            so = SharedObject(uuid.uuid4().hex, data=data)
            so2 = SharedObject(so.name)
            assert not so.modified
            assert not so2.modified
            check_object_equal(so, so2, data)

            if object_type_idx == 6:  # np.ndarray
                new_data = create_random_ndarray(data.dtype, data.shape)
            else:
                new_data = create_random_object(object_type_idx)
            so2.assign(new_data)
            assert so.modified
            assert not so2.modified
            check_object_equal(so, so2, new_data)

            if object_type_idx == 6:  # np.ndarray
                new_data = create_random_ndarray(data.dtype, data.shape)
            else:
                new_data = create_random_object(object_type_idx)
            so.assign(new_data)
            assert not so.modified
            assert so2.modified
            check_object_equal(so, so2, new_data)

            so.close()
            so = SharedObject(so.name)

            if object_type_idx == 6:  # np.ndarray
                new_data = create_random_ndarray(data.dtype, data.shape)
            else:
                new_data = create_random_object(object_type_idx)
            so2.assign(new_data)
            assert so.modified
            assert not so2.modified
            check_object_equal(so, so2, new_data)

            if object_type_idx == 6:  # np.ndarray
                new_data = create_random_ndarray(data.dtype, data.shape)
            else:
                new_data = create_random_object(object_type_idx)
            so.assign(new_data)
            assert not so.modified
            assert so2.modified
            check_object_equal(so, so2, new_data)

            del so2
            so2 = SharedObject(so.name)

            if object_type_idx == 6:  # np.ndarray
                new_data = create_random_ndarray(data.dtype, data.shape)
            else:
                new_data = create_random_object(object_type_idx)
            so2.assign(new_data)
            assert so.modified
            assert not so2.modified
            check_object_equal(so, so2, new_data)

            if object_type_idx == 6:  # np.ndarray
                new_data = create_random_ndarray(data.dtype, data.shape)
            else:
                new_data = create_random_object(object_type_idx)
            so.assign(new_data)
            assert not so.modified
            assert so2.modified
            check_object_equal(so, so2, new_data)

    def test_five_instances(self):
        for object_type_idx in range(len(SharedObject._object_types)):
            data = create_random_object(object_type_idx)
            so = SharedObject(uuid.uuid4().hex, data=data)
            assert not so.modified
            sos = [SharedObject(so.name) for _ in range(4)]
            for so2 in sos:
                assert not so2.modified
                check_object_equal(so, so2, data)

            if object_type_idx == 6:  # np.ndarray
                new_data = create_random_ndarray(data.dtype, data.shape)
            else:
                new_data = create_random_object(object_type_idx)
            so.assign(new_data)
            assert not so.modified
            for so2 in sos:
                assert so2.modified
                check_object_equal(so, so2, new_data)

            for so2 in sos:
                if object_type_idx == 6:  # np.ndarray
                    new_data = create_random_ndarray(data.dtype, data.shape)
                else:
                    new_data = create_random_object(object_type_idx)
                so2.assign(new_data)
                assert so.modified
                assert not so2.modified
                check_object_equal(so, so2, new_data)


class TestModified:
    """Test so.modified with multiple SharedObject"""

    def test_two_instances(self):
        for object_type_idx in range(len(SharedObject._object_types)):
            data = create_random_object(object_type_idx)
            so = SharedObject(uuid.uuid4().hex, data=data)
            so2 = SharedObject(so.name)
            assert not so.modified
            assert not so2.modified
            check_object_equal(so, so2, data)

            if object_type_idx == 6:  # np.ndarray
                new_data = create_random_ndarray(data.dtype, data.shape)
            else:
                new_data = create_random_object(object_type_idx)
            so2.assign(new_data)
            assert so.modified
            assert not so2.modified
            so.fetch()
            assert not so.modified
            so2.fetch()
            assert not so2.modified
            check_object_equal(so, so2, new_data)

            if object_type_idx == 6:  # np.ndarray
                new_data = create_random_ndarray(data.dtype, data.shape)
            else:
                new_data = create_random_object(object_type_idx)
            so.assign(new_data)
            assert not so.modified
            assert so2.modified
            so2.fetch()
            assert not so2.modified
            so.fetch()
            assert not so.modified
            check_object_equal(so, so2, new_data)

    def test_five_instances(self):
        for object_type_idx in range(len(SharedObject._object_types)):
            data = create_random_object(object_type_idx)
            so = SharedObject(uuid.uuid4().hex, data=data)
            assert not so.modified
            sos = [SharedObject(so.name) for _ in range(4)]
            for so2 in sos:
                assert not so2.modified
                check_object_equal(so, so2, data)

            if object_type_idx == 6:  # np.ndarray
                new_data = create_random_ndarray(data.dtype, data.shape)
            else:
                new_data = create_random_object(object_type_idx)
            so.assign(new_data)
            assert not so.modified
            so.fetch()
            assert not so.modified
            for so2 in sos:
                assert so2.modified
                so2.fetch()
                assert not so2.modified
                check_object_equal(so, so2, new_data)

            for so2 in sos:
                if object_type_idx == 6:  # np.ndarray
                    new_data = create_random_ndarray(data.dtype, data.shape)
                else:
                    new_data = create_random_object(object_type_idx)
                so2.assign(new_data)
                assert not so2.modified
                assert so.modified
                so.fetch()
                assert not so.modified
                check_object_equal(so, so2, new_data)


class TestTrigger:
    """Test so.trigger() / so.triggered"""

    def test_trigger(self):
        so = SharedObject(uuid.uuid4().hex)
        assert not so.triggered
        so.trigger()
        assert so.triggered
        assert not so.triggered
        assert not so.triggered
        so.trigger()
        so.trigger()
        assert so.triggered
        assert not so.triggered
        assert not so.triggered

    def test_two_instances(self):
        for object_type_idx in range(len(SharedObject._object_types)):
            data = create_random_object(object_type_idx)
            so = SharedObject(uuid.uuid4().hex, data=data)
            so2 = SharedObject(so.name)
            assert not so.triggered
            assert not so2.triggered
            check_object_equal(so, so2, data)

            if object_type_idx == 6:  # np.ndarray
                new_data = create_random_ndarray(data.dtype, data.shape)
            else:
                new_data = create_random_object(object_type_idx)
            so2.trigger()
            assert so2.triggered
            assert not so2.triggered
            assert so.triggered
            assert not so.triggered
            so2.assign(new_data)
            assert so.triggered
            assert not so2.triggered
            assert not so.triggered
            assert not so2.triggered
            check_object_equal(so, so2, new_data)

            if object_type_idx == 6:  # np.ndarray
                new_data = create_random_ndarray(data.dtype, data.shape)
            else:
                new_data = create_random_object(object_type_idx)
            so.trigger()
            assert so.triggered
            assert not so.triggered
            assert so2.triggered
            assert not so2.triggered
            so.assign(new_data)
            assert not so.triggered
            assert so2.triggered
            assert not so.triggered
            assert not so2.triggered
            check_object_equal(so, so2, new_data)

    def test_five_instances(self):
        for object_type_idx in range(len(SharedObject._object_types)):
            data = create_random_object(object_type_idx)
            so = SharedObject(uuid.uuid4().hex, data=data)
            assert not so.triggered
            sos = [SharedObject(so.name) for _ in range(4)]
            for so2 in sos:
                assert not so2.triggered
                check_object_equal(so, so2, data)

            if object_type_idx == 6:  # np.ndarray
                new_data = create_random_ndarray(data.dtype, data.shape)
            else:
                new_data = create_random_object(object_type_idx)
            so.trigger()
            assert so.triggered
            assert not so.triggered
            for so2 in sos:
                assert so2.triggered
                assert not so2.triggered
            so.assign(new_data)
            assert not so.triggered
            assert not so.triggered
            for so2 in sos:
                assert so2.triggered
                assert not so2.triggered
                check_object_equal(so, so2, new_data)

            for so2 in sos:
                if object_type_idx == 6:  # np.ndarray
                    new_data = create_random_ndarray(data.dtype, data.shape)
                else:
                    new_data = create_random_object(object_type_idx)
                so2.trigger()
                assert so2.triggered
                assert not so2.triggered
                assert so.triggered
                assert not so.triggered
                so2.assign(new_data)
                assert so.triggered
                assert not so2.triggered
                assert not so.triggered
                assert not so2.triggered
                check_object_equal(so, so2, new_data)


class TestMultiProcess:
    """Test multiple processes"""

    @staticmethod
    def child_test_race_condition_with_extra_bool():
        # NOTE:
        # For processes that are always waiting for a massive SharedObject (e.g., np.ndarray),
        # it's better to add tiny delay to avoid starving processes that are assigning to it.
        # Even better, use a bool to indicate whether the data is updated yet and
        #   then only fetching the update flag inside the fetching processes to avoid this.
        so_data = SharedObject("data")
        so_data_updated = SharedObject("data_updated")
        so_result = SharedObject("result")
        so_joined = SharedObject("joined")

        while True:
            if so_joined.fetch():
                break

            if so_data_updated.fetch():
                res = float(so_data.fetch(lambda x: x.sum()))
                # res = float(so_data.np_ndarray.sum())  # Not protected by lock
                # print(f"[Child] {res} {perf_counter()}", flush=True)
                so_result.assign(res)
                so_data_updated.assign(False)

    def test_2_proc_race_condition_with_extra_bool(self):
        data = np.ones((10000, 10000))
        so_data = SharedObject("data", data=data)
        so_data_updated = SharedObject("data_updated", data=False)
        so_result = SharedObject("result", data=0.0)
        so_joined = SharedObject("joined", data=False)

        results = []
        n_iters = 10
        procs = [_ctx.Process(target=self.child_test_race_condition_with_extra_bool,
                              args=()) for _ in range(n_iters)]
        start_time = perf_counter()
        for i in range(n_iters):
            data = np.ones((10000, 10000))
            so_joined.assign(False)
            so_data_updated.assign(False)

            procs[i].start()
            for _ in range(5):
                data += 1
                so_data.assign(data)
                # so_data.np_ndarray[:] = data  # Not protected by lock
                # print("[Main]", data[0], flush=True)
                so_data_updated.assign(True)
                result = so_result.fetch()

                results.append(result)

            so_joined.assign(True)
            procs[i].join()
        _logger.info(f"test: Took {perf_counter() - start_time:.3f} seconds")

        print(results, flush=True)
        assert not np.any(np.array(results) % data.size), results

        so_data.unlink()
        so_data_updated.unlink()
        so_result.unlink()
        so_joined.unlink()

    def test_5_proc_race_condition_with_extra_bool(self):
        data = np.ones((10000, 10000))
        so_data = SharedObject("data", data=data)
        so_data_updated = SharedObject("data_updated", data=False)
        so_result = SharedObject("result", data=0.0)
        so_joined = SharedObject("joined", data=False)

        results = []
        n_iters = 10
        procs = [_ctx.Process(target=self.child_test_race_condition_with_extra_bool,
                              args=()) for _ in range(n_iters*5)]
        start_time = perf_counter()
        for i in range(n_iters):
            data = np.ones((10000, 10000))
            so_joined.assign(False)
            so_data_updated.assign(False)

            [proc.start() for proc in procs[5*i:5*(i+1)]]
            for _ in range(5):
                data += 1
                so_data.assign(data)
                # so_data.np_ndarray[:] = data  # Not protected by lock
                # print("[Main]", data[0], flush=True)
                so_data_updated.assign(True)
                result = so_result.fetch()

                results.append(result)

            so_joined.assign(True)
            [proc.join() for proc in procs[5*i:5*(i+1)]]
        _logger.info(f"test: Took {perf_counter() - start_time:.3f} seconds")

        print(results, flush=True)
        assert not np.any(np.array(results) % data.size), results

        so_data.unlink()
        so_data_updated.unlink()
        so_result.unlink()
        so_joined.unlink()

    @staticmethod
    def child_test_race_condition_with_modified():
        # NOTE:
        # For processes that are always waiting for a massive SharedObject (e.g., np.ndarray),
        # it's best to use so.modified to check whether the data is updated yet.
        so_data = SharedObject("data")
        so_result = SharedObject("result")
        so_joined = SharedObject("joined")

        while True:
            if so_joined.fetch():
                break

            if so_data.modified:
                res = float(so_data.fetch(lambda x: x.sum()))
                # res = float(so_data.np_ndarray.sum())  # Not protected by lock
                # print(f"[Child] {res} {perf_counter()}", flush=True)
                so_result.assign(res)

    def test_2_proc_race_condition_with_modified(self):
        data = np.ones((10000, 10000))
        so_data = SharedObject("data", data=data)
        so_result = SharedObject("result", data=0.0)
        so_joined = SharedObject("joined", data=False)

        results = []
        n_iters = 10
        procs = [_ctx.Process(target=self.child_test_race_condition_with_modified,
                              args=()) for _ in range(n_iters)]
        start_time = perf_counter()
        for i in range(n_iters):
            data = np.ones((10000, 10000))
            so_joined.assign(False)

            procs[i].start()
            for _ in range(5):
                data += 1
                so_data.assign(data)
                # so_data.np_ndarray[:] = data  # Not protected by lock
                # print("[Main]", data[0], flush=True)
                result = so_result.fetch()

                results.append(result)

            so_joined.assign(True)
            procs[i].join()
        _logger.info(f"test: Took {perf_counter() - start_time:.3f} seconds")

        print(results, flush=True)
        assert not np.any(np.array(results) % data.size), results

        so_data.unlink()
        so_result.unlink()
        so_joined.unlink()

    def test_5_proc_race_condition_with_modified(self):
        data = np.ones((10000, 10000))
        so_data = SharedObject("data", data=data)
        so_result = SharedObject("result", data=0.0)
        so_joined = SharedObject("joined", data=False)

        results = []
        n_iters = 10
        procs = [_ctx.Process(target=self.child_test_race_condition_with_modified,
                              args=()) for _ in range(n_iters*5)]
        start_time = perf_counter()
        for i in range(n_iters):
            data = np.ones((10000, 10000))
            so_joined.assign(False)

            [proc.start() for proc in procs[5*i:5*(i+1)]]
            for _ in range(5):
                data += 1
                so_data.assign(data)
                # so_data.np_ndarray[:] = data  # Not protected by lock
                # print("[Main]", data[0], flush=True)
                result = so_result.fetch()

                results.append(result)

            so_joined.assign(True)
            [proc.join() for proc in procs[5*i:5*(i+1)]]
        _logger.info(f"test: Took {perf_counter() - start_time:.3f} seconds")

        print(results, flush=True)
        assert not np.any(np.array(results) % data.size), results

        so_data.unlink()
        so_result.unlink()
        so_joined.unlink()


if __name__ == '__main__':
    t = TestMultiProcess()
    t.test_2_proc_race_condition_with_extra_bool()
    t.test_2_proc_race_condition_with_modified()

    t.test_5_proc_race_condition_with_extra_bool()
    t.test_5_proc_race_condition_with_modified()
