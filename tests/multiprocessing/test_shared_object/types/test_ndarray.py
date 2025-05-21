import random
import uuid

import numpy as np

from real_robot.utils.multiprocessing import SharedObject
from real_robot.utils.multiprocessing.shared_object_metas import NP_DTYPES
from test_shared_object.utils import (
    NDARRAY_NBYTES_LIMIT,
    create_random_ndarray,
    create_random_object,
)


class TestNDArray:
    """Test SharedObject with np.ndarray"""

    def test_create(self):
        data = np.ones(1)
        so = SharedObject(uuid.uuid4().hex, data=data)
        data_fetched = so.fetch()
        np.testing.assert_equal(data_fetched, data)
        assert data_fetched.flags.owndata
        assert data_fetched.flags.writeable
        assert not so.modified

        for _ in range(500):
            data = create_random_object(SharedObject._object_types.index(np.ndarray))
            so = SharedObject(uuid.uuid4().hex, data=data)
            data_fetched = so.fetch()
            np.testing.assert_equal(data_fetched, data)
            assert data_fetched.flags.owndata
            assert data_fetched.flags.writeable
            assert not so.modified

    def test_fetch_fn_None(self):
        rng = np.random.default_rng()
        data = rng.random((480, 848, 3))
        so = SharedObject(uuid.uuid4().hex, data=data)
        data_fetched = so.fetch()
        data_fetched.fill(123)
        np.testing.assert_equal(so.fetch(), data)

    def test_fetch_fn_easy_op(self):
        rng = np.random.default_rng()
        data = rng.random((480, 848, 3))
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
        data_fetched = so.fetch(lambda x: x**v)
        np.testing.assert_equal(data_fetched, data**v)
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

    def test_fetch_fn_slice(self):
        rng = np.random.default_rng()
        data = rng.random((480, 848, 3))
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
        mask = rng.integers(2, size=data.shape, dtype=bool)
        data_fetched = so.fetch(lambda x: x[mask])
        np.testing.assert_equal(data_fetched, data[mask])
        assert data_fetched.flags.owndata
        assert data_fetched.flags.writeable
        data_fetched.fill(123)
        np.testing.assert_equal(so.fetch(), data)

    def test_fetch_fn_slice_and_op(self):
        rng = np.random.default_rng()
        data = rng.random((480, 848, 3))
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
        mask = rng.integers(2, size=data.shape, dtype=bool)
        data_fetched = so.fetch(lambda x: x[mask] + v)
        np.testing.assert_equal(data_fetched, data[mask] + v)
        assert data_fetched.flags.owndata
        assert data_fetched.flags.writeable
        data_fetched.fill(123)
        np.testing.assert_equal(so.fetch(), data)

    def test_fetch_fn_modify_inplace(self):
        rng = np.random.default_rng()
        data = rng.random((480, 848, 3))
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
            raise AssertionError(
                "Should raise ValueError when attempting to modify in fetch"
            )
        np.testing.assert_equal(so.fetch(), data)

        # inplace sort
        try:
            _ = so.fetch(lambda x: x.sort())
        except ValueError as e:
            print(e)
        else:
            raise AssertionError(
                "Should raise ValueError when attempting to modify in fetch"
            )
        np.testing.assert_equal(so.fetch(), data)

    def test_assign(self):
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
                dtype = random.choice(NP_DTYPES)
                size = dtype().itemsize * np.prod(shape, dtype=np.uint64)

            for i in range(10):
                data = create_random_ndarray(dtype, shape)  # type: ignore

                if i == 0:
                    so = SharedObject(uuid.uuid4().hex, data=data)
                else:
                    so.assign(data)
                assert not so.modified
                data_fetched = so.fetch()
                np.testing.assert_equal(data_fetched, data)
                assert data_fetched.flags.owndata
                assert data_fetched.flags.writeable

    def test_assign_changed_meta(self):
        rng = np.random.default_rng()
        data = rng.integers(256, size=(480, 848, 3), dtype=np.uint8)
        so = SharedObject(uuid.uuid4().hex, data=data)
        data_fetched = so.fetch()
        np.testing.assert_equal(data_fetched, data)
        data = np.ones((480, 848, 3))
        try:
            so.assign(data)
        except BufferError as e:
            print(e)
        else:
            raise AssertionError("Should raise BufferError when dtype is changed")

        for _ in range(10):
            size = NDARRAY_NBYTES_LIMIT + 1
            while size > NDARRAY_NBYTES_LIMIT:
                ndim = random.randint(1, 5)
                shape = tuple(random.randint(1, 1000) for _ in range(ndim))
                dtype = random.choice(NP_DTYPES)
                size = dtype().itemsize * np.prod(shape, dtype=np.uint64)

            data = create_random_ndarray(dtype, shape)  # type: ignore

            so = SharedObject(uuid.uuid4().hex, data=data)

            # Changed np dtype
            for new_dtype in NP_DTYPES:
                if new_dtype != dtype:  # type: ignore
                    data = create_random_ndarray(new_dtype, shape)  # type: ignore
                    try:
                        so.assign(data)
                    except BufferError as e:
                        print(e)
                    else:
                        raise AssertionError(
                            "Should raise BufferError when dtype is changed"
                        )

            # Changed ndim
            for i in range(5):
                new_shape = shape + (1,) * (i + 1)  # type: ignore
                data = create_random_ndarray(dtype, new_shape)  # type: ignore
                try:
                    so.assign(data)
                except BufferError as e:
                    print(e)
                else:
                    raise AssertionError(
                        "Should raise BufferError when ndim is changed"
                    )

            # Changed shape
            for i in range(5):
                new_shape = shape[:-1] + (shape[-1] + i + 1,)  # type: ignore
                data = create_random_ndarray(dtype, new_shape)  # type: ignore
                try:
                    so.assign(data)
                except BufferError as e:
                    print(e)
                else:
                    raise AssertionError(
                        "Should raise BufferError when shape is changed"
                    )
