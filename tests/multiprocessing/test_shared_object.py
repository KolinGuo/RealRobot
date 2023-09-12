"""Unittests for real_robot.utils.multiprocessing.shared_object"""
import uuid
import random
import string

import numpy as np

from real_robot.utils.multiprocessing import SharedObject


class TestCreate:
    def test_None(self):
        so = SharedObject(uuid.uuid4().hex, data=None)
        assert so.fetch() is None

    def test_bool(self):
        so = SharedObject(uuid.uuid4().hex, data=False)
        assert so.fetch() is False
        so = SharedObject(uuid.uuid4().hex, data=True)
        assert so.fetch() is True

    def test_int(self):
        for _ in range(500):
            data = random.randint(-9223372036854775808, 9223372036854775807)
            so = SharedObject(uuid.uuid4().hex, data=data)
            assert so.fetch() == data

    def test_float(self):
        for _ in range(500):
            data = random.uniform(-100, 100)
            so = SharedObject(uuid.uuid4().hex, data=data)
            assert so.fetch() == data

        for _ in range(500):
            data = random.uniform(-1e307, 1e308)
            so = SharedObject(uuid.uuid4().hex, data=data)
            assert so.fetch() == data

    def test_str(self):
        data = ""
        so = SharedObject(uuid.uuid4().hex, data=data)
        assert so.fetch() == data

        for _ in range(500):
            str_len = random.randrange(100)
            data = ''.join(random.choices(string.printable, k=str_len))
            so = SharedObject(uuid.uuid4().hex, data=data)
            assert so.fetch() == data

    def test_bytes(self):
        data = b""
        so = SharedObject(uuid.uuid4().hex, data=data)
        assert so.fetch() == data

        data = b"\x00"
        so = SharedObject(uuid.uuid4().hex, data=data)
        assert so.fetch() == data

        data = b"asdlkj123\x01asd\x00\x00"
        so = SharedObject(uuid.uuid4().hex, data=data)
        assert so.fetch() == data

        for _ in range(500):
            bytes_len = random.randrange(100)
            data = random.randbytes(bytes_len)
            so = SharedObject(uuid.uuid4().hex, data=data)
            assert so.fetch() == data

    def test_ndarray(self):
        data = np.ones(1)
        so = SharedObject(uuid.uuid4().hex, data=data)
        assert so.fetch() == data

        # TODO: add more tests for all dtypes
