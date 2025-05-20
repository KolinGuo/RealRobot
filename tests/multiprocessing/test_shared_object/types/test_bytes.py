import random
import uuid

from real_robot.utils.multiprocessing import SharedObject


class TestBytes:
    """Test SharedObject with bytes"""

    def test_create(self):
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

    def test_fetch(self):
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

        bytes_len = random.randrange(1, 100)
        data = random.randbytes(bytes_len)
        so = SharedObject(uuid.uuid4().hex, data=data)
        assert so.fetch(lambda x: len(x)) == len(data)
        assert so.fetch(lambda x: x[0]) == data[0]
        assert so.fetch(lambda x: x[:20]) == data[:20]
        assert so.fetch() == data

    def test_assign(self):
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

    def test_assign_overflow(self):
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
            raise AssertionError(
                "Should raise BufferError when assigning causes overflow"
            )

        for _ in range(10):
            bytes_len = random.randrange(51, 100)
            data = random.randbytes(bytes_len)
            try:
                so.assign(data)
            except BufferError as e:
                print(e)
            else:
                raise AssertionError(
                    "Should raise BufferError when assigning causes overflow"
                )
