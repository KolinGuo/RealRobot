import random
import string
import uuid

from real_robot.utils.multiprocessing import SharedObject


class TestStr:
    """Test SharedObject with str"""

    def test_create(self):
        data = ""
        so = SharedObject(uuid.uuid4().hex, data=data)
        assert so.fetch() == data
        assert not so.modified

        for _ in range(500):
            str_len = random.randrange(100)
            data = "".join(random.choices(string.printable, k=str_len))
            so = SharedObject(uuid.uuid4().hex, data=data)
            assert so.fetch() == data
            assert not so.modified

    def test_fetch(self):
        data = ""
        so = SharedObject(uuid.uuid4().hex, data=data)
        assert so.fetch(lambda x: len(x)) == len(data)
        assert so.fetch() == data

        str_len = random.randrange(1, 100)
        data = "".join(random.choices(string.printable, k=str_len))
        so = SharedObject(uuid.uuid4().hex, data=data)

        assert so.fetch(lambda x: len(x)) == len(data)
        assert so.fetch(lambda x: x[0]) == data[0]
        assert so.fetch(lambda x: x[:20]) == data[:20]
        assert so.fetch() == data

    def test_assign(self):
        data = ""
        so = SharedObject(uuid.uuid4().hex, data=data, init_size=200)
        assert so.fetch() == data
        assert not so.modified

        for _ in range(500):
            str_len = random.randrange(100)
            data = "".join(random.choices(string.printable, k=str_len))
            so.assign(data)
            assert not so.modified
            assert so.fetch() == data

    def test_assign_overflow(self):
        data = ""
        so = SharedObject(uuid.uuid4().hex, data=data)
        assert so.fetch() == data
        assert not so.modified

        str_len = 50
        data = "".join(random.choices(string.printable, k=str_len))
        so.assign(data)
        assert so.fetch() == data
        assert not so.modified

        str_len = 51
        data = "".join(random.choices(string.printable, k=str_len))
        try:
            so.assign(data)
        except BufferError as e:
            print(e)
        else:
            raise AssertionError(
                "Should raise BufferError when assigning causes overflow"
            )

        for _ in range(10):
            str_len = random.randrange(51, 100)
            data = "".join(random.choices(string.printable, k=str_len))
            try:
                so.assign(data)
            except BufferError as e:
                print(e)
            else:
                raise AssertionError(
                    "Should raise BufferError when assigning causes overflow"
                )
