import random
import uuid

from real_robot.utils.multiprocessing import SharedObject


class TestInt:
    """Test SharedObject with int"""

    def test_create(self):
        for _ in range(500):
            data = random.randint(-9223372036854775808, 9223372036854775807)
            so = SharedObject(uuid.uuid4().hex, data=data)
            assert so.fetch() == data
            assert not so.modified

    def test_fetch(self):
        data = random.randint(-999999, 999999)
        so = SharedObject(uuid.uuid4().hex, data=data)

        v = random.randint(-100, 100)
        assert so.fetch(lambda x: x + v) == data + v
        assert so.fetch(lambda x: x * v) == data * v
        v = random.uniform(-100, 100)
        assert so.fetch(lambda x: x + v) == data + v
        assert so.fetch(lambda x: x * v) == data * v
        assert so.fetch() == data

    def test_assign(self):
        data = random.randint(-999999, 999999)
        so = SharedObject(uuid.uuid4().hex, data=data)
        assert so.fetch() == data
        assert not so.modified

        for _ in range(500):
            data = random.randint(-9223372036854775808, 9223372036854775807)
            so.assign(data)
            assert not so.modified
            assert so.fetch() == data
