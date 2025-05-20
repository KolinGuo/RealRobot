import random
import uuid

from real_robot.utils.multiprocessing import SharedObject


class TestFloat:
    """Test SharedObject with float"""

    def test_create(self):
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

    def test_fetch(self):
        data = random.uniform(-100000, 100000)
        so = SharedObject(uuid.uuid4().hex, data=data)

        v = random.randint(-100, 100)
        assert so.fetch(lambda x: x + v) == data + v
        assert so.fetch(lambda x: x * v) == data * v
        v = random.uniform(-100, 100)
        assert so.fetch(lambda x: x + v) == data + v
        assert so.fetch(lambda x: x * v) == data * v
        assert so.fetch() == data

    def test_assign(self):
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
