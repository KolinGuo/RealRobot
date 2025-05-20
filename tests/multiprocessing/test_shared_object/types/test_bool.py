import random
import uuid

from real_robot.utils.multiprocessing import SharedObject


class TestBool:
    """Test SharedObject with bool"""

    def test_create(self):
        so = SharedObject(uuid.uuid4().hex, data=False)
        assert so.fetch() is False
        assert not so.modified
        so = SharedObject(uuid.uuid4().hex, data=True)
        assert so.fetch() is True
        assert not so.modified

    def test_fetch(self):
        so = SharedObject(uuid.uuid4().hex, data=False)
        v = random.uniform(-100, 100)
        assert so.fetch(lambda x: x + v) == v
        assert so.fetch(lambda x: x * v) == 0.0
        assert so.fetch(lambda x: not x) is True
        assert so.fetch() is False

        so = SharedObject(uuid.uuid4().hex, data=True)
        v = random.uniform(-100, 100)
        assert so.fetch(lambda x: x + v) == v + 1
        assert so.fetch(lambda x: x * v) == v
        assert so.fetch(lambda x: not x) is False
        assert so.fetch() is True

    def test_assign(self):
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
