import uuid

from real_robot.utils.multiprocessing import SharedObject


class TestNone:
    """Test SharedObject with None"""

    def test_create(self):
        so = SharedObject(uuid.uuid4().hex, data=None)
        assert so.fetch() is None
        assert not so.modified

    def test_fetch(self):
        so = SharedObject(uuid.uuid4().hex, data=None)
        assert so.fetch(lambda x: type(x)) is None.__class__
        assert so.fetch(lambda x: 10) == 10
        assert so.fetch() is None

    def test_assign(self):
        so = SharedObject(uuid.uuid4().hex, data=None)
        assert so.fetch() is None
        assert not so.modified

        so.assign(None)
        assert not so.modified
        assert so.fetch() is None
