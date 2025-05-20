import random
import uuid

from real_robot.utils.multiprocessing import SharedObject
from test_shared_object.utils import create_random_object


class TestComplex:
    """Test SharedObject with complex"""

    def test_create(self):
        for _ in range(1000):
            data = create_random_object(
                object_type_idx=SharedObject._object_types.index(complex)
            )
            so = SharedObject(uuid.uuid4().hex, data=data)
            assert so.fetch() == data
            assert not so.modified

    def test_fetch(self):
        data = create_random_object(
            object_type_idx=SharedObject._object_types.index(complex)
        )
        so = SharedObject(uuid.uuid4().hex, data=data)

        v = random.randint(-100, 100)
        assert so.fetch(lambda x: x + v) == data + v
        assert so.fetch(lambda x: x * v) == data * v
        v = random.uniform(-100, 100)
        assert so.fetch(lambda x: x + v) == data + v
        assert so.fetch(lambda x: x * v) == data * v
        assert so.fetch() == data

    def test_assign(self):
        data = create_random_object(
            object_type_idx=SharedObject._object_types.index(complex)
        )
        so = SharedObject(uuid.uuid4().hex, data=data)
        assert so.fetch() == data
        assert not so.modified

        for _ in range(1000):
            data = create_random_object(
                object_type_idx=SharedObject._object_types.index(complex)
            )
            so.assign(data)
            assert not so.modified
            assert so.fetch() == data
