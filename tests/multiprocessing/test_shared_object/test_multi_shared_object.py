import uuid

from real_robot.utils.multiprocessing import SharedObject
from test_shared_object.utils import check_object_equal, create_random_object


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

            new_data = create_random_object(object_type_idx, data_sample=data)
            so2.assign(new_data)
            assert so.modified
            assert not so2.modified
            check_object_equal(so, so2, new_data)

            new_data = create_random_object(object_type_idx, data_sample=data)
            so.assign(new_data)
            assert not so.modified
            assert so2.modified
            check_object_equal(so, so2, new_data)

            so.close()
            so = SharedObject(so.name)

            new_data = create_random_object(object_type_idx, data_sample=data)
            so2.assign(new_data)
            assert so.modified
            assert not so2.modified
            check_object_equal(so, so2, new_data)

            new_data = create_random_object(object_type_idx, data_sample=data)
            so.assign(new_data)
            assert not so.modified
            assert so2.modified
            check_object_equal(so, so2, new_data)

            del so2
            so2 = SharedObject(so.name)

            new_data = create_random_object(object_type_idx, data_sample=data)
            so2.assign(new_data)
            assert so.modified
            assert not so2.modified
            check_object_equal(so, so2, new_data)

            new_data = create_random_object(object_type_idx, data_sample=data)
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

            new_data = create_random_object(object_type_idx, data_sample=data)
            so.assign(new_data)
            assert not so.modified
            for so2 in sos:
                assert so2.modified
                check_object_equal(so, so2, new_data)

            for so2 in sos:
                new_data = create_random_object(object_type_idx, data_sample=data)
                so2.assign(new_data)
                assert so.modified
                assert not so2.modified
                check_object_equal(so, so2, new_data)
