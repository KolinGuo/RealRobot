import uuid

from real_robot.utils.multiprocessing import SharedObject
from test_shared_object.utils import create_random_object


class TestAssign:
    """Test assigning SharedObject"""

    def test_changed_object_type(self):
        for object_type_idx in range(len(SharedObject._object_types)):
            data = create_random_object(object_type_idx)
            so = SharedObject(uuid.uuid4().hex, data=data)

            for new_object_type_idx in range(len(SharedObject._object_types)):
                if object_type_idx != new_object_type_idx:
                    new_data = create_random_object(new_object_type_idx)
                    try:
                        so.assign(new_data)
                    except BufferError as e:
                        print(e)
                    else:
                        raise AssertionError(
                            "Should raise BufferError on changed object type"
                        )
