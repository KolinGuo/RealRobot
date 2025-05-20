import uuid

import numpy as np

from real_robot.utils.multiprocessing import SharedObject
from test_shared_object.utils import (
    check_object_equal,
    create_random_ndarray,
    create_random_object,
)


class TestTrigger:
    """Test so.trigger() / so.triggered"""

    def test_trigger(self):
        so = SharedObject(uuid.uuid4().hex)
        assert not so.triggered
        so.trigger()
        assert so.triggered
        assert not so.triggered
        assert not so.triggered
        so.trigger()
        so.trigger()
        assert so.triggered
        assert not so.triggered
        assert not so.triggered

    def test_two_instances(self):
        for object_type_idx in range(len(SharedObject._object_types)):
            data = create_random_object(object_type_idx)
            so = SharedObject(uuid.uuid4().hex, data=data)
            so2 = SharedObject(so.name)
            assert not so.triggered
            assert not so2.triggered
            check_object_equal(so, so2, data)

            if object_type_idx == SharedObject._object_types.index(np.ndarray):
                new_data = create_random_ndarray(data.dtype, data.shape)
            else:
                new_data = create_random_object(object_type_idx)
            so2.trigger()
            assert so2.triggered
            assert not so2.triggered
            assert so.triggered
            assert not so.triggered
            so2.assign(new_data)
            assert so.triggered
            assert not so2.triggered
            assert not so.triggered
            assert not so2.triggered
            check_object_equal(so, so2, new_data)

            if object_type_idx == SharedObject._object_types.index(np.ndarray):
                new_data = create_random_ndarray(data.dtype, data.shape)
            else:
                new_data = create_random_object(object_type_idx)
            so.trigger()
            assert so.triggered
            assert not so.triggered
            assert so2.triggered
            assert not so2.triggered
            so.assign(new_data)
            assert not so.triggered
            assert so2.triggered
            assert not so.triggered
            assert not so2.triggered
            check_object_equal(so, so2, new_data)

    def test_five_instances(self):
        for object_type_idx in range(len(SharedObject._object_types)):
            data = create_random_object(object_type_idx)
            so = SharedObject(uuid.uuid4().hex, data=data)
            assert not so.triggered
            sos = [SharedObject(so.name) for _ in range(4)]
            for so2 in sos:
                assert not so2.triggered
                check_object_equal(so, so2, data)

            if object_type_idx == SharedObject._object_types.index(np.ndarray):
                new_data = create_random_ndarray(data.dtype, data.shape)
            else:
                new_data = create_random_object(object_type_idx)
            so.trigger()
            assert so.triggered
            assert not so.triggered
            for so2 in sos:
                assert so2.triggered
                assert not so2.triggered
            so.assign(new_data)
            assert not so.triggered
            assert not so.triggered
            for so2 in sos:
                assert so2.triggered
                assert not so2.triggered
                check_object_equal(so, so2, new_data)

            for so2 in sos:
                if object_type_idx == SharedObject._object_types.index(np.ndarray):
                    new_data = create_random_ndarray(data.dtype, data.shape)
                else:
                    new_data = create_random_object(object_type_idx)
                so2.trigger()
                assert so2.triggered
                assert not so2.triggered
                assert so.triggered
                assert not so.triggered
                so2.assign(new_data)
                assert so.triggered
                assert not so2.triggered
                assert not so.triggered
                assert not so2.triggered
                check_object_equal(so, so2, new_data)
