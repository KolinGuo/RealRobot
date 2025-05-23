import time
import uuid

import numpy as np

from real_robot.utils.multiprocessing import SharedObject
from test_shared_object.utils import check_object_equal


class TestDict:
    """Test SharedObject with dict"""

    def test_create(self):
        rng = np.random.default_rng()
        data = {
            "timestamp": time.time_ns(),
            "image": rng.integers(0, 256, size=(848, 480, 3), dtype=np.uint8),
        }
        so = SharedObject(uuid.uuid4().hex, data=data)
        check_object_equal(so, data)
        assert not so.modified

    def test_fetch(self):
        rng = np.random.default_rng()
        data = {
            "timestamp": time.time_ns(),
            "image": rng.integers(0, 256, size=(848, 480, 3), dtype=np.uint8),
        }
        so_name = uuid.uuid4().hex
        so = SharedObject(so_name, data=data)
        check_object_equal(so, data)

        so2 = SharedObject(so_name)
        check_object_equal(so, so2, data)

    def test_assign(self):
        rng = np.random.default_rng()
        data = {
            "timestamp": time.time_ns(),
            "image": rng.integers(0, 256, size=(848, 480, 3), dtype=np.uint8),
        }
        so = SharedObject(uuid.uuid4().hex, data=data)
        check_object_equal(so, data)

        data = {
            "timestamp": time.time_ns(),
            "rgb": rng.integers(0, 256, size=(848, 480, 3), dtype=np.uint8),
        }
        so.assign(data)
        assert not so.modified
        check_object_equal(so, data)
