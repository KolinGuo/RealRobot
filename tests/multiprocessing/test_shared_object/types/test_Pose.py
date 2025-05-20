import uuid

import numpy as np
import pytest

pytest.importorskip("sapien")
from sapien import Pose

from real_robot.utils.multiprocessing import SharedObject
from test_shared_object.utils import create_random_object


class TestPose:
    """Test SharedObject with Pose"""

    def test_create(self):
        for _ in range(500):
            pose = create_random_object(SharedObject._object_types.index(Pose))
            so = SharedObject(uuid.uuid4().hex, data=pose)
            np.testing.assert_equal(so.fetch().__getstate__(), pose.__getstate__())
            assert not so.modified

    def test_fetch(self):
        pose = create_random_object(SharedObject._object_types.index(Pose))
        so = SharedObject(uuid.uuid4().hex, data=pose)

        for _ in range(500):
            pose2 = create_random_object(SharedObject._object_types.index(Pose))
            np.testing.assert_equal(
                so.fetch(lambda x, pose=pose2: x * pose).__getstate__(),
                (pose * pose2).__getstate__(),
            )

        for _ in range(500):
            pose2 = create_random_object(SharedObject._object_types.index(Pose))
            np.testing.assert_equal(
                so.fetch(lambda x, pose=pose2: pose * x).__getstate__(),
                (pose2 * pose).__getstate__(),
            )

        np.testing.assert_equal(
            so.fetch(lambda x: x.inv()).__getstate__(), pose.inv().__getstate__()
        )

        np.testing.assert_equal(
            so.fetch(lambda x: x.to_transformation_matrix()),
            pose.to_transformation_matrix(),
        )

    def test_fetch_fn_modify_inplace(self):
        pose = create_random_object(SharedObject._object_types.index(Pose))
        so = SharedObject(uuid.uuid4().hex, data=pose)

        # modify
        def inplace_modify(x):
            x.set_p([1, 2, 3])
            return x

        so.fetch(inplace_modify)  # no change to buffer
        np.testing.assert_equal(
            so.fetch(inplace_modify).__getstate__(),
            Pose(p=[1, 2, 3], q=pose.q).__getstate__(),
        )
        np.testing.assert_equal(so.fetch().__getstate__(), pose.__getstate__())

    def test_assign(self):
        pose = create_random_object(SharedObject._object_types.index(Pose))
        so = SharedObject(uuid.uuid4().hex, data=pose)
        np.testing.assert_equal(so.fetch().__getstate__(), pose.__getstate__())
        assert not so.modified

        for _ in range(500):
            pose = create_random_object(SharedObject._object_types.index(Pose))
            so.assign(pose)
            assert not so.modified
            np.testing.assert_equal(so.fetch().__getstate__(), pose.__getstate__())
