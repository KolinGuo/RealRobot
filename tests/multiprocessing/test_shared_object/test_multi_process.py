"""Unittests for real_robot.utils.multiprocessing.shared_object"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from real_robot import LOGGER
from real_robot.utils.multiprocessing import SharedObject, ctx


class TestMultiProcess:
    """Test multiple processes"""

    @staticmethod
    def child_test_race_condition_with_extra_bool():
        # NOTE:
        # For processes that are always waiting for a massive SharedObject
        #   (e.g., np.ndarray),
        # it's better to add tiny delay to avoid starving processes that
        #   are assigning to it.
        # Even better, use a bool to indicate whether the data is updated yet and then
        #   only fetching the update flag inside the fetching processes to avoid this.
        so_data = SharedObject("data")
        so_data_updated = SharedObject("data_updated")
        so_result = SharedObject("result")
        so_joined = SharedObject("joined")

        while True:
            if so_joined.fetch():
                break

            if so_data_updated.fetch():
                res = float(so_data.fetch(lambda x: x.sum()))
                # res = float(so_data.np_ndarray.sum())  # Not protected by lock
                # print(f"[Child] {res} {perf_counter()}", flush=True)
                so_result.assign(res)
                so_data_updated.assign(False)

    def test_2_proc_race_condition_with_extra_bool(self):
        data = np.ones((10000, 10000))
        so_data = SharedObject("data", data=data)
        so_data_updated = SharedObject("data_updated", data=False)
        so_result = SharedObject("result", data=0.0)
        so_joined = SharedObject("joined", data=False)

        results = []
        n_iters = 10
        procs = [
            ctx.Process(target=self.child_test_race_condition_with_extra_bool, args=())  # type: ignore
            for _ in range(n_iters)
        ]
        start_time = perf_counter()
        for i in range(n_iters):
            data = np.ones((10000, 10000))
            so_joined.assign(False)
            so_data_updated.assign(False)

            procs[i].start()
            for _ in range(5):
                data += 1
                so_data.assign(data)
                # so_data.np_ndarray[:] = data  # Not protected by lock
                # print("[Main]", data[0], flush=True)
                so_data_updated.assign(True)
                result = so_result.fetch()

                results.append(result)

            so_joined.assign(True)
            procs[i].join()
        LOGGER.info("test: Took {:.3f} seconds", perf_counter() - start_time)

        print(results, flush=True)
        assert not np.any(np.array(results) % data.size), results

        so_data.unlink()
        so_data_updated.unlink()
        so_result.unlink()
        so_joined.unlink()

    def test_5_proc_race_condition_with_extra_bool(self):
        data = np.ones((10000, 10000))
        so_data = SharedObject("data", data=data)
        so_data_updated = SharedObject("data_updated", data=False)
        so_result = SharedObject("result", data=0.0)
        so_joined = SharedObject("joined", data=False)

        results = []
        n_iters = 10
        procs = [
            ctx.Process(target=self.child_test_race_condition_with_extra_bool, args=())  # type: ignore
            for _ in range(n_iters * 5)
        ]
        start_time = perf_counter()
        for i in range(n_iters):
            data = np.ones((10000, 10000))
            so_joined.assign(False)
            so_data_updated.assign(False)

            [proc.start() for proc in procs[5 * i : 5 * (i + 1)]]
            for _ in range(5):
                data += 1
                so_data.assign(data)
                # so_data.np_ndarray[:] = data  # Not protected by lock
                # print("[Main]", data[0], flush=True)
                so_data_updated.assign(True)
                result = so_result.fetch()

                results.append(result)

            so_joined.assign(True)
            [proc.join() for proc in procs[5 * i : 5 * (i + 1)]]
        LOGGER.info("test: Took {:.3f} seconds", perf_counter() - start_time)

        print(results, flush=True)
        assert not np.any(np.array(results) % data.size), results

        so_data.unlink()
        so_data_updated.unlink()
        so_result.unlink()
        so_joined.unlink()

    @staticmethod
    def child_test_race_condition_with_modified():
        # NOTE:
        # For processes that are always waiting for a massive SharedObject
        #   (e.g., np.ndarray),
        # it's best to use so.modified to check whether the data is updated yet.
        so_data = SharedObject("data")
        so_result = SharedObject("result")
        so_joined = SharedObject("joined")

        while True:
            if so_joined.fetch():
                break

            if so_data.modified:
                res = float(so_data.fetch(lambda x: x.sum()))
                # res = float(so_data.np_ndarray.sum())  # Not protected by lock
                # print(f"[Child] {res} {perf_counter()}", flush=True)
                so_result.assign(res)

    def test_2_proc_race_condition_with_modified(self):
        data = np.ones((10000, 10000))
        so_data = SharedObject("data", data=data)
        so_result = SharedObject("result", data=0.0)
        so_joined = SharedObject("joined", data=False)

        results = []
        n_iters = 10
        procs = [
            ctx.Process(target=self.child_test_race_condition_with_modified, args=())  # type: ignore
            for _ in range(n_iters)
        ]
        start_time = perf_counter()
        for i in range(n_iters):
            data = np.ones((10000, 10000))
            so_joined.assign(False)

            procs[i].start()
            for _ in range(5):
                data += 1
                so_data.assign(data)
                # so_data.np_ndarray[:] = data  # Not protected by lock
                # print("[Main]", data[0], flush=True)
                result = so_result.fetch()

                results.append(result)

            so_joined.assign(True)
            procs[i].join()
        LOGGER.info("test: Took {:.3f} seconds", perf_counter() - start_time)

        print(results, flush=True)
        assert not np.any(np.array(results) % data.size), results

        so_data.unlink()
        so_result.unlink()
        so_joined.unlink()

    def test_5_proc_race_condition_with_modified(self):
        data = np.ones((10000, 10000))
        so_data = SharedObject("data", data=data)
        so_result = SharedObject("result", data=0.0)
        so_joined = SharedObject("joined", data=False)

        results = []
        n_iters = 10
        procs = [
            ctx.Process(target=self.child_test_race_condition_with_modified, args=())  # type: ignore
            for _ in range(n_iters * 5)
        ]
        start_time = perf_counter()
        for i in range(n_iters):
            data = np.ones((10000, 10000))
            so_joined.assign(False)

            [proc.start() for proc in procs[5 * i : 5 * (i + 1)]]
            for _ in range(5):
                data += 1
                so_data.assign(data)
                # so_data.np_ndarray[:] = data  # Not protected by lock
                # print("[Main]", data[0], flush=True)
                result = so_result.fetch()

                results.append(result)

            so_joined.assign(True)
            [proc.join() for proc in procs[5 * i : 5 * (i + 1)]]
        LOGGER.info("test: Took {:.3f} seconds", perf_counter() - start_time)

        print(results, flush=True)
        assert not np.any(np.array(results) % data.size), results

        so_data.unlink()
        so_result.unlink()
        so_joined.unlink()


if __name__ == "__main__":
    t = TestMultiProcess()
    t.test_2_proc_race_condition_with_extra_bool()
    t.test_2_proc_race_condition_with_modified()

    t.test_5_proc_race_condition_with_extra_bool()
    t.test_5_proc_race_condition_with_modified()
