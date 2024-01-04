"""
Shared object implemented with SharedMemory and synchronization
Careful optimization is done to make it run as fast as possible
version 0.0.6

Written by Kolin Guo

Implementation Notes:
  * Methods `fetch()` and `assign()` are chosen to be distinctive from common python
    class methods (e.g., get(), set(), update(), read(), write(), fill(), put(), etc.)
  * Readers-Writer synchronization is achieved by `fcntl.flock()` (filesystem advisory
    lock). It's chosen over `multiprocessing.Lock` / `multiprocessing.Condition` so that
    no lock needs to be explicitly passed to child processes.
    However, note that acquiring `flock` locks are not guaranteed to be in order.
  * An object modified timestamp is maintained using `time.time_ns()` which is
    system-wide and has highest resolution. (https://peps.python.org/pep-0564/#linux)

Usage Notes:
  * For processes that are always waiting for a massive SharedObject (e.g., np.ndarray),
    it's best to use so.modified to check whether the data has been updated yet to avoid
    starving processes that are assigning to it (only fetch when so.modified is True).
    Alternatively, a separate boolean update flag can be used to achieve the same.
  * Examples:
    # Creates SharedObject with data
    >>> so = SharedObject("test", data=np.ones((480, 848, 3)))
    # Mounts existing SharedObject
    >>> so = SharedObject("test")
    # Creates a trigger SharedObject (data is None, can be used for joining processes)
    >>> so = SharedObject("trigger")
    >>> so.trigger()  # trigger the SharedObject
    >>> so.triggered  # check if triggered
  * Best practices when fetching np.ndarray (see `SharedObject._fetch_ndarray()`):
    >>> so.fetch(lambda x: x.sum())  # Apply operation only
    >>> so.fetch(lambda x: x + 1)    # Apply operation only
    >>> so.fetch()  # If different operations need to be applied on the same data
    >>> so.fetch()[..., 0]           # Slice only
    >>> so.fetch(lambda x: x[..., 0].copy()) + 1  # Slice and apply operation
"""

from __future__ import annotations

import struct
import time
from collections.abc import Callable
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Union

import numpy as np
from sapien import Pose

from ..logger import get_logger

try:
    import fcntl
except ModuleNotFoundError as e:
    print("Not supported on Windows")
    raise e


_encoding = "utf8"


class ReadersLock:
    """lock.acquire() / .release() is slightly faster than using as a contextmanager"""

    def __init__(self, fd):
        self.fd = fd

    def acquire(self):
        fcntl.flock(self.fd, fcntl.LOCK_SH)

    def release(self):
        fcntl.flock(self.fd, fcntl.LOCK_UN)

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()


class WriterLock:
    """lock.acquire() / .release() is slightly faster than using as a contextmanager"""

    def __init__(self, fd):
        self.fd = fd

    def acquire(self):
        fcntl.flock(self.fd, fcntl.LOCK_EX)

    def release(self):
        fcntl.flock(self.fd, fcntl.LOCK_UN)

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()


class SharedObject:
    """Shared object implemented with SharedMemory and synchronization
    SharedMemory reallocation, casting object_type, changing numpy metas are not allowed
    Use SharedDynamicObject instead

    The shared memory buffer is organized as follows:
    - 8 bytes: object modified timestamp in ns (since the epoch), stored as 'Q'
    - 1 byte: object data type index, stored as 'B'
    - X bytes: data area
      For NoneType, data area is ignored
      For bool, 1 byte data
      For int / float, 8 bytes data
      For sapien.Pose, 7*4 = 28 bytes data ([xyz, wxyz], float32)
      For str / bytes, (N + 1) bytes data, N is str / bytes length, 1 is for termination
      For np.ndarray,
      - 1 byte: array dtype index, stored as 'B'
      - 8 bytes: array ndim, stored as 'Q'
      - (K * 8) bytes: array shape for each dimension, stored as 'Q'
      - D bytes: array data buffer
    """

    # object_type_idx:     0          1     2     3     4     5     6        7
    _object_types = (None.__class__, bool, int, float, Pose, str, bytes, np.ndarray)

    @staticmethod
    def _get_bytes_size(enc_str: bytes, init_size: int) -> int:
        if (sz := len(enc_str) << 1) >= init_size:
            return sz + 10
        else:
            return init_size + 10

    _object_sizes = (
        9,  # NoneType
        10,  # bool
        17,  # int
        17,  # float
        37,  # sapien.Pose
        _get_bytes_size.__func__,  # str
        _get_bytes_size.__func__,  # bytes
        lambda array, ndim: array.nbytes + ndim * 8 + 18,  # ndarray
    )

    @staticmethod
    def _fetch_metas(shm: SharedMemory) -> tuple:
        nbytes = shm._size
        mtime, object_type_idx = struct.unpack_from("QB", shm.buf, offset=0)
        np_metas = ()
        if object_type_idx == 7:  # np.ndarray
            np_metas = SharedObject._fetch_np_metas(shm.buf)
        return nbytes, mtime, object_type_idx, np_metas

    @staticmethod
    def _fetch_np_metas(buf) -> tuple:
        np_dtype_idx, data_ndim = struct.unpack_from("=BQ", buf, offset=9)
        data_shape = struct.unpack_from("Q" * data_ndim, buf, offset=18)
        return np_dtype_idx, data_ndim, data_shape

    _fetch_fn_type = Callable[[Union[_object_types]], Any] | None

    @staticmethod
    def _fetch_None(buf, fn: Callable[[None.__class__], Any] | None, *args) -> Any:
        return None if fn is None else fn(None)

    @staticmethod
    def _fetch_bool(buf, fn: Callable[[bool], Any] | None, *args) -> Any:
        return bool(buf[9]) if fn is None else fn(bool(buf[9]))

    @staticmethod
    def _fetch_int(buf, fn: Callable[[int], Any] | None, *args) -> Any:
        v = struct.unpack_from("q", buf, offset=9)[0]
        return v if fn is None else fn(v)

    @staticmethod
    def _fetch_float(buf, fn: Callable[[float], Any] | None, *args) -> Any:
        v = struct.unpack_from("d", buf, offset=9)[0]
        return v if fn is None else fn(v)

    @staticmethod
    def _fetch_pose(buf, fn: Callable[[Pose], Any] | None, *args) -> Any:
        """Fetch and construct a sapien.Pose (using __setstate__)"""
        pose = Pose.__new__(Pose)
        pose.__setstate__(struct.unpack_from("7f", buf, offset=9))
        return pose if fn is None else fn(pose)

    @staticmethod
    def _fetch_str(buf, fn: Callable[[str], Any] | None, *args) -> Any:
        v = buf[9:].tobytes().rstrip(b"\x00")[:-1].decode(_encoding)
        return v if fn is None else fn(v)

    @staticmethod
    def _fetch_bytes(buf, fn: Callable[[bytes], Any] | None, *args) -> Any:
        v = buf[9:].tobytes().rstrip(b"\x00")[:-1]
        return v if fn is None else fn(v)

    @staticmethod
    def _fetch_ndarray(
        buf, fn: Callable[[np.ndarray], Any] | None, data_buf_ro: np.ndarray
    ) -> Any:
        """Always return a copy of the underlying buffer
        Examples (ordered from fastest to slowest, benchmarked with 480x848x3 np.uint8):

            # Apply operation only
            so.fetch(lambda x: x.sum())  # contiguous sum (triggers a copy)
            so.fetch().sum()  # contiguous copy => sum

            # Apply operation only
            so.fetch(lambda x: x + 1)  # contiguous add (triggers a copy)
            so.fetch() + 1  # contiguous copy => add

            # Slice only (results might vary depending on array size)
            so.fetch()[..., 0]  # contiguous copy => slice
            so.fetch(lambda x: x[..., 0].copy())  # non-contiguous copy

            # Slice and apply operation (results might vary depending on array size)
            so.fetch(lambda x: x[..., 0].copy()) + 1  # non-contiguous copy => add
            so.fetch(lambda x: x[..., 0] + 1)  # non-contiguous add (triggers a copy)
            so.fetch()[..., 0] + 1  # contiguous copy => non-contiguous add
        """
        if fn is not None:
            data = fn(data_buf_ro)
            if not isinstance(data, np.ndarray) or data.flags.owndata:
                return data
            else:
                get_logger("SharedObject").warning(
                    "Fetching ndarray with fn that does not trigger a copy "
                    "induces an extra copy. Consider changing to improve performance."
                )
                return data.copy()
        else:
            return data_buf_ro.copy()

    _fetch_objects = (
        _fetch_None.__func__,
        _fetch_bool.__func__,
        _fetch_int.__func__,
        _fetch_float.__func__,
        _fetch_pose.__func__,
        _fetch_str.__func__,
        _fetch_bytes.__func__,
        _fetch_ndarray.__func__,
    )

    @staticmethod
    def _assign_np_metas(buf, np_dtype_idx: int, data_ndim: int, data_shape: tuple):
        struct.pack_into(
            "=BQ" + "Q" * data_ndim, buf, 9, np_dtype_idx, data_ndim, *data_shape
        )

    @staticmethod
    def _assign_None(*args):
        pass

    @staticmethod
    def _assign_bool(buf, data: bool, *args):
        buf[9] = data

    @staticmethod
    def _assign_int(buf, data: int, *args):
        struct.pack_into("q", buf, 9, data)

    @staticmethod
    def _assign_float(buf, data: float, *args):
        struct.pack_into("d", buf, 9, data)

    @staticmethod
    def _assign_pose(buf, pose: Pose, *args):
        struct.pack_into("7f", buf, 9, *pose.__getstate__())

    @staticmethod
    def _assign_bytes(buf, enc_data: bytes, buf_nbytes: int, *args):
        struct.pack_into(f"{buf_nbytes-9}s", buf, 9, enc_data + b"\xff")

    @staticmethod
    def _assign_ndarray(buf, data: np.ndarray, buf_nbytes: int, data_buf: np.ndarray):
        data_buf[:] = data

    _assign_objects = (
        _assign_None.__func__,
        _assign_bool.__func__,
        _assign_int.__func__,
        _assign_float.__func__,
        _assign_pose.__func__,
        _assign_bytes.__func__,
        _assign_bytes.__func__,
        _assign_ndarray.__func__,
    )
    _np_dtypes = (
        np.bool_,
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
        np.float128,
        np.complex64,
        np.complex128,
        np.complex256,
    )

    def __init__(self, name: str, *, data: Union[_object_types] = None, init_size=100):
        """
        Examples:
            # Mounts SharedMemory "test" if exists,
            # Else creates SharedMemory "test" which holds None by default
            so = SharedObject("test")

            # Mounts SharedMemory "test" if exists and assign data (True) to it,
            # Else creates SharedMemory "test" and assigns data
            so = SharedObject("test", data=True)

            # Mounts SharedMemory "test" if exists and assign data (np.ones(10)) to it,
            # Else creates SharedMemory "test" and assigns data
            so = SharedObject("test", data=np.ones(10))

        :param init_size: only used for str and bytes,
                          initial buffer size to save frequent reallocation.
                          The buffer is expanded with exponential growth rate of 2
        """
        self.init_size = init_size
        data, object_type_idx, nbytes, np_metas = self._preprocess_data(data)

        try:
            self.shm = SharedMemory(name)
            created = False
        except FileNotFoundError:  # no SharedMemory with given name
            self.shm = SharedMemory(name, create=True, size=nbytes)
            created = True
        self.name = name
        self._readers_lock = ReadersLock(self.shm._fd)
        self._writer_lock = WriterLock(self.shm._fd)

        if created:
            self.nbytes = nbytes
            self.mtime = time.time_ns()
            self.object_type_idx = object_type_idx
            self.np_metas = np_metas
            # Assign object_type, np_metas to init object meta info
            self._writer_lock.acquire()
            self.shm.buf[8] = object_type_idx
            if object_type_idx == 7:  # np.ndarray
                self._assign_np_metas(self.shm.buf, *np_metas)
            self._writer_lock.release()
        else:
            self._readers_lock.acquire()
            self.nbytes, self.mtime, self.object_type_idx, self.np_metas = (
                self._fetch_metas(self.shm)
            )
            self._readers_lock.release()

        # Create np.ndarray here to save frequent np.ndarray construction
        self.np_ndarray, self.np_ndarray_ro = None, None
        if self.object_type_idx == 7:  # np.ndarray
            np_dtype_idx, data_ndim, data_shape = self.np_metas
            self.np_ndarray = np.ndarray(
                data_shape,
                dtype=self._np_dtypes[np_dtype_idx],
                buffer=self.shm.buf,
                offset=data_ndim * 8 + 18,
            )
            # Create a read-only view for fetch()
            self.np_ndarray_ro = self.np_ndarray.view()
            self.np_ndarray_ro.setflags(write=False)

        # fill data
        if data is not None:
            if not created:
                get_logger("SharedObject").warning(
                    f"Implicitly overwriting data of {self!r}"
                )
            self._assign(data, object_type_idx, nbytes, np_metas)

    def _preprocess_data(self, data: Union[_object_types]) -> tuple:
        """Preprocess object data and return useful informations

        :return data: processed data. Only changed for str (=> encoded bytes)
        :return object_type_idx: object type index
        :return nbytes: number of bytes needed for SharedMemory
        :return np_metas: numpy meta info, (np_dtype_idx, data_ndim, data_shape)
        """
        try:
            object_type_idx = self._object_types.index(type(data))
        except ValueError:
            raise TypeError(f"Not supported object_type: {type(data)}")

        # Get shared memory size in bytes
        np_metas = ()
        if object_type_idx <= 4:  # NoneType, bool, int, float, sapien.Pose
            nbytes = self._object_sizes[object_type_idx]
        elif object_type_idx == 5:  # str
            data = data.encode(_encoding)  # encode strings into bytes
            nbytes = self._object_sizes[object_type_idx](data, self.init_size)
        elif object_type_idx == 6:  # bytes
            nbytes = self._object_sizes[object_type_idx](data, self.init_size)
        elif object_type_idx == 7:  # np.ndarray
            try:
                np_dtype_idx = self._np_dtypes.index(data.dtype)
            except ValueError:
                raise TypeError(f"Not supported numpy dtype: {data.dtype}")
            data_ndim = data.ndim
            np_metas = (np_dtype_idx, data_ndim, data.shape)

            nbytes = self._object_sizes[object_type_idx](data, data_ndim)
        else:
            raise ValueError(f"Unknown {object_type_idx = }")

        return data, object_type_idx, nbytes, np_metas

    @property
    def modified(self) -> bool:
        """Returns whether the object's data has been modified by another process.
        Check by fetching object modified timestamp and comparing with self.mtime
        """
        self._readers_lock.acquire()
        mtime = struct.unpack_from("Q", self.shm.buf, offset=0)[0]
        self._readers_lock.release()
        return mtime > self.mtime

    @property
    def triggered(self) -> bool:
        """Returns whether the object is triggered (protected by readers lock)
        Check by fetching object modified timestamp, comparing with self.mtime
        and updating self.mtime
        """
        self._readers_lock.acquire()
        mtime = struct.unpack_from("Q", self.shm.buf, offset=0)[0]
        self._readers_lock.release()
        modified = mtime > self.mtime
        self.mtime = mtime
        return modified

    def fetch(self, fn: _fetch_fn_type = None) -> Any:
        """Fetch a copy of data from SharedMemory (protected by readers lock)
        See SharedObject._fetch_ndarray() for best practices of fn with np.ndarray

        :param fn: function to apply on data, e.g., lambda x: x + 1.
                   If fn is None or does not trigger a copy for ndarray
                     (e.g., slicing, masking), a manual copy is applied.
                   Thus, the best practices are ordered as:
                   fn (triggers a copy) > fn = None >> fn (does not trigger a copy)
                     because copying non-contiguous ndarray takes much longer time.
        :return data: a copy of data read from SharedMemory
        """
        self._readers_lock.acquire()
        # Update modified timestamp
        self.mtime = struct.unpack_from("Q", self.shm.buf, offset=0)[0]
        data = self._fetch_objects[self.object_type_idx](
            self.shm.buf, fn, self.np_ndarray_ro
        )
        self._readers_lock.release()
        return data

    def trigger(self) -> "SharedObject":
        """Trigger by modifying object mtime (protected by writer lock)"""
        self._writer_lock.acquire()
        # Update mtime
        struct.pack_into("Q", self.shm.buf, 0, time.time_ns())
        self._writer_lock.release()
        return self

    def assign(self, data: Union[_object_types]) -> "SharedObject":
        """Assign data to SharedMemory (protected by writer lock)"""
        return self._assign(*self._preprocess_data(data))

    def _assign(
        self, data, object_type_idx: int, nbytes: int, np_metas: tuple
    ) -> "SharedObject":
        """Inner function for assigning data (protected by writer lock)
        For SharedObject, object_type_idx, nbytes, and np_metas cannot be modified
        """
        if (
            object_type_idx != self.object_type_idx
            or nbytes > self.nbytes
            or np_metas != self.np_metas
        ):
            raise BufferError(
                f"Casting object type (new={self._object_types[object_type_idx]}, "
                f"old={self._object_types[self.object_type_idx]}) OR "
                f"Buffer overflow (new={nbytes} > {self.nbytes}=old) OR "
                f"Changed numpy meta (new={np_metas}, old={self.np_metas}) in {self!r}"
            )

        self._writer_lock.acquire()
        # Assign mtime
        self.mtime = time.time_ns()
        struct.pack_into("Q", self.shm.buf, 0, self.mtime)

        # Assign object data
        self._assign_objects[self.object_type_idx](
            self.shm.buf, data, self.nbytes, self.np_ndarray
        )
        self._writer_lock.release()
        return self

    def close(self):
        """Closes access to the shared memory from this instance but does
        not destroy the shared memory block."""
        self.shm.close()

    def unlink(self):
        """Requests that the underlying shared memory block be destroyed.

        In order to ensure proper cleanup of resources, unlink should be
        called once (and only once) across all processes which have access
        to the shared memory block."""
        self.shm.unlink()

    def __del__(self):
        self.close()

    def __reduce__(self):
        return self.__class__, (self.name,)

    def __repr__(self):
        return (
            f"<{self.__class__.__name__}: name={self.name}, "
            f"data_type={self._object_types[self.object_type_idx]}, "
            f"nbytes={self.nbytes}>"
        )


class SharedDynamicObject(SharedObject):
    """Shared object implemented with SharedMemory and synchronization
    Allow reallocating SharedMemory.
        Need more checks and thus is slower than SharedObject.
        In fact, this should never be implemented.
        For size-variable np.ndarray, just implement similar support as str/bytes
    """

    @staticmethod
    def _fetch_metas(shm: SharedMemory) -> tuple:
        nbytes = shm._mmap.size()  # _mmap size will be updated by os.ftruncate()
        mtime, object_type_idx = struct.unpack_from("QB", shm.buf, offset=0)
        np_metas = ()
        if object_type_idx == 7:  # np.ndarray
            np_metas = SharedObject._fetch_np_metas(shm.buf)
        return nbytes, mtime, object_type_idx, np_metas

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Implementation not complete")

    def fetch(self, fn: SharedObject._fetch_fn_type = None) -> Any:
        """Fetch a copy of data from SharedMemory (protected by readers lock)
        :param fn: function to apply on data, e.g., lambda x: x + 1.
        :return data: a copy of data read from SharedMemory
        """
        self._readers_lock.acquire()
        # Fetch shm info
        self.nbytes, self.mtime, self.object_type_idx, self.np_metas = (
            self._fetch_metas(self.shm)
        )

        data = self._fetch_objects[self.object_type_idx](
            self.shm.buf, fn, self.np_ndarray_ro
        )
        self._readers_lock.release()
        return data

    def assign(self, data: Union[SharedObject._object_types], reallocate=False) -> None:
        """Assign data to SharedMemory (protected by writer lock)
        :param reallocate: whether to force reallocation
        """
        # Check object type
        data, object_type_idx, nbytes, np_metas = self._preprocess_data(data)

        self._writer_lock.acquire()
        # Fetch shm info
        self.nbytes, self.mtime, self.object_type_idx, self.np_metas = (
            self._fetch_metas(self.shm)
        )

        # Reallocate if necessary
        if reallocate or nbytes > self.nbytes or np_metas != self.np_metas:
            # NOTE: Cannot use unlink() to reallocate SharedMemory
            # Otherwise, existing SharedObject instances to the same SharedMemory
            # will not be updated
            # Need to use os.ftruncate(sm._fd, new_size)
            raise NotImplementedError("reallocate is not yet implemented")

        self._assign_objects[self.object_type_idx](
            self.shm.buf, data, self.nbytes, self.np_ndarray
        )
        self._writer_lock.release()
