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
    >>> so.fetch(lambda x: x + 1)  # Apply operation only
    >>> so.fetch()  # If different operations need to be applied on the same data
    >>> so.fetch()[..., 0]  # Slice only
    >>> so.fetch(lambda x: x[..., 0].copy()) + 1  # Slice and apply operation
"""
# ruff: noqa: UP007

from __future__ import annotations

import struct
import time
from collections.abc import Callable
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Union

import numpy as np

from real_robot import LOGGER

from .shared_object_metas import (
    META_TYPES,
    NP_DTYPES,
    OBJECT_BUF_SIZES,
    BytesMeta,
    NDArrayMeta,
)

try:
    from sapien import Pose  # type: ignore
except ModuleNotFoundError:
    LOGGER.warning("No sapien installed. Will not support sapien.Pose")

    class Pose:
        def __init__(self) -> None:
            pass

        def __setstate__(self, state) -> None:
            pass

        def __getstate__(self) -> tuple:
            return ()


try:
    import fcntl
except ModuleNotFoundError as e:
    LOGGER.critical("Not supported on Windows: failed to import fcntl")
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
    """
    Shared object implemented with SharedMemory and synchronization.
    SharedMemory reallocation, casting object_type, changing numpy metas are not allowed
    Use SharedDynamicObject instead

    The shared memory buffer is organized as follows:

    - 8 bytes: object modified timestamp in ns (since the epoch), stored as 'Q'
    - 1 byte: object data type index, stored as 'B'
    - X bytes: data area.

      For `NoneType`, data area is ignored.
      For `bool`, 1 byte data.
      For `int` / `float`, 8 bytes data.
      For `complex`, 16 bytes data.
      For `sapien.Pose`, 7*4 = 28 bytes data ([xyz, wxyz], float32).
      For `str` / `bytes` / `bytearray`, (8 + N + 1) bytes data.
        - 8 bytes: size of the string / bytes / bytearray buffer (N + 1)
        - N bytes: data buffer
        - 1 byte: termination byte (b"\xff")
        - padded zero bytes until length indicated in the first 8 bytes.
      For `np.ndarray`,

      - 1 byte: array dtype index, stored as 'B'
      - 8 bytes: array ndim, stored as 'Q'
      - (K * 8) bytes: array shape for each dimension, stored as 'Q'
      - D bytes: array data buffer

      For compound data types (`tuple` / `list` / `set` / `dict`),
      the buffer is organized as follows:

      -
    """

    # object_type_idx:
    _object_types = (
        None.__class__,  # 0
        bool,  # 1
        int,  # 2
        float,  # 3
        complex,  # 4
        Pose,  # 5
        str,  # 6
        bytes,  # 7
        bytearray,  # 8
        np.ndarray,  # 9
        # tuple,  # 10
        # list,  # 11
        # set,  # 12
        dict,  # 13
    )

    @staticmethod
    def _fetch_metas(shm: SharedMemory) -> tuple[int, int, META_TYPES, int]:
        nbytes = shm._size  # type: ignore
        mtime, object_type_idx = struct.unpack_from("QB", shm.buf, offset=0)
        metadata = None
        if 6 <= object_type_idx <= 8:  # str / bytes / bytearray
            metadata = BytesMeta.from_buf(shm.buf)
        elif object_type_idx == 9:  # np.ndarray
            metadata = NDArrayMeta.from_buf(shm.buf)
        elif object_type_idx == 10:  # dict
            metadata = DictMeta.from_buf(shm.buf)
        return object_type_idx, nbytes, metadata, mtime

    _fetch_fn_type = "Callable[[Union[_object_types]], Any] | None"

    @staticmethod
    def _fetch_None(
        buf: memoryview,
        fn: Callable[[None.__class__], Any] | None,
        *args,
        offset: int = 9,
    ) -> Any:
        return None if fn is None else fn(None)

    @staticmethod
    def _fetch_bool(
        buf: memoryview, fn: Callable[[bool], Any] | None, *args, offset: int = 9
    ) -> Any:
        return bool(buf[offset]) if fn is None else fn(bool(buf[offset]))

    @staticmethod
    def _fetch_int(
        buf: memoryview, fn: Callable[[int], Any] | None, *args, offset: int = 9
    ) -> Any:
        v = struct.unpack_from("q", buf, offset=offset)[0]
        return v if fn is None else fn(v)

    @staticmethod
    def _fetch_float(
        buf: memoryview, fn: Callable[[float], Any] | None, *args, offset: int = 9
    ) -> Any:
        v = struct.unpack_from("d", buf, offset=offset)[0]
        return v if fn is None else fn(v)

    @staticmethod
    def _fetch_complex(
        buf: memoryview, fn: Callable[[complex], Any] | None, *args, offset: int = 9
    ) -> Any:
        v = complex(*struct.unpack_from("2d", buf, offset=offset))
        return v if fn is None else fn(v)

    @staticmethod
    def _fetch_pose(
        buf: memoryview, fn: Callable[[Pose], Any] | None, *args, offset: int = 9
    ) -> Any:
        """Fetch and construct a sapien.Pose (using __setstate__)"""
        pose = Pose.__new__(Pose)
        pose.__setstate__(struct.unpack_from("7f", buf, offset=offset))
        return pose if fn is None else fn(pose)

    @staticmethod
    def _fetch_str(
        buf: memoryview, fn: Callable[[str], Any] | None, *args, offset: int = 9
    ) -> Any:
        # TODO: does it need to unpack buf_size
        data_buf_size = struct.unpack_from("Q", buf, offset=offset)[0]  # (N + 1) bytes
        v = (
            buf[offset + 8 : offset + 8 + data_buf_size]
            .tobytes()
            .rstrip(b"\x00")[:-1]
            .decode(_encoding)
        )
        return v if fn is None else fn(v)

    @staticmethod
    def _fetch_bytes(
        buf: memoryview, fn: Callable[[bytes], Any] | None, *args, offset: int = 9
    ) -> Any:
        data_buf_size = struct.unpack_from("Q", buf, offset=offset)[0]  # (N + 1) bytes
        v = buf[offset + 8 : offset + 8 + data_buf_size].tobytes().rstrip(b"\x00")[:-1]
        return v if fn is None else fn(v)

    @staticmethod
    def _fetch_bytearray(
        buf: memoryview, fn: Callable[[bytearray], Any] | None, *args, offset: int = 9
    ) -> Any:
        data_buf_size = struct.unpack_from("Q", buf, offset=offset)[0]  # (N + 1) bytes
        v = bytearray(
            buf[offset + 8 : offset + 8 + data_buf_size].tobytes().rstrip(b"\x00")[:-1]
        )
        return v if fn is None else fn(v)

    @staticmethod
    def _fetch_ndarray(
        buf: memoryview,
        fn: Callable[[np.ndarray], Any] | None,
        data_buf_ro: np.ndarray,
        *args,
        offset: int = 9,
    ) -> Any:
        """
        Always return a copy of the underlying buffer
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
                LOGGER.warning(
                    "Fetching ndarray with fn that does not trigger a copy "
                    "induces an extra copy. Consider changing to improve performance."
                )
                return data.copy()
        else:
            return data_buf_ro.copy()

    @staticmethod
    def _fetch_dict(
        buf: memoryview,
        fn: Callable[[dict], Any] | None,
        data_buf_ro: np.ndarray,
        metadata: DictMeta,
        offset: int = 9,
    ) -> Any:
        data = {}
        offset += 8  # dict, buf_size

        for (key_obj_type_idx, key_meta), (value_obj_type_idx, value_meta) in zip(
            metadata.keys_metas, metadata.values_metas
        ):
            key = SharedObject._fetch_objects[key_obj_type_idx](
                buf, None, offset=offset + 1
            )
            if key_meta:
                offset += key_meta.buf_size - 8  # mtime
            else:
                offset += OBJECT_BUF_SIZES[key_obj_type_idx] - 8  # mtime

            value = SharedObject._fetch_objects[value_obj_type_idx](
                buf, None, data_buf_ro, offset=offset + 1
            )
            if value_meta:
                offset += value_meta.buf_size - 8  # mtime
            else:
                offset += OBJECT_BUF_SIZES[value_obj_type_idx] - 8  # mtime
            data[key] = value

        return data if fn is None else fn(data)

    _fetch_objects = (
        _fetch_None.__func__,  # type: ignore
        _fetch_bool.__func__,  # type: ignore
        _fetch_int.__func__,  # type: ignore
        _fetch_float.__func__,  # type: ignore
        _fetch_complex.__func__,  # type: ignore
        _fetch_pose.__func__,  # type: ignore
        _fetch_str.__func__,  # type: ignore
        _fetch_bytes.__func__,  # type: ignore
        _fetch_bytearray.__func__,  # type: ignore
        _fetch_ndarray.__func__,  # type: ignore
        _fetch_dict.__func__,  # type: ignore
    )

    @staticmethod
    def _assign_None(*args, offset: int = 9):
        pass

    @staticmethod
    def _assign_bool(buf: memoryview, data: bool, *args, offset: int = 9):
        buf[offset] = data

    @staticmethod
    def _assign_int(buf: memoryview, data: int, *args, offset: int = 9):
        struct.pack_into("q", buf, offset, data)

    @staticmethod
    def _assign_float(buf: memoryview, data: float, *args, offset: int = 9):
        struct.pack_into("d", buf, offset, data)

    @staticmethod
    def _assign_complex(buf: memoryview, data: complex, *args, offset: int = 9):
        struct.pack_into("2d", buf, offset, data.real, data.imag)

    @staticmethod
    def _assign_pose(buf: memoryview, pose: Pose, *args, offset: int = 9):
        struct.pack_into("7f", buf, offset, *pose.__getstate__())

    @staticmethod
    def _assign_bytes(
        buf: memoryview, enc_data: bytes, metadata: BytesMeta, *args, offset: int = 9
    ):
        """
        :param metadata: metadata.data_buf_size is the bytes buffer size
            (w/ termination byte) (i.e., N + 1)
        """
        struct.pack_into(
            f"{metadata.data_buf_size}s", buf, offset + 8, enc_data + b"\xff"
        )

    @staticmethod
    def _assign_ndarray(
        buf: memoryview,
        data: np.ndarray,
        metadata: NDArrayMeta,
        data_buf: np.ndarray,
        offset: int = 9,
    ):
        data_buf[:] = data

    @staticmethod
    def _assign_dict(
        buf: memoryview,
        data: dict,
        metadata: DictMeta,
        np_data_buf: np.ndarray,
        offset: int = 9,
    ):
        offset += 8  # dict, buf_size

        for (key, value), (key_obj_type_idx, key_meta), (
            value_obj_type_idx,
            value_meta,
        ) in zip(data.items(), metadata.keys_metas, metadata.values_metas):
            # FIXME: Do not encode again
            if isinstance(key, str):
                key = key.encode(_encoding)  # encode strings into bytes
            SharedObject._assign_objects[key_obj_type_idx](
                buf, key, key_meta, offset=offset + 1
            )
            if key_meta:
                offset += key_meta.buf_size - 8  # mtime
            else:
                offset += OBJECT_BUF_SIZES[key_obj_type_idx] - 8  # mtime

            # FIXME: Do not encode again
            if isinstance(value, str):
                value = value.encode(_encoding)  # encode strings into bytes
            SharedObject._assign_objects[value_obj_type_idx](
                buf, value, value_meta, np_data_buf, offset=offset + 1
            )
            if value_meta:
                offset += value_meta.buf_size - 8  # mtime
            else:
                offset += OBJECT_BUF_SIZES[value_obj_type_idx] - 8  # mtime

    _assign_objects = (
        _assign_None.__func__,  # type: ignore
        _assign_bool.__func__,  # type: ignore
        _assign_int.__func__,  # type: ignore
        _assign_float.__func__,  # type: ignore
        _assign_complex.__func__,  # type: ignore
        _assign_pose.__func__,  # type: ignore
        _assign_bytes.__func__,  # type: ignore
        _assign_bytes.__func__,  # type: ignore
        _assign_bytes.__func__,  # type: ignore
        _assign_ndarray.__func__,  # type: ignore
        _assign_dict.__func__,  # type: ignore
    )

    def __init__(self, name: str, *, data: Union[_object_types] = None, init_size=100):  # type: ignore
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
        data, object_type_idx, nbytes, metadata = self._preprocess_data(data)

        try:
            self.shm = SharedMemory(name)
            created = False
        except FileNotFoundError:  # no SharedMemory with given name
            self.shm = SharedMemory(name, create=True, size=nbytes)
            created = True
        self.name = name
        self._readers_lock = ReadersLock(self.shm._fd)  # type: ignore
        self._writer_lock = WriterLock(self.shm._fd)  # type: ignore

        if created:
            self.object_type_idx = object_type_idx
            self.nbytes = nbytes
            self.metadata = metadata
            self.mtime = time.time_ns()
            # Assign object_type, np_metas to init object meta info
            self._writer_lock.acquire()
            self.shm.buf[8] = object_type_idx
            if metadata is not None:
                metadata.assign_buf(self.shm.buf)
            self._writer_lock.release()
        else:
            self._readers_lock.acquire()
            self.object_type_idx, self.nbytes, self.metadata, self.mtime = (
                self._fetch_metas(self.shm)
            )
            self._readers_lock.release()

        # Create np.ndarray here to save frequent np.ndarray construction
        self.np_ndarray, self.np_ndarray_ro = None, None
        if self.object_type_idx == 9:  # np.ndarray
            self.np_ndarray = np.ndarray(
                self.metadata.shape,  # type: ignore
                dtype=NP_DTYPES[self.metadata.dtype_idx],  # type: ignore
                buffer=self.shm.buf,
                offset=self.metadata.ndim * 8 + 18,  # type: ignore
            )
            # Create a read-only view for fetch()
            self.np_ndarray_ro = self.np_ndarray.view()
            self.np_ndarray_ro.setflags(write=False)
        elif (
            self.object_type_idx == 10 and self.metadata.ndarray_meta  # type: ignore
        ):  # dict
            self.np_ndarray = np.ndarray(
                self.metadata.ndarray_meta.shape,  # type: ignore
                dtype=NP_DTYPES[self.metadata.ndarray_meta.dtype_idx],  # type: ignore
                buffer=self.shm.buf,
                offset=self.metadata.ndarray_offset,  # type: ignore
            )
            # Create a read-only view for fetch()
            self.np_ndarray_ro = self.np_ndarray.view()
            self.np_ndarray_ro.setflags(write=False)

        # fill data
        if data is not None:
            if not created:
                LOGGER.warning("Implicitly overwriting data of {!r}", self)
            self._assign(data, object_type_idx, nbytes, metadata)

    def _preprocess_data(
        self,
        data: Union[_object_types],  # type: ignore
    ) -> tuple[Union[_object_types], int, int, META_TYPES]:  # type: ignore
        """Preprocess object data and return useful informations

        :return data: processed data. Only changed for str (=> encoded bytes)
        :return object_type_idx: object type index
        :return nbytes: number of bytes needed for SharedMemory
        :return metadata: metadata info
        """
        try:
            object_type_idx = self._object_types.index(type(data))
        except ValueError as e:
            raise TypeError(f"Not supported object_type: {type(data)}") from e

        # Get shared memory size in bytes
        metadata = None
        if object_type_idx <= 5:  # NoneType, bool, int, float, complex, sapien.Pose
            nbytes = OBJECT_BUF_SIZES[object_type_idx]
        elif object_type_idx == 6:  # str
            data = data.encode(_encoding)  # encode strings into bytes
            # TODO: Change self.init_size to str
            # TODO: Allow filling up to the full data_buf_size w/o expanding on assign()
            metadata = BytesMeta.from_data(data, self.init_size)
            nbytes = metadata.buf_size
        elif 7 <= object_type_idx <= 8:  # bytes, bytearray
            metadata = BytesMeta.from_data(data, self.init_size)
            nbytes = metadata.buf_size
        elif object_type_idx == 9:  # np.ndarray
            metadata = NDArrayMeta.from_data(data)
            nbytes = metadata.buf_size
        elif object_type_idx == 10:  # dict
            metadata = DictMeta.from_data(data, self.init_size)
            nbytes = metadata.buf_size
        else:
            raise ValueError(f"Unknown {object_type_idx = }")

        return data, object_type_idx, nbytes, metadata

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

    def fetch(self, fn: _fetch_fn_type = None) -> Any:  # type: ignore
        """
        Fetch a copy of data from SharedMemory (protected by readers lock)
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
            self.shm.buf, fn, self.np_ndarray_ro, self.metadata
        )
        self._readers_lock.release()
        return data

    def trigger(self) -> SharedObject:
        """Trigger by modifying object mtime (protected by writer lock)"""
        self._writer_lock.acquire()
        # Update mtime
        struct.pack_into("Q", self.shm.buf, 0, time.time_ns())
        self._writer_lock.release()
        return self

    def assign(self, data: Union[_object_types]) -> SharedObject:  # type: ignore
        """Assign data to SharedMemory (protected by writer lock)"""
        return self._assign(*self._preprocess_data(data))

    def _assign(
        self, data, object_type_idx: int, nbytes: int, metadata: META_TYPES
    ) -> SharedObject:
        """Inner function for assigning data (protected by writer lock)
        For SharedObject, object_type_idx, nbytes, and np_metas cannot be modified
        """
        if (
            object_type_idx != self.object_type_idx
            or nbytes > self.nbytes
            or metadata != self.metadata
        ):
            raise BufferError(
                f"Casting object type (new={self._object_types[object_type_idx]}, "
                f"old={self._object_types[self.object_type_idx]}) OR "
                f"Buffer overflow (new={nbytes} > {self.nbytes}=old) OR "
                f"Changed metadata (new={metadata}, old={self.metadata}) in {self!r}"
            )

        self._writer_lock.acquire()
        # Assign mtime
        self.mtime = time.time_ns()
        struct.pack_into("Q", self.shm.buf, 0, self.mtime)

        # Assign object data
        self._assign_objects[self.object_type_idx](
            self.shm.buf, data, self.metadata, self.np_ndarray
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
    """
    Shared object implemented with SharedMemory and synchronization
    Allow reallocating SharedMemory.

    Need more checks and thus is slower than SharedObject.
    In fact, this should never be implemented.
    For size-variable np.ndarray, just implement similar support as str/bytes
    """

    @staticmethod
    def _fetch_metas(shm: SharedMemory) -> tuple[int, int, META_TYPES, int]:
        nbytes = (
            shm._mmap.size()  # type: ignore
        )  # _mmap size will be updated by os.ftruncate()
        mtime, object_type_idx = struct.unpack_from("QB", shm.buf, offset=0)
        metadata = None
        if 6 <= object_type_idx <= 8:  # str / bytes / bytearray
            metadata = BytesMeta.from_buf(shm.buf)
        elif object_type_idx == 9:  # np.ndarray
            metadata = NDArrayMeta.from_buf(shm.buf)
        elif object_type_idx == 10:  # dict
            metadata = DictMeta.from_buf(shm.buf)
        return object_type_idx, nbytes, metadata, mtime

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Implementation not complete")

    def fetch(self, fn: SharedObject._fetch_fn_type = None) -> Any:  # type: ignore
        """Fetch a copy of data from SharedMemory (protected by readers lock)
        :param fn: function to apply on data, e.g., lambda x: x + 1.
        :return data: a copy of data read from SharedMemory
        """
        self._readers_lock.acquire()
        # Fetch shm info
        self.object_type_idx, self.nbytes, self.metadata, self.mtime = (
            self._fetch_metas(self.shm)
        )

        data = self._fetch_objects[self.object_type_idx](
            self.shm.buf, fn, self.np_ndarray_ro
        )
        self._readers_lock.release()
        return data

    def assign(self, data: Union[SharedObject._object_types], reallocate=False) -> None:  # type: ignore
        """Assign data to SharedMemory (protected by writer lock)
        :param reallocate: whether to force reallocation
        """
        # Check object type
        data, object_type_idx, nbytes, metadata = self._preprocess_data(data)  # noqa: F841

        self._writer_lock.acquire()
        # Fetch shm info
        self.object_type_idx, self.nbytes, self.metadata, self.mtime = (
            self._fetch_metas(self.shm)
        )

        # Reallocate if necessary
        if reallocate or nbytes > self.nbytes or metadata != self.metadata:
            # NOTE: Cannot use unlink() to reallocate SharedMemory
            # Otherwise, existing SharedObject instances to the same SharedMemory
            # will not be updated
            # Need to use os.ftruncate(sm._fd, new_size)
            raise NotImplementedError("reallocate is not yet implemented")

        self._assign_objects[self.object_type_idx](
            self.shm.buf, data, self.nbytes, self.np_ndarray
        )
        self._writer_lock.release()
