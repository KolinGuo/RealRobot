"""
Shared object implemented with SharedMemory and synchronization
version 0.0.1

Written by Kolin Guo
"""
import struct
from multiprocessing.shared_memory import SharedMemory

import numpy as np

from ..logger import get_logger

try:
    import fcntl
except ModuleNotFoundError as e:
    print("Not supported on Windows")
    raise e


_logger = get_logger("SharedObject")
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
    - 1 byte: object data type index, stored as 'B'
    - X bytes: data area
      For NoneType, data area is ignored
      For bool, 1 byte data
      For int / float, 8 bytes data
      For str / bytes, (N + 1) bytes data, N is str / bytes length, 1 is for termination
      For np.ndarray,
      - 1 byte: array dtype index, stored as 'B'
      - 8 bytes: array ndim, stored as 'Q'
      - (K * 8) bytes: array shape for each dimension, stored as 'Q'
      - D bytes: array data buffer
    """

    _object_types = [None.__class__, bool, int, float, str, bytes, np.ndarray]

    @staticmethod
    def _get_bytes_size(enc_str: bytes, init_size: int) -> int:
        if (sz := len(enc_str) << 1) >= init_size:
            return sz + 2
        else:
            return init_size + 2

    _object_sizes = [
        1, 2, 9, 9,  # NoneType, bool, int, float
        _get_bytes_size,  # str
        _get_bytes_size,  # bytes
        lambda array, ndim: array.nbytes + ndim * 8 + 10,  # ndarray
    ]

    @staticmethod
    def _fetch_shm_metas(shm: SharedMemory) -> tuple:
        # nbytes, object_type_idx, np_metas
        return shm._size, shm.buf[0], SharedObject._fetch_np_metas(shm.buf)

    @staticmethod
    def _fetch_np_metas(buf) -> tuple:
        np_dtype_idx, data_ndim = struct.unpack_from("=BQ", buf, offset=1)
        data_shape = struct.unpack_from("Q" * data_ndim, buf, offset=10)
        return (np_dtype_idx, data_ndim, data_shape)

    @staticmethod
    def _fetch_None(*args) -> None:
        return None

    @staticmethod
    def _fetch_bool(buf, fn, *args) -> bool:
        return bool(buf[1]) if fn is None else fn(bool(buf[1]))

    @staticmethod
    def _fetch_int(buf, fn, *args) -> int:
        v = struct.unpack_from('q', buf, offset=1)[0]
        return v if fn is None else fn(v)

    @staticmethod
    def _fetch_float(buf, fn, *args) -> float:
        v = struct.unpack_from('d', buf, offset=1)[0]
        return v if fn is None else fn(v)

    @staticmethod
    def _fetch_str(buf, fn, *args) -> str:
        v = buf[1:].tobytes().rstrip(b'\x00')[:-1].decode(_encoding)
        return v if fn is None else fn(v)

    @staticmethod
    def _fetch_bytes(buf, fn, *args) -> bytes:
        v = buf[1:].tobytes().rstrip(b'\x00')[:-1]
        return v if fn is None else fn(v)

    @staticmethod
    def _fetch_ndarray(buf, fn, data_buf) -> np.ndarray:
        """Always return a copy"""
        if fn is None:
            _logger.warning(
                "Fetching ndarray with no applied function induces an extra copy"
            )
            return data_buf.copy()
        else:
            return fn(data_buf)

    _fetch_objects = [
        _fetch_None,
        _fetch_bool,
        _fetch_int,
        _fetch_float,
        _fetch_str,
        _fetch_bytes,
        _fetch_ndarray,
    ]

    @staticmethod
    def _assign_np_metas(buf, np_dtype_idx: int, data_ndim: int, data_shape: tuple):
        struct.pack_into("=BQ" + "Q" * data_ndim, buf, 1,
                         np_dtype_idx, data_ndim, *data_shape)

    @staticmethod
    def _assign_None(*args):
        pass

    @staticmethod
    def _assign_bool(buf, data: bool, *args):
        buf[1] = data

    @staticmethod
    def _assign_int(buf, data: int, *args):
        struct.pack_into('q', buf, 1, data)

    @staticmethod
    def _assign_float(buf, data: float, *args):
        struct.pack_into('d', buf, 1, data)

    @staticmethod
    def _assign_bytes(buf, enc_data: bytes, buf_nbytes: int, *args):
        struct.pack_into(f"{buf_nbytes-1}s", buf, 1, enc_data+b'\xff')

    @staticmethod
    def _assign_ndarray(buf, data: np.ndarray, buf_nbytes: int, data_buf: np.ndarray):
        data_buf[:] = data

    _assign_objects = [
        _assign_None,
        _assign_bool,
        _assign_int,
        _assign_float,
        _assign_bytes,
        _assign_bytes,
        _assign_ndarray,
    ]
    _np_dtypes = [
        np.bool_, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32,
        np.int64, np.uint64,
        np.float16, np.float32, np.float64, np.float128,
        np.complex64, np.complex128, np.complex256
    ]

    def __init__(self, name, *, data=None, init_size=100):
        """
        Example:
            # Mounts SharedMemory "test" if exists,
            # Else creates SharedMemory "test" which holds None by default
            so = SharedObject("test")

            # Mounts SharedMemory "test" if exists and assign data (True) to it,
            # Else creates SharedMemory "test" and assigns data
            so = SharedObject("test", data=True)

            # Mounts SharedMemory "test" if exists and assign data (np.ones(10)) to it,
            # Else creates SharedMemory "test" and assigns data
            so = SharedObject("test", data=np.ones(10))

        :param init_size: only used for str, bytes and np.ndarray shape segment,
                          initial buffer size to save frequent reallocation
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
            self.object_type_idx = object_type_idx
            self.np_metas = np_metas
        else:
            self._readers_lock.acquire()
            self.nbytes, self.object_type_idx, self.np_metas \
                = self._fetch_shm_metas(self.shm)
            self._readers_lock.release()

        # Create np.ndarray here to save frequent np.ndarray construction
        self.np_ndarray = None
        if self.object_type_idx == 6:  # np.ndarray
            np_dtype_idx, data_ndim, data_shape = self.np_metas
            self.np_ndarray = np.ndarray(
                data_shape, dtype=self._np_dtypes[np_dtype_idx],
                buffer=self.shm.buf, offset=data_ndim * 8 + 10
            )

        # fill data
        if data is not None:
            if not created:
                _logger.warning(f"Implicitly overwriting data of {self!r}")
            self._assign(data, object_type_idx, nbytes, np_metas)

    def _preprocess_data(self, data):
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
        if object_type_idx <= 3:  # NoneType, bool, int, float
            nbytes = self._object_sizes[object_type_idx]
        elif object_type_idx == 4:  # str
            data = data.encode(_encoding)  # encode strings into bytes
            nbytes = self._object_sizes[object_type_idx](data, self.init_size)
        elif object_type_idx == 5:  # bytes
            nbytes = self._object_sizes[object_type_idx](data, self.init_size)
        elif object_type_idx == 6:  # np.ndarray
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

    def fetch(self, fn=None):
        """Fetch a copy of data from SharedMemory (protected by readers lock)
        :param fn: function to apply on data, e.g., lambda x: x + 1.
        :return data: a copy of data read from SharedMemory
        """
        self._readers_lock.acquire()
        data = self._fetch_objects[self.object_type_idx](self.shm.buf, fn,
                                                         self.np_ndarray)
        self._readers_lock.release()
        return data

    def assign(self, data) -> None:
        """Assign data to SharedMemory (protected by writer lock)"""
        self._assign(*self._preprocess_data(data))

    def _assign(self, data, object_type_idx: int, nbytes: int, np_metas: tuple) -> None:
        """Inner function for assigning data (protected by writer lock)"""
        if (object_type_idx != self.object_type_idx or nbytes > self.nbytes
                or np_metas != self.np_metas):
            raise ValueError(
                f"Casting object type (new={self._object_types[object_type_idx]}, "
                f"old={self._object_types[self.object_type_idx]}) or "
                f"Buffer overflow (data_nbytes={nbytes} > {self.nbytes}) or "
                f"Changed numpy meta (new={np_metas}, old={self.np_metas}) in {self!r}"
            )

        self._writer_lock.acquire()
        # Assign object type
        self.shm.buf[0] = object_type_idx
        if object_type_idx == 6:  # np.ndarray
            self._assign_np_metas(self.shm.buf, *np_metas)

        # Assign object data
        self._assign_objects[self.object_type_idx](self.shm.buf, data,
                                                   self.nbytes, self.np_ndarray)
        self._writer_lock.release()

    def __reduce__(self):
        return self.__class__, (self.name,)

    def __repr__(self):
        return (f'{self.__class__.__name__}(name={self.name}, '
                f'data_type={self._object_types[self.object_type_idx]}, '
                f'nbytes={self.nbytes})')


class SharedDynamicObject(SharedObject):
    """Shared object implemented with SharedMemory and synchronization
    Allow reallocating SharedMemory.
        Need more checks and thus is slower than SharedObject.
    """

    @staticmethod
    def _fetch_shm_metas(shm: SharedMemory) -> tuple:
        nbytes = shm._mmap.size()  # _mmap size will be updated by os.ftruncate()
        object_type_idx = shm.buf[0]
        np_metas = SharedObject._fetch_np_metas(shm.buf)
        return nbytes, object_type_idx, np_metas

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Implementation not complete")

    def fetch(self, fn=None):
        """Fetch a copy of data from SharedMemory (protected by readers lock)
        :param fn: function to apply on data, e.g., lambda x: x + 1.
        :return data: a copy of data read from SharedMemory
        """
        self._readers_lock.acquire()
        # Fetch shm info
        self.nbytes, self.object_type_idx, self.np_metas \
            = self._fetch_shm_metas(self.shm)

        data = self._fetch_objects[self.object_type_idx](self.shm.buf, fn,
                                                         self.np_ndarray)
        self._readers_lock.release()
        return data

    def assign(self, data, reallocate=False) -> None:
        """Assign data to SharedMemory (protected by writer lock)
        :param reallocate: whether to force reallocation
        """
        # Check object type
        data, object_type_idx, nbytes, np_metas = self._preprocess_data(data)

        self._writer_lock.acquire()
        # Fetch shm info
        self.nbytes, self.object_type_idx, self.np_metas \
            = self._fetch_shm_metas(self.shm)

        # Reallocate if necessary
        if reallocate or nbytes > self.nbytes or np_metas != self.np_metas:
            # NOTE: Cannot use unlink() to reallocate SharedMemory
            # Otherwise, existing SharedObject instances to the same SharedMemory
            # will not be updated
            # Need to use os.ftruncate(sm._fd, new_size)
            raise NotImplementedError("reallocate is not yet implemented")

        self._assign_objects[self.object_type_idx](self.shm.buf, data,
                                                   self.nbytes, self.np_ndarray)
        self._writer_lock.release()
