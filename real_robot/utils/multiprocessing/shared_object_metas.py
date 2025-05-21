from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

OBJECT_BUF_SIZES = (
    9,  # NoneType
    10,  # bool
    17,  # int
    17,  # float
    25,  # complex
    37,  # sapien.Pose
)


@dataclass
class BytesMeta:
    """Metadata for string / bytes / bytearray"""

    buf_size: int
    """Size of the entire buffer"""

    data_buf_size: int
    """Size of the string / bytes / bytearray buffer (N + 1)"""

    @classmethod
    def from_data(cls, data: bytes | bytearray, init_size=100) -> BytesMeta:
        # 8 + 1 + 8 + N + 1 = N + 18
        if (sz := len(data) << 1) >= init_size:
            return cls(buf_size=sz + 18, data_buf_size=sz + 1)
        else:
            return cls(buf_size=init_size + 18, data_buf_size=init_size + 1)

    def assign_buf(self, buf: memoryview, *, offset: int = 9) -> None:
        """Assign metadata to buffer"""
        struct.pack_into("Q", buf, offset, self.data_buf_size)

    @classmethod
    def from_buf(cls, buf: memoryview, *, offset: int = 9) -> BytesMeta:
        """Construct metadata from buffer"""
        data_buf_size = struct.unpack_from("Q", buf, offset=offset)[0]  # (N + 1) bytes
        return cls(buf_size=data_buf_size + 17, data_buf_size=data_buf_size)


NP_DTYPES = (
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


@dataclass
class NDArrayMeta:
    """Metadata for np.ndarray"""

    buf_size: int
    """Size of the entire buffer"""

    data_buf_size: int
    """Size of the np.ndarray"""

    dtype_idx: int
    """np.ndarray dtype index"""

    ndim: int
    """np.ndarray ndim"""

    shape: tuple[int, ...]
    """np.ndarray shape as a tuple of ints"""

    @classmethod
    def from_data(cls, data: np.ndarray) -> NDArrayMeta:
        try:
            return cls(
                buf_size=data.nbytes + data.ndim * 8 + 18,  # 8 + 1 + 1 + 8 + ndim * 8
                data_buf_size=data.nbytes,
                dtype_idx=NP_DTYPES.index(data.dtype),
                ndim=data.ndim,
                shape=data.shape,
            )
        except ValueError as e:
            raise TypeError(f"Not supported numpy dtype: {data.dtype}") from e

    def assign_buf(self, buf: memoryview, *, offset: int = 9) -> None:
        """Assign metadata to buffer"""
        struct.pack_into(
            "=BQ" + "Q" * self.ndim, buf, offset, self.dtype_idx, self.ndim, *self.shape
        )

    @classmethod
    def from_buf(cls, buf: memoryview, *, offset: int = 9) -> NDArrayMeta:
        """Construct metadata from buffer"""
        dtype_idx, ndim = struct.unpack_from("=BQ", buf, offset=offset)
        shape = struct.unpack_from("Q" * ndim, buf, offset=offset + 9)

        data_buf_size = int(np.prod(shape) * NP_DTYPES[dtype_idx]().nbytes)
        return cls(
            buf_size=data_buf_size + ndim * 8 + 18,
            data_buf_size=data_buf_size,
            dtype_idx=dtype_idx,
            ndim=ndim,
            shape=shape,
        )


@dataclass
class DictMeta:
    """Metadata for python dict"""

    buf_size: int
    """dict entire buffer size"""

    keys_metas: list[tuple[int, BytesMeta | NDArrayMeta | None]]
    """List of meta for keys, as a tuple of (key_object_type_idx, key Metadata)"""

    values_metas: list[tuple[int, BytesMeta | NDArrayMeta | DictMeta | None]]
    """List of meta for values, as a tuple of (value_object_type_idx, value Metadata)"""

    def assign_buf(self, buf: memoryview, *, offset: int = 9) -> None:
        """Assign metadata to buffer"""
        raise NotImplementedError()


META_TYPES = BytesMeta | NDArrayMeta | DictMeta | None
