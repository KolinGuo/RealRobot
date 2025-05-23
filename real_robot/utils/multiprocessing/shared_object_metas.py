# ruff: noqa: UP007
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import ClassVar, Union

import numpy as np

# Size of the entire buffer when a SharedObject contains an object of this type
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
    """
    Metadata for string / bytes / bytearray

    - 8 bytes: object modified timestamp in ns (since the epoch), stored as 'Q'
    - 1 byte: object data type index, stored as 'B'
    - X bytes: data area.
      - 8 bytes: size of the string / bytes / bytearray buffer (N + 1)
      - N bytes: data buffer
      - 1 byte: termination byte (b"\xff")
      - padded zero bytes until length indicated in the first 8 bytes.
    """

    METADATA_BUF_SIZE: ClassVar[int] = 8

    buf_size: int
    """
    Size of the entire buffer when a SharedObject contains a string / bytes / bytearray.
    Including the modified timestamp `mtime` (8 bytes), object type index (1 byte).
    """

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
    """
    Metadata for np.ndarray

    - 8 bytes: object modified timestamp in ns (since the epoch), stored as 'Q'
    - 1 byte: object data type index, stored as 'B'
    - X bytes: data area.
      - 1 byte: array dtype index, stored as 'B'
      - 8 bytes: array ndim, stored as 'Q'
      - (K * 8) bytes: array shape for each dimension, stored as 'Q'
      - D bytes: array data buffer
    """

    buf_size: int
    """
    Size of the entire buffer when a SharedObject contains a np.ndarray.
    Including the modified timestamp `mtime` (8 bytes), object type index (1 byte).
    """

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
    """
    Metadata for python dict

    - 8 bytes: object modified timestamp in ns (since the epoch), stored as 'Q'
    - 1 byte: object data type index, stored as 'B'
    - X bytes: data area.
      - 8 bytes: size of the dict buffer (data buffer size excluding this 8 bytes)
      - 1 byte: key_1 object_type_idx
      - X bytes: key_1 data area
      - 1 byte: value_1 object_type_idx
      - X bytes: value_1 data area
      - 1 byte: key_2 object_type_idx
      - X bytes: key_2 data area
      - 1 byte: value_2 object_type_idx
      - X bytes: value_2 data area
      - ...
    """

    buf_size: int
    """dict entire buffer size"""

    keys_metas: list[tuple[int, BytesMeta | NDArrayMeta | None]]
    """List of meta for keys, as a tuple of (key_object_type_idx, key Metadata)"""

    values_metas: list[tuple[int, META_TYPES]]
    """List of meta for values, as a tuple of (value_object_type_idx, value Metadata)"""

    # FIXME: remove this hack that only supports 1 ndarray
    ndarray_offset: int
    """Offset for ndarray"""

    # FIXME: remove this hack that only supports 1 ndarray
    ndarray_meta: NDArrayMeta | None

    @staticmethod
    def preprocess_data(
        data: Union[_object_types],  # type: ignore  # noqa: F821
        init_size=100,
    ) -> tuple[Union[_object_types], int, int, META_TYPES]:  # type: ignore  # noqa: F821
        from .shared_object import SharedObject

        try:
            object_type_idx = SharedObject._object_types.index(type(data))
        except ValueError as e:
            raise TypeError(f"Not supported object_type: {type(data)}") from e

        # Get shared memory size in bytes
        metadata = None
        if object_type_idx <= 5:  # NoneType, bool, int, float, complex, sapien.Pose
            nbytes = OBJECT_BUF_SIZES[object_type_idx]
        elif object_type_idx == 6:  # str
            data = data.encode("utf8")  # encode strings into bytes
            # TODO: Change self.init_size to str
            # TODO: Allow filling up to the full data_buf_size w/o expanding on assign()
            metadata = BytesMeta.from_data(data, init_size)
            nbytes = metadata.buf_size
        elif 7 <= object_type_idx <= 8:  # bytes, bytearray
            metadata = BytesMeta.from_data(data, init_size)
            nbytes = metadata.buf_size
        elif object_type_idx == 9:  # np.ndarray
            metadata = NDArrayMeta.from_data(data)
            nbytes = metadata.buf_size
        else:
            raise ValueError(f"Unknown {object_type_idx = }")

        return data, object_type_idx, nbytes, metadata

    @classmethod
    def from_data(cls, data: dict, init_size=100) -> DictMeta:
        # FIXME: remove
        ndarray_offset = 0
        ndarray_meta = None

        buf_size = 17  # 8 + 1 + 8
        keys_metas = []
        values_metas = []
        for key, value in data.items():
            data, object_type_idx, nbytes, metadata = cls.preprocess_data(
                key, init_size
            )
            buf_size += nbytes - 8  # 8 bytes mtime
            keys_metas.append((object_type_idx, metadata))

            data, object_type_idx, nbytes, metadata = cls.preprocess_data(
                value, init_size
            )
            buf_size += nbytes - 8  # 8 bytes mtime
            values_metas.append((object_type_idx, metadata))

            # FIXME: remove
            if isinstance(metadata, NDArrayMeta):
                if ndarray_meta:
                    raise NotImplementedError("Only support 1 ndarray in dict")
                ndarray_offset = buf_size - metadata.data_buf_size
                ndarray_meta = metadata

        return cls(
            buf_size=buf_size,
            keys_metas=keys_metas,
            values_metas=values_metas,
            ndarray_offset=ndarray_offset,
            ndarray_meta=ndarray_meta,
        )

    def assign_buf(self, buf: memoryview, *, offset: int = 9) -> None:
        """Assign metadata to buffer"""
        struct.pack_into("Q", buf, offset, self.buf_size - 17)
        offset += 8  # dict, buf_size
        for (key_obj_type_idx, key_meta), (value_obj_type_idx, value_meta) in zip(
            self.keys_metas, self.values_metas
        ):
            buf[offset] = key_obj_type_idx
            offset += 1
            if key_meta:
                key_meta.assign_buf(buf, offset=offset)
                offset += key_meta.buf_size - 9  # mtime + object_type_idx
            else:
                offset += (
                    OBJECT_BUF_SIZES[key_obj_type_idx] - 9  # mtime + object_type_idx
                )

            buf[offset] = value_obj_type_idx
            offset += 1
            if value_meta:
                value_meta.assign_buf(buf, offset=offset)
                offset += value_meta.buf_size - 9  # mtime + object_type_idx
            else:
                offset += (
                    OBJECT_BUF_SIZES[value_obj_type_idx] - 9  # mtime + object_type_idx
                )

    @staticmethod
    def _from_buf(buf, *, offset: int = 9) -> tuple[int, int, META_TYPES]:
        metadata = None
        object_type_idx = buf[offset]
        if object_type_idx <= 5:
            offset += OBJECT_BUF_SIZES[object_type_idx] - 8  # mtime
        elif 6 <= object_type_idx <= 8:
            metadata = BytesMeta.from_buf(buf, offset=offset + 1)
            offset += metadata.buf_size - 8  # mtime
        elif object_type_idx == 9:
            metadata = NDArrayMeta.from_buf(buf, offset=offset + 1)
            offset += metadata.buf_size - 8  # mtime
        elif object_type_idx == 10:  # dict
            metadata = DictMeta.from_buf(buf, offset=offset + 1)
            offset += metadata.buf_size - 8  # mtime
        else:
            raise ValueError(f"Unknown {object_type_idx = }")

        return (offset, object_type_idx, metadata)

    @classmethod
    def from_buf(cls, buf: memoryview, *, offset: int = 9) -> DictMeta:
        """Construct metadata from buffer"""
        data_buf_size = struct.unpack_from("Q", buf, offset=offset)[0]
        offset += 8  # dict, buf_size
        end = offset + data_buf_size

        # FIXME: remove
        ndarray_offset = 0
        ndarray_meta = None

        keys_metas = []
        values_metas = []
        while offset < end:
            offset, key_obj_type_idx, metadata = cls._from_buf(buf, offset=offset)
            keys_metas.append((key_obj_type_idx, metadata))
            offset, value_obj_type_idx, metadata = cls._from_buf(buf, offset=offset)
            values_metas.append((value_obj_type_idx, metadata))

            # FIXME: remove
            if isinstance(metadata, NDArrayMeta):
                if ndarray_meta:
                    raise NotImplementedError("Only support 1 ndarray in dict")
                ndarray_offset = offset - metadata.data_buf_size
                ndarray_meta = metadata

        return cls(
            buf_size=data_buf_size + 17,
            keys_metas=keys_metas,
            values_metas=values_metas,
            ndarray_offset=ndarray_offset,
            ndarray_meta=ndarray_meta,
        )


META_TYPES = BytesMeta | NDArrayMeta | DictMeta | None
