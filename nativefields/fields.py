from copy import deepcopy
from enum import Enum
import struct
from typing import Iterable, Tuple
import numpy as np
from nativefields.base import NativeStruct, NativeField


class StructField(NativeField):
    def __init__(self, offset: Tuple[NativeField, int], struct_type: type, **kwargs):
        assert issubclass(struct_type, NativeStruct), 'struct_type must be an inheritant of type NativeStruct'
        self.offset = offset
        self.struct_type = struct_type
        self.size = struct_type.min_size
        self.is_instance = True
        self.inner = None
        super().__init__(**kwargs)

    def _getvalue(self, native_struct: NativeStruct):
        # We have to update both the data and master_offset of the inner struct:
        # the inner data can come from user code and master_offset is dependant
        # on the sizing of dynamic fields
        if not self.inner:
            self.inner = self.struct_type(native_struct.data)
        else:
            self.inner.data = native_struct.data

        self.inner.master_offset = native_struct._calc_offset(self)
        return self.inner

    def _setvalue(self, native_struct: NativeStruct, value):
        old_size = self.size
        self.size = value.size
        if self.size != old_size:
            native_struct._resize_data(self, old_size)

        offset = native_struct._calc_offset(self)
        native_struct.data[offset:offset + self.size] = value.data[:]
        value.data = bytearray()
        self.inner = deepcopy(value)


class SimpleField(NativeField):
    def __init__(self, offset: Tuple[NativeField, int], struct_format: str, **kwargs):
        self.offset = offset
        self.format = struct_format
        self.size = struct.calcsize(struct_format)
        super().__init__(**kwargs)

    def _getvalue(self, native_struct: NativeStruct):
        offset = native_struct._calc_offset(self)
        if offset + self.size > len(native_struct.data):
            raise Exception('Failed to get value: field is out of bounds')

        return struct.unpack(self.format, native_struct.data[offset:offset + self.size])[0]

    def _setvalue(self, native_struct: NativeStruct, value):
        offset = native_struct._calc_offset(self)
        if offset + self.size > len(native_struct.data):
            raise Exception('Failed to set value: field is out of bounds')

        native_struct.data[offset:offset + self.size] = struct.pack(self.format, value)


class ByteArrayField(NativeField):
    def __init__(self, offset: Tuple[NativeField, int], length: int, **kwargs):
        self.offset = offset
        if length is None:
            self.size = 0
            self.is_instance = True
        else:
            self.size = length

        super().__init__(**kwargs)

    def resize(self, length: int):
        self.size = length

    def _getvalue(self, native_struct: NativeStruct):
        offset = native_struct._calc_offset(self)
        if offset + self.size > len(native_struct.data):
            raise Exception('Failed to get value: field is out of bounds')

        return native_struct.data[offset:offset + self.size]

    def _setvalue(self, native_struct: NativeStruct, value):
        new_length = len(value)
        if self.is_instance and new_length != self.size:
            self._resize_with_data(native_struct, new_length)
        else:
            assert new_length == self.size, f'bytearray size {new_length} not matching, should be {self.size}'

        offset = native_struct._calc_offset(self)
        if offset + self.size > len(native_struct.data):
            raise Exception(
                f'Failed to set value: field at offset {offset} and size {self.size} '
                f'is out of bounds for struct with size {len(native_struct.data)}'
            )

        native_struct.data[offset:offset + self.size] = value


class Endianness(Enum):
    NATIVE = 0,
    LITTLE = 1,
    BIG = 2


class IntegerField(SimpleField):
    def __init__(
        self,
        offset:
        Tuple[NativeField, int],
        signed: bool = True,
        size: int = 4,
        endianness: Endianness = Endianness.NATIVE,
        **kwargs
    ):
        prefix = '@'
        if endianness == Endianness.BIG:
            prefix = '>'
        elif endianness == Endianness.LITTLE:  # NOQA
            prefix = '<'

        if size == 4:
            super().__init__(offset, f'{prefix}i' if signed else f'{prefix}I', **kwargs)
        elif size == 2:
            super().__init__(offset, f'{prefix}h' if signed else f'{prefix}H', **kwargs)
        elif size == 1:
            super().__init__(offset, f'{prefix}b' if signed else f'{prefix}B', **kwargs)
        elif size == 8:
            super().__init__(offset, f'{prefix}q' if signed else f'{prefix}Q', **kwargs)
        else:
            raise Exception('size has to be either 8, 4, 2 or 1')


class DoubleField(SimpleField):
    def __init__(self, offset: Tuple[NativeField, int], **kwargs):
        super().__init__(offset, 'd', **kwargs)


class FloatField(SimpleField):
    def __init__(self, offset: Tuple[NativeField, int], **kwargs):
        super().__init__(offset, 'f', **kwargs)


class BooleanField(IntegerField):
    def __init__(self, offset: Tuple[NativeField, int], **kwargs):
        super().__init__(offset, signed=False, **kwargs)

    def _getvalue(self, native_struct: NativeStruct) -> bool:
        return bool(super()._getvalue(native_struct))

    def _setvalue(self, native_struct: NativeStruct, value: bool):
        return super()._setvalue(native_struct, int(value))


class StringField(SimpleField):
    def __init__(self, offset: Tuple[NativeField, int], length: int, encoding='utf-8', **kwargs):
        self.encoding = encoding
        if length is None:
            self.is_instance = True
            super().__init__(offset, '0s', **kwargs)
        else:
            super().__init__(offset, f'{length}s', **kwargs)

    def resize(self, length: int):
        self.size = length
        self.format = f'{self.size}s'

    def _getvalue(self, native_struct: NativeStruct) -> str:
        return super()._getvalue(native_struct).decode(self.encoding)

    def _setvalue(self, native_struct: NativeStruct, value: str):
        new_length = len(value)
        if self.is_instance and new_length != self.size:
            self._resize_with_data(native_struct, new_length)

        return super()._setvalue(native_struct, value.encode(self.encoding))


class ArrayField(NativeField):
    def __init__(
        self,
        offset: Tuple[NativeField, int],
        shape: Tuple[tuple, int],
        elem_field_type: type,
        **kwargs
    ):
        self.offset = offset

        if issubclass(elem_field_type, NativeField):
            self._elem_field = elem_field_type(0, **kwargs)
        else:
            self._elem_field = StructField(0, elem_field_type)

        if shape:
            self.shape = (shape,) if isinstance(shape, int) else shape[:]
        else:
            self.shape = (0,)
            self.is_instance = True

        self._update_size()

    def resize(self, shape: Tuple[tuple, int]):
        self.shape = (shape,) if isinstance(shape, int) else shape[:]
        self._update_size()

    def _update_size(self):
        self.size = self._elem_field.size * int(np.prod(self.shape))

    @staticmethod
    def get_array_index(shape: tuple, index: tuple) -> int:
        assert shape, 'shape cannot be an empty list'
        assert len(shape) == len(index), 'shape and index need to be the same length'
        return sum([index[i] * int(np.prod(shape[i + 1:])) for i in range(len(index) - 1)]) + index[-1]

    def _getvalue(self, native_struct: NativeStruct) -> np.array:
        arr = np.empty(self.shape, dtype=object)

        arr_offset = self.real_offset
        is_struct_field = isinstance(self._elem_field, StructField)

        for index in np.ndindex(self.shape):
            self._elem_field.offset = (
                arr_offset + ArrayField.get_array_index(self.shape, index) * self._elem_field.size
            )

            if is_struct_field:
                arr[index] = deepcopy(self._elem_field)._getvalue(native_struct)
            else:
                arr[index] = self._elem_field._getvalue(native_struct)

        return arr

    def _setvalue(self, native_struct: NativeStruct, value: Iterable):
        if isinstance(value, np.ndarray):
            assert value.shape == self.shape, f'array shape {value.shape} not matching, should be {self.shape}'

        value_shape = np.array(value).shape
        if self.is_instance and value_shape != self.shape:
            self._resize_with_data(native_struct, value_shape)

        arr = np.empty(self.shape, dtype=object)
        arr[:] = value

        arr_offset = self.real_offset
        for index in np.ndindex(self.shape):
            self._elem_field.offset = (
                arr_offset + ArrayField.get_array_index(self.shape, index) * self._elem_field.size
            )

            self._elem_field._setvalue(native_struct, arr[index])


class VariableField(NativeField):
    def __init__(self, offset: Tuple[NativeField, int], **kwargs):
        self.offset = offset
        self.size = 0
        self.child = None
        self.is_instance = True

    def resize(self, child: NativeField):
        self.size = child.size
        self.child = child
        self.child.offset = self.offset

    def _getvalue(self, native_struct: NativeStruct):
        if not self.child:
            raise Exception('VariableField does not contain any field')

        return self.child._getvalue(native_struct)

    def _setvalue(self, native_struct: NativeStruct, value):
        if not self.child:
            raise Exception('VariableField does not contain any field')

        val = self.child._setvalue(native_struct, value)
        self.size = self.child.size
        return val


def unpack_bytes(data: Tuple[bytearray, NativeStruct], field: NativeField):
    if isinstance(data, NativeStruct):
        return field._getvalue(data)

    elem = NativeStruct(data)
    return field._getvalue(elem)


def pack_value(value, field: NativeField):
    elem = NativeStruct(bytearray(field.real_offset + field.size))
    field._setvalue(elem, value)
    return elem
