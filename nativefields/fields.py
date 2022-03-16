from abc import ABC, abstractmethod
import struct
from typing import Iterable, Tuple
import numpy as np
import copy

class StructBase(type):
    def __new__(cls, name, bases, attrs):
        struct_size = 0
        for key, field in attrs.copy().items():
            if key.startswith('_'):
                continue

            attrs[key] = property(field._getvalue, field._setvalue)
            attrs[f'{key}_field'] = field
            struct_size = max(struct_size, field.offset + field.size)

        attrs['get_size'] = staticmethod(lambda: struct_size)
        return super(StructBase, cls).__new__(cls, name, bases, attrs)


class NativeField(ABC):
    offset: int
    size: int

    @abstractmethod
    def _getvalue(self, native_struct):
        pass

    @abstractmethod
    def _setvalue(self, native_struct, value):
        pass


def _translate_offset(offset: Tuple[NativeField, int]):
    if isinstance(offset, int):
        return offset
    
    return offset.offset + offset.size


class NativeStruct(metaclass=StructBase):
    def __init__(self, data: bytearray = None, master_offset=0, **kwargs):
        if data:
            self.data = data
        else:
            self.data = bytearray(self.get_size())
        self.master_offset = master_offset

        for key in kwargs:
            setattr(self, key, kwargs[key])

    def _calc_offset(self, native_field):
        return self.master_offset + native_field.offset


class StructField(NativeField):
    def __init__(self, offset: Tuple[NativeField, int], struct_type: type):
        assert issubclass(struct_type, NativeStruct), 'struct_type must be an inheritant of type NativeStruct'
        self.offset = _translate_offset(offset)
        self.struct_type = struct_type
        self.size = struct_type.get_size()

    def _getvalue(self, native_struct: NativeStruct):
        inner = self.struct_type(native_struct.data, master_offset=self.offset)
        return inner

    def _setvalue(self, native_struct: NativeStruct, value):
        offset = native_struct._calc_offset(self)
        native_struct.data[offset:offset + self.size] = value[:]


class SimpleField(NativeField):
    def __init__(self, offset: Tuple[NativeField, int], struct_format: str, **kwargs):
        self.offset = _translate_offset(offset)
        self.format = struct_format
        self.size = struct.calcsize(struct_format)

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
    def __init__(self, offset: Tuple[NativeField, int], length: int):
        self.offset = _translate_offset(offset)
        self.size = length

    def _getvalue(self, native_struct: NativeStruct):
        offset = native_struct._calc_offset(self)
        if offset + self.size > len(native_struct.data):
            raise Exception('Failed to get value: field is out of bounds')

        return native_struct.data[offset:offset + self.size]

    def _setvalue(self, native_struct: NativeStruct, value):
        offset = native_struct._calc_offset(self)
        if offset + self.size > len(native_struct.data):
            raise Exception(
                f'Failed to set value: field at offset {offset} and size {self.size} is out of bounds for struct with size {len(native_struct.data)}'
            )

        native_struct.data[offset:offset + self.size] = value


class IntegerField(SimpleField):
    def __init__(self, offset: Tuple[NativeField, int], signed: bool = True, size: int = 4):
        if size == 4:
            super().__init__(offset, 'i' if signed else 'I')
        elif size == 2:
            super().__init__(offset, 'h' if signed else 'H')
        elif size == 1:
            super().__init__(offset, 'b' if signed else 'B')
        else:
            raise Exception('size has to be either 4, 2 or 1')


class DoubleField(SimpleField):
    def __init__(self, offset: Tuple[NativeField, int]):
        super().__init__(offset, 'd')


class FloatField(SimpleField):
    def __init__(self, offset: Tuple[NativeField, int]):
        super().__init__(offset, 'f')


class BooleanField(IntegerField):
    def __init__(self, offset: Tuple[NativeField, int], **kwargs):
        super().__init__(offset, signed=False, **kwargs)

    def _getvalue(self, native_struct: NativeStruct) -> bool:
        return bool(super()._getvalue(native_struct))

    def _setvalue(self, native_struct: NativeStruct, value: bool):
        return super()._setvalue(native_struct, int(value))


class StringField(SimpleField):
    def __init__(self, offset: Tuple[NativeField, int], length: int, encoding='utf-8'):
        self.encoding = encoding
        super().__init__(offset, f'{length}s')

    def _getvalue(self, native_struct: NativeStruct) -> str:
        return super()._getvalue(native_struct).decode(self.encoding)

    def _setvalue(self, native_struct: NativeStruct, value: str):
        return super()._setvalue(native_struct, value.encode(self.encoding))


class ArrayField(NativeField):
    def __init__(self, offset: Tuple[NativeField, int], shape: tuple, elem_field_type: type, **kwargs) -> None:
        self.offset = _translate_offset(offset)
        self.shape = (shape,) if isinstance(shape, int) else shape[:]
        if issubclass(elem_field_type, NativeField):
            self._elem_field = elem_field_type(0, **kwargs)
        else:
            self._elem_field = StructField(0, elem_field_type)

        self.size = self._elem_field.size * np.prod(self.shape)

    @staticmethod
    def get_array_index(shape: tuple, index: tuple) -> int:
        assert shape, 'shape cannot be an empty list'
        assert len(shape) == len(index), 'shape and index need to be the same length'
        return sum([index[i] * int(np.prod(shape[i + 1:])) for i in range(len(index) - 1)]) + index[-1]

    def _getvalue(self, native_struct: NativeStruct) -> np.array:
        arr = np.empty(self.shape, dtype=object)
        
        for index in np.ndindex(self.shape):
            self._elem_field.offset = self.offset + ArrayField.get_array_index(self.shape, index) * self._elem_field.size
            arr[index] = self._elem_field._getvalue(native_struct)

        return arr

    def _setvalue(self, native_struct: NativeStruct, value: Iterable):
        if isinstance(value, np.ndarray):
            assert value.shape == self.shape, f'array shape {value.shape} not matching, should be {self.shape}'

        arr = np.empty(self.shape, dtype=object)
        arr[:] = value

        for index in np.ndindex(self.shape):
            self._elem_field.offset = self.offset + ArrayField.get_array_index(self.shape, index) * self._elem_field.size
            self._elem_field._setvalue(native_struct, arr[index])


def unpack_bytes(data: bytearray, field: NativeField):
    elem = NativeStruct(data)
    return field._getvalue(elem)


def pack_value(value, field: NativeField):
    elem = NativeStruct(bytearray(field.offset + field.size))
    field._setvalue(elem, value)
    return elem