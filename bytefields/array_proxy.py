

from bytefields.format import _format_bytearray, _format_numpy
from bytefields import base, fields
import numpy as np


def _to_absolute_indices(shape: tuple, index: tuple) -> tuple:
    index = list(index)
    for i, ind in enumerate(index):
        index[i] = shape[i] + ind if ind < 0 else ind
        if index[i] < 0 or index[i] > shape[i] - 1:
            return None

    return index


def _get_array_index(shape: tuple, index: tuple) -> int:
    assert shape, 'shape cannot be an empty list'
    assert len(shape) == len(index), 'shape and index need to be the same length'
    return sum([index[i] * int(np.prod(shape[i + 1:])) for i in range(len(index) - 1)]) + index[-1]


class ArrayFieldProxy:
    def __init__(self, byte_struct: base.ByteStruct, field: base.ByteField):
        self.byte_struct = byte_struct
        self.field = field

    @property
    def shape(self):
        return self.field.shape

    def to_numpy(self):
        arr = np.empty(self.shape, dtype=object)
        arr_offset = self.byte_struct.calc_field_offset(self.field)
        is_struct_field = isinstance(self.field._elem_field, fields.StructField)
        for index in np.ndindex(self.shape):
            self.field._elem_field.offset = (
                arr_offset + _get_array_index(self.shape, index) * self.field._elem_field.size
            )

            if is_struct_field:
                self.field._elem_field.reset()

            arr[index] = self.field._elem_field._getvalue(self.byte_struct)

        return arr

    def __getitem__(self, index):
        index = self._validate_index(index)

        arr_offset = self.byte_struct.calc_field_offset(self.field)
        self.field._elem_field.offset = (
            arr_offset + _get_array_index(self.shape, index) * self.field._elem_field.size
        )

        if isinstance(self.field._elem_field, fields.StructField):
            self.field._elem_field.reset()

        return self.field._elem_field._getvalue(self.byte_struct)

    def __setitem__(self, index, value):
        index = self._validate_index(index)

        arr_offset = self.byte_struct.calc_field_offset(self.field)
        self.field._elem_field.offset = (
            arr_offset + _get_array_index(self.shape, index) * self.field._elem_field.size
        )

        return self.field._elem_field._setvalue(self.byte_struct, value)

    def _validate_index(self, index):
        if isinstance(index, int):
            index = (index,)

        assert not any(map(lambda elem: isinstance(elem, slice), index)), \
            'slices are not supported with ArrayFieldProxy'

        user_index = index[:]
        index = _to_absolute_indices(self.shape, index)
        assert index, f'Index {user_index} is out of bounds for shape {self.shape}'
        return index

    def __len__(self):
        return self.shape[0]

    def __repr__(self) -> str:
        arr_repr = _format_numpy(self.to_numpy())
        return f'[{self.__class__.__name__} object at {hex(id(self))}]\nData: {arr_repr}'


class ByteArrayFieldProxy:
    def __init__(self, byte_struct: base.ByteStruct, field: base.ByteField):
        self.byte_struct = byte_struct
        self.field = field

    def to_bytearray(self):
        offset = self._validate_offset()
        return self.byte_struct.data[offset:offset + self.field.size]

    def __getitem__(self, index: int):
        offset = self._validate_offset()
        index = self._validate_index(index)
        return self.byte_struct.data[offset + index]

    def __setitem__(self, index: int, value):
        offset = self._validate_offset()
        index = self._validate_index(index)
        self.byte_struct.data[offset + index] = value

    def _validate_offset(self):
        offset = self.byte_struct.calc_offset(self.field)
        if offset + self.field.size > len(self.byte_struct.data):
            raise Exception('Failed to get value: field is out of bounds')

        return offset

    def _validate_index(self, index: int):
        user_index = index
        index = _to_absolute_indices((self.field.size,), (index,))
        assert index, f'Index {user_index} is out of bounds for size {self.field.size}'
        return index[0]

    def __len__(self):
        return self.field.size

    def __repr__(self) -> str:
        bytes_repr = _format_bytearray(self.to_bytearray())
        return f'[{self.__class__.__name__} object at {hex(id(self))}]\nData: {bytes_repr}'
