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
    '''
    An ArrayFieldProxy is a proxy object for ArrayFields that allows modifying and retrieval
    of individual elements without examining the full array.

    ArrayField and ArrayFieldProxy do not support storing
    instanced fields with dynamic sizes. Storing StructFields is supported,
    but in limited capacity. See ArrayField for more detail.

    ArrayFieldProxy supports converting the data into a full numpy array
    with the to_numpy() method, and supports tuple indexing such as

        byte_struct.arr[0, 2] = 100

    Slicing the array or retrieving parts of the array such as rows or columns
    is not supported.

    Attributes:
        byte_struct (ByteStruct): the struct this proxy is attached to
        field (ByteField): the specific field this proxy is attached to

    Args:
        byte_struct (ByteStruct): the struct this proxy to attach to
        field (ByteField): the specific ArrayField to attach to
    '''
    def __init__(self, byte_struct: base.ByteStruct, field: base.ByteField):
        self.byte_struct = byte_struct
        self.field = field

    @property
    def shape(self) -> tuple:
        '''
        Retrieves the shape of this proxy.
        Always returns a tuple, if the array is one dimensional,
        a tuple with one integer is returned

        Returns:
            tuple: the shape of the array proxy
        '''
        data = self.byte_struct._get_instance_data(self.field)
        if not data.get('shape'):
            data['shape'] = self.field.shape

        return data['shape']

    def to_numpy(self) -> np.array:
        '''
        Converts the proxy into a numpy array reading all
        the elements.

        Note that to_numpy does not cache results and the
        time to construct the full array is dependant on
        the size of it.

        Returns:
            np.array: the full numpy array
        '''
        arr = np.empty(self.shape, dtype=object)
        arr_offset = self.byte_struct.calc_field_offset(self.field)
        is_struct_field = isinstance(self.field._elem_field, fields.StructField)
        for index in np.ndindex(self.shape):
            self.field._elem_field.computed_offset = (
                arr_offset + _get_array_index(self.shape, index) * self.field._elem_field.size
            )

            if is_struct_field:
                self.field._elem_field.reset(self.byte_struct)

            arr[index] = self.field._elem_field._getvalue(self.byte_struct)

        return arr

    def __getitem__(self, index):
        index = self._validate_index(index)

        arr_offset = self.byte_struct.calc_field_offset(self.field)
        self.field._elem_field.computed_offset = (
            arr_offset + _get_array_index(self.shape, index) * self.field._elem_field.size
        )

        if isinstance(self.field._elem_field, fields.StructField):
            self.field._elem_field.reset(self.byte_struct)

        return self.field._elem_field._getvalue(self.byte_struct)

    def __setitem__(self, index, value):
        index = self._validate_index(index)

        arr_offset = self.byte_struct.calc_field_offset(self.field)
        self.field._elem_field.computed_offset = (
            arr_offset + _get_array_index(self.shape, index) * self.field._elem_field.size
        )

        return self.field._elem_field._setvalue(self.byte_struct, value)

    def _validate_index(self, index):
        if isinstance(index, int):
            index = (index,)

        if any(map(lambda elem: isinstance(elem, slice), index)):
            raise NotImplementedError('slices are not supported with ArrayFieldProxy')

        user_index = index[:]
        index = _to_absolute_indices(self.shape, index)
        if not index:
            raise IndexError(f'index {user_index} is out of bounds for shape {self.shape}')

        return index

    def __len__(self):
        return self.shape[0]

    def __repr__(self) -> str:
        arr_repr = _format_numpy(self.to_numpy())
        return f'[{self.__class__.__name__} object at {hex(id(self))}]\nData: {arr_repr}'


class ByteArrayFieldProxy:
    '''
    A ByteArrayFieldProxy is a proxy object for ByteArrayFields that allows modifying
    and retrieval of individual elements without examining the full array.

    Just like ArrayFieldProxy, ByteArrayFieldProxy supports
    converting the data into a full bytearray with the to_bytearray() method.

    Slicing the array is not supported.

    Attributes:
        byte_struct (ByteStruct): the struct this proxy is attached to
        field (ByteField): the specific field this proxy is attached to

    Args:
        byte_struct (ByteStruct): the struct this proxy to attach to
        field (ByteField): the specific ArrayField to attach to
    '''
    def __init__(self, byte_struct: base.ByteStruct, field: base.ByteField):
        self.byte_struct = byte_struct
        self.field = field

    def to_bytearray(self) -> bytearray:
        '''
        Converts the proxy into a bytearray reading all
        the elements.

        Note that to_bytearray does not cache results.

        Returns:
            bytearray: the full bytearray
        '''
        offset = self._validate_offset()
        data = self.byte_struct._get_instance_data(self.field)
        return self.byte_struct.data[offset:offset + data['size']]

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
        data = self.byte_struct._get_instance_data(self.field)
        if offset + data['size'] > len(self.byte_struct.data):
            raise IndexError('failed to get value: field is out of bounds')

        return offset

    def _validate_index(self, index: int):
        data = self.byte_struct._get_instance_data(self.field)
        user_index = index
        index = _to_absolute_indices((data['size'],), (index,))
        if not index:
            raise IndexError(f'index {user_index} is out of bounds for size {self.field.size}')

        return index[0]

    def __len__(self):
        data = self.byte_struct._get_instance_data(self.field)
        return data['size']

    def __repr__(self) -> str:
        bytes_repr = _format_bytearray(self.to_bytearray())
        return f'[{self.__class__.__name__} object at {hex(id(self))}]\nData: {bytes_repr}'
