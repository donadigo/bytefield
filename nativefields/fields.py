from copy import deepcopy
from enum import Enum
import struct
from typing import Iterable, Tuple
import numpy as np
from nativefields.base import NativeStruct, NativeField


class StructField(NativeField):
    '''
    A StructField provides a way to embed other native structs inside a native struct.

    The data is stored as if the StructField was replaced with the fields from
    the struct that this field holds: that is, the offset of the StructField equals
    the first field of the inner struct + additional offset of the first field, if it
    has any.

    Internally, StructFields pass the bytearray data reference to the child struct and
    set their master offset to the current offset of the StructField. This allows for
    accessing inner fields with native syntax:

        master_struct.inner_struct.inner_int_field = 5

    However, this also means that the value of StructField (a NativeStruct instance),
    is not valid after e.g. resizing a field inside the master struct:

        class Inner(NativeStruct):
            inner_int_field = IntegerField(offset=0)

        class Master(NativeStruct):
            bytes = ByteArrayField(offset=0, length=None)
            inner_struct = StructField(offset=bytes, Inner)

        master_struct = Master()
        inner = master_struct.inner_struct
        master_struct.resize('bytes', 8)

        inner.inner_int_field = 5  # Invalid, the master struct changed its layout and
                                   # the inner reference is now invalid

        master_struct.inner_struct.inner_int_field = 5  # Valid, accessing inner structs this way always
                                                        # makes sure they are valid

        Attributes:
            offset (Tuple[NativeField, int]): the offset of this field
            struct_type (type): the NativeStruct type that this field holds
            size (int): the size of this field in bytes
            is_instance (bool): always True, the field is always only an instance field
            inner: the inner struct that is being stored

        Args:
            offset (Tuple[NativeField, int]): the offset of this field
            struct_type (type): the NativeStruct type that this field holds
    '''
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
            self.inner = self.struct_type(native_struct.data, native_struct.calc_offset(self))
        else:
            self.inner.data = native_struct.data
            self.inner.master_offset = native_struct.calc_offset(self)

        return self.inner

    def _setvalue(self, native_struct: NativeStruct, value):
        old_size = self.size
        self.size = value.size
        if self.size != old_size:
            native_struct._resize_data(self, old_size)

        offset = native_struct.calc_offset(self)
        native_struct.data[offset:offset + self.size] = value.data[:]
        value.data = bytearray()
        self.inner = deepcopy(value)


class SimpleField(NativeField):
    '''
    A SimpleField is a base field to simple type fields such as IntegerField, FloatField,
    DoubleField, StringField and BooleanField. SimpleField interprets the data
    according to the struct module format provided by the user. Visit
    https://docs.python.org/3/library/struct.html to view this format specification.

    Attributes:
        offset (Tuple[NativeField, int]): the offset of this field
        format (str): the format this field uses to interpret the data
        size (int): the size of this field in bytes

    Args:
        offset (Tuple[NativeField, int]): the offset of this field
        struct_format (str): the format this field should use to interpret the data
    '''
    def __init__(self, offset: Tuple[NativeField, int], struct_format: str, **kwargs):
        self.offset = offset
        self.format = struct_format
        self.size = struct.calcsize(struct_format)
        super().__init__(**kwargs)

    def _getvalue(self, native_struct: NativeStruct):
        offset = native_struct.calc_offset(self)
        if offset + self.size > len(native_struct.data):
            raise Exception('Failed to get value: field is out of bounds')

        return struct.unpack(self.format, native_struct.data[offset:offset + self.size])[0]

    def _setvalue(self, native_struct: NativeStruct, value):
        offset = native_struct.calc_offset(self)
        if offset + self.size > len(native_struct.data):
            raise Exception('Failed to set value: field is out of bounds')

        native_struct.data[offset:offset + self.size] = struct.pack(self.format, value)


class ByteArrayField(NativeField):
    '''
    ByteArrayField allows for slicing the underlying bytearray data,
    that can be interpreted later.

    The returned value is a bytearray of size provided by the user, beginning
    at index of the field's offset. The returned data is copied and modifying will
    not change the source bytearray. To modify it, you need to explicitly set the field
    to the new value:

        val = native_struct.byte_field
        val[0] = 200
        native_struct.byte_field = val

    ByteArrayField supports variable sized bytearray chunks. To make the field
    dynamically sized, pass None to the length parameter and call
    native_struct.resize() to resize the field to a new length.

    Args:
        offset (Tuple[NativeField, int]): the offset of this field
        length (int): the length of the sliced array in bytes
    '''
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
        offset = native_struct.calc_offset(self)
        if offset + self.size > len(native_struct.data):
            raise Exception('Failed to get value: field is out of bounds')

        return native_struct.data[offset:offset + self.size]

    def _setvalue(self, native_struct: NativeStruct, value):
        new_length = len(value)
        if self.is_instance and new_length != self.size:
            self._resize_with_data(native_struct, new_length)
        else:
            assert new_length == self.size, f'bytearray size {new_length} not matching, should be {self.size}'

        offset = native_struct.calc_offset(self)
        if offset + self.size > len(native_struct.data):
            raise Exception(
                f'Failed to set value: field at offset {offset} and size {self.size} '
                f'is out of bounds for struct with size {len(native_struct.data)}'
            )

        native_struct.data[offset:offset + self.size] = value


class Endianness(Enum):
    '''
    Describes the endianness of the value.
    Endianness.NATIVE depends to the host endianness.
    '''
    NATIVE = 0,
    LITTLE = 1,
    BIG = 2

    def to_format(self) -> str:
        '''
        Returns the format character according to struct module format.
        Empty if the value == Endianness.NATIVE.

        Returns:
            str: the format character
        '''
        if self == Endianness.LITTLE:
            return '<'
        elif self == Endianness.BIG:
            return '>'

        return ''


class IntegerField(SimpleField):
    '''
    An IntegerField allows for parsing integers of different sizes inside the
    native struct. IntegerField supports specifying the endianness of the integer
    and if it is signed or not.

    Args:
        offset (Tuple[NativeField, int]): the offset of this field
        signed (bool): whether the integer is signed or unsigned, True by default
        size (int): the size of the integer in bytes, either 1, 2, 4 or 8. 4 by default
        endianness (Endianness): the endianness of the integer, Endianness.NATIVE by default
    '''
    def __init__(
        self,
        offset: Tuple[NativeField, int],
        signed: bool = True,
        size: int = 4,
        endianness: Endianness = Endianness.NATIVE,
        **kwargs
    ):
        prefix = endianness.to_format()
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
    '''
    DoubleField parses 8 byte floating point numbers.
    Supports specifying the endianness of the integer.

    Args:
        offset (Tuple[NativeField, int]): the offset of this field
        endianness (Endianness): the endianness of the double, Endianness.NATIVE by default
    '''
    def __init__(self, offset: Tuple[NativeField, int], endianness: Endianness = Endianness.NATIVE, **kwargs):
        prefix = endianness.to_format()
        super().__init__(offset, f'{prefix}d', **kwargs)


class FloatField(SimpleField):
    '''
    FloatField parses 4 byte floating point numbers.
    Supports specifying the endianness of the integer.

    Args:
        offset (Tuple[NativeField, int]): the offset of this field
        endianness (Endianness): the endianness of the double, Endianness.NATIVE by default
    '''
    def __init__(self, offset: Tuple[NativeField, int], endianness: Endianness = Endianness.NATIVE, **kwargs):
        prefix = endianness.to_format()
        super().__init__(offset, f'{prefix}f', **kwargs)


class BooleanField(IntegerField):
    '''
    A BooleanField is an IntegerField, which converts between
    booleans and integers. The conversion is carried out with
    using the bool() and int() casts. That is, a non-zero value
    translates to True and False otherwise.

    By default, BooleanField has a size of 4. You can however
    specify a different size by supplying a size= keyword argument.

    Args:
        offset (Tuple[NativeField, int]): the offset of this field

    Keyword Args:
        size (int): the size of the BooleanField in bytes
    '''
    def __init__(self, offset: Tuple[NativeField, int], **kwargs):
        super().__init__(offset, signed=False, **kwargs)

    def _getvalue(self, native_struct: NativeStruct) -> bool:
        return bool(super()._getvalue(native_struct))

    def _setvalue(self, native_struct: NativeStruct, value: bool):
        return super()._setvalue(native_struct, int(value))


class StringField(SimpleField):
    '''
    A StringField interprets strings of constant or variable size with the
    specified encoding.

    StringField supports variable sized bytearray chunks. To make the field
    dynamically sized, pass None to the length parameter and call
    native_struct.resize() to resize the field to a new length.

    To decode the string, the encoding type is required. By default, this is "utf-8".
    To view the list of possible encodings,
    visit https://docs.python.org/3/library/codecs.html#standard-encodings.

    Args:
        offset (Tuple[NativeField, int]): the offset of this field
        length (int): the length of the string in bytes
        encoding (str): the encoding of the string, from Standard Encodings of the codecs module
    '''
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
    '''
    ArrayField is a general purpose field for parsing arrays of elements.

    You can specify what type of elements the array stores with the
    elem_field_type argument.

    As ArrayField utilizes numpy arrays to handle arrays, the field
    supports parsing multidimensional arrays. The shape argument
    takes a single int (one dimensional array) or a tuple of dimension
    sizes. In case of multi dimensional arrays, the elements are always
    parsed sequentially, but are reshaped into the target shape.

    To make the array dynamically sized, pass None as its shape and
    use native_struct.resize() to resize the field to a new size.

    Attributes:
        shape (tuple): the shape of the array

    Args:
        offset (Tuple[NativeField, int]): the offset of this field
        shape (Tuple[tuple, int]): a tuple or an integer specifying the dimensions of the array
        elem_field_type (type): the type of the field elements inside the array.
                                The type has to be a subclass of NativeField or NativeStruct.

    The rest of keyword arguments are passed to the elem_field_type constructor.
    For example, to create an array field with IntegerField's of size 2, pass the size
    keyword argument:

        class Struct(NativeStruct):
            array = ArrayField(0, None, IntegerField, size=2)
    '''
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
        elif issubclass(elem_field_type, NativeStruct):
            self._elem_field = StructField(0, elem_field_type)
        else:
            raise Exception('elem_field_type has to be a subclass of NativeField or NativeStruct, '
                            f'got {elem_field_type.__name__}')

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
    def _get_array_index(shape: tuple, index: tuple) -> int:
        assert shape, 'shape cannot be an empty list'
        assert len(shape) == len(index), 'shape and index need to be the same length'
        return sum([index[i] * int(np.prod(shape[i + 1:])) for i in range(len(index) - 1)]) + index[-1]

    def _getvalue(self, native_struct: NativeStruct) -> np.array:
        arr = np.empty(self.shape, dtype=object)

        arr_offset = native_struct.calc_field_offset(self)
        is_struct_field = isinstance(self._elem_field, StructField)

        for index in np.ndindex(self.shape):
            self._elem_field.offset = (
                arr_offset + ArrayField._get_array_index(self.shape, index) * self._elem_field.size
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

        arr_offset = native_struct.calc_field_offset(self)
        for index in np.ndindex(self.shape):
            self._elem_field.offset = (
                arr_offset + ArrayField._get_array_index(self.shape, index) * self._elem_field.size
            )

            self._elem_field._setvalue(native_struct, arr[index])


class VariableField(NativeField):
    '''
    A VariableField is a field which initially does not hold any
    type of field. To set the type that's being parsed,
    call the resize method of its parent NativeStruct:

        class Struct(NativeStruct):
            variable_field = VariableField(0)

        native_struct = Struct()
        native_struct.resize('variable_field', IntegerField(0, size=2), resize_bytes=True)
        native_struct.variable_field = 5

    Note that trying to access the VariableField without resizing it first
    will result in an exception.

    Attributes:
        child (NativeField): the child field that is currently stored

    Args:
        offset (Tuple[NativeField, int]): the offset of this field
    '''
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
            raise Exception('VariableField does not contain any field, '
                            'call resize() with the field instance you want to store')

        return self.child._getvalue(native_struct)

    def _setvalue(self, native_struct: NativeStruct, value):
        if not self.child:
            raise Exception('VariableField does not contain any field, '
                            'call resize() with the field instance you want to store')

        val = self.child._setvalue(native_struct, value)
        self.size = self.child.size
        return val


def unpack_bytes(data: Tuple[bytearray, bytes, NativeStruct], field: NativeField):
    '''
    Interprets the bytes inside the data using the supplied field.

    If data is a NativeStruct instance, the value is directly
    retrieved from the bytearray of the NativeStruct.

    Args:
        data (Tuple[bytearray, bytes, NativeStruct]): the data to interpret
        field (NativeField): the field used to interpret the data

    Returns:
        the interpreted value
    '''
    if isinstance(data, NativeStruct):
        return field._getvalue(data)

    native_struct = NativeStruct(data)
    return field._getvalue(native_struct)


def pack_value(value, field: NativeField) -> bytearray:
    '''
    Produces a NativeStruct with the data representing the
    supplied field and value.

    Args:
        value: the value to encode
        field (NativeField): the field used to encode the value

    Returns:
        bytearray: the resulting bytes of the encoding
    '''
    native_struct = NativeStruct(bytearray(field.min_offset + field.size))
    field._setvalue(native_struct, value)
    return native_struct.data
