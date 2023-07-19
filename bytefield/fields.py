from bytefield.array_proxy import ArrayFieldProxy, ByteArrayFieldProxy, _get_array_index
from bytefield.base import ByteStruct, ByteField
from enum import Enum
import struct
import math
from typing import Iterable, Tuple
import numpy as np


class StructField(ByteField):
    '''
    A StructField provides a way to embed other byte structs inside a byte struct.

    The data is stored as if the StructField was replaced with the fields from
    the struct that this field holds: that is, the offset of the StructField equals
    the first field of the inner struct + additional offset of the first field, if it
    has any.

    Internally, StructFields pass the bytearray data reference to the child struct and
    set their master offset to the current offset of the StructField. This allows for
    accessing inner fields with native syntax:

        master_struct.inner_struct.inner_int = 5

    However, this also means that the value of StructField (a ByteStruct instance),
    is not valid after e.g. resizing a field inside the master struct:

        class Inner(ByteStruct):
            inner_int = IntegerField()

        class Master(ByteStruct):
            bytes = ByteArrayField(length=None)
            inner_struct = StructField(Inner)

        master_struct = Master()
        inner = master_struct.inner_struct
        master_struct.resize(Master.bytes_field, 8)

        inner.inner_int = 5  # Invalid, the master struct changed its layout and
                             # the inner reference is now invalid

        master_struct.inner_struct.inner_int = 5  # Valid, accessing inner structs this way always
                                                  # makes sure they are valid

        Attributes:
            struct_type (type): the ByteStruct type that this field holds
            size (int): the size of this field in bytes
            is_instance (bool): always True, the field is always only an instance field
            inner: the inner struct that is being stored
            offset (Tuple[ByteField, int]): the offset of this field
        Args:
            offset (Tuple[ByteField, int]): the offset of this field
            struct_type (type): the ByteStruct type that this field holds
    '''
    def __init__(
        self,
        struct_type: type,
        offset: Tuple[ByteField, int] = None,
        instance_with_parent: bool = True, **kwargs
    ):
        assert issubclass(struct_type, ByteStruct), 'struct_type must be an inheritant of type ByteStruct'
        super().__init__(offset, struct_type.min_size, instance_with_parent=instance_with_parent, **kwargs)
        self.struct_type = struct_type
        self.offset = offset
        self.is_instance = True
        self.size_includes_computation = True

    def reset(self, byte_struct):
        data = byte_struct._get_instance_data(self)
        data['inner'] = None

    def get_size(self, byte_struct: ByteStruct):
        data = byte_struct._get_instance_data(self)
        if not data.get('inner'):
            # The user may resize fields inside his own ByteStruct subclass, so we have to
            # make sure to instantiate inner to get the proper size.
            return self.size

        return data['inner'].size

    def _getvalue(self, byte_struct: ByteStruct):
        data = byte_struct._get_instance_data(self)
        # We have to update both the data and master_offset of the inner struct:
        # the inner data can come from user code and master_offset is dependant
        # on the sizing of dynamic fields
        if not data.get('inner'):
            data['inner'] = self.struct_type(byte_struct.data, byte_struct.calc_offset(self))
        else:
            data['inner'].data = byte_struct.data
            data['inner'].master_offset = byte_struct.calc_offset(self)

        return data['inner']

    def _setvalue(self, byte_struct: ByteStruct, value):
        data = byte_struct._get_instance_data(self)
        old_inner = data.get('inner')
        if old_inner:
            old_size = old_inner.size
        else:
            old_size = 0

        data['inner'] = value
        new_size = data['inner'].size
        if new_size != old_size:
            byte_struct._resize_data(self, old_size)

        offset = byte_struct.calc_offset(self)
        byte_struct.data[offset:offset + new_size] = value.data[:]

        data['inner'].data = byte_struct.data
        data['inner'].master_offset = byte_struct.calc_offset(self)


class SimpleField(ByteField):
    '''
    A SimpleField is a base field to simple type fields such as IntegerField, FloatField,
    DoubleField, StringField and BooleanField. SimpleField interprets the data
    according to the struct module format provided by the user. Visit
    https://docs.python.org/3/library/struct.html to view this format specification.

    Attributes:
        format (str): the format this field uses to interpret the data
        size (int): the size of this field in bytes
        offset (Tuple[ByteField, int]): the offset of this field, None by default
    Args:
        offset (Tuple[ByteField, int]): the offset of this field
        struct_format (str): the format this field should use to interpret the data
    '''
    def __init__(self, struct_format: str, offset: Tuple[ByteField, int] = None, **kwargs):
        self.format = struct_format
        self.offset = offset
        super().__init__(offset, struct.calcsize(struct_format), **kwargs)

    def _getvalue(self, byte_struct: ByteStruct):
        offset = byte_struct.calc_offset(self)
        if offset + self.size > len(byte_struct.data):
            raise IndexError('failed to get value: field is out of bounds')

        return struct.unpack(self.format, byte_struct.data[offset:offset + self.size])[0]

    def _setvalue(self, byte_struct: ByteStruct, value):
        offset = byte_struct.calc_offset(self)
        if offset + self.size > len(byte_struct.data):
            raise IndexError('failed to set value: field is out of bounds')

        byte_struct.data[offset:offset + self.size] = struct.pack(self.format, value)


class ByteArrayField(ByteField):
    '''
    ByteArrayField allows for slicing the underlying bytearray data,
    that can be interpreted later.

    Accessing a ByteArrayField does not directly return a bytearray.
    Instead, a ByteArrayFieldProxy object is returned that allows for
    modifying bytes inside the array individually without rewriting
    the entire bytearray data. You can then call to_bytearray() on the proxy
    object to retrieve the full bytearray:

        class Struct(ByteStruct):
            byte_array = ByteArrayField(None)

        s = Struct()
        s.byte_array = [1, 2, 3, 4, 5]
        s.byte_array[2] = 2

        print(s) # [ 01 02 02 04 05 ]
        print(type(s.byte_array)) # s.byte_array is a ByteArrayFieldProxy object
        print(s.byte_array.to_bytearray()) # call to_bytearray() to retrieve the full bytearray

    ByteArrayField supports variable sized bytearray chunks. To make the field
    dynamically sized, pass None to the length parameter and call
    byte_struct.resize() to resize the field to a new length.

    Args:
        offset (Tuple[ByteField, int]): the offset of this field
        length (int): the length of the sliced array in bytes
    '''
    def __init__(self, length: int, offset: Tuple[ByteField, int] = None, **kwargs):
        self.is_instance = length is None

        super().__init__(offset, length or 0, **kwargs)

    def resize(self, length: int, byte_struct):
        data = byte_struct._get_instance_data(self)
        data['size'] = length

    def _getvalue(self, byte_struct: ByteStruct):
        return ByteArrayFieldProxy(byte_struct, self)

    def _setvalue(self, byte_struct: ByteStruct, value):
        data = byte_struct._get_instance_data(self)
        new_length = len(value)
        if new_length != data['size']:
            self._resize_with_data(byte_struct, new_length)
        else:
            assert new_length == self.size, f'bytearray size {new_length} not matching, should be {self.size}'

        offset = byte_struct.calc_offset(self)
        if offset + data['size'] > len(byte_struct.data):
            raise IndexError(
                f'failed to set value: field at offset {offset} and size {self.size} '
                f'is out of bounds for struct with size {len(byte_struct.data)}'
            )

        byte_struct.data[offset:offset + self.size] = value


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
    byte struct. IntegerField supports specifying the endianness of the integer
    and if it is signed or not.

    Args:
        offset (Tuple[ByteField, int]): the offset of this field
        signed (bool): whether the integer is signed or unsigned, True by default
        size (int): the size of the integer in bytes, either 1, 2, 4 or 8. 4 by default
        endianness (Endianness): the endianness of the integer, Endianness.NATIVE by default
    '''
    def __init__(
        self,
        signed: bool = True,
        size: int = 4,
        endianness: Endianness = Endianness.NATIVE,
        offset: Tuple[ByteField, int] = None,
        **kwargs
    ):
        prefix = endianness.to_format()
        if size == 4:
            super().__init__(f'{prefix}i' if signed else f'{prefix}I', offset=offset, **kwargs)
        elif size == 2:
            super().__init__(f'{prefix}h' if signed else f'{prefix}H', offset=offset, **kwargs)
        elif size == 1:
            super().__init__(f'{prefix}b' if signed else f'{prefix}B', offset=offset, **kwargs)
        elif size == 8:
            super().__init__(f'{prefix}q' if signed else f'{prefix}Q', offset=offset, **kwargs)
        else:
            raise ValueError('size has to be either 8, 4, 2 or 1')


class DoubleField(SimpleField):
    '''
    DoubleField parses 8 byte floating point numbers.
    Supports specifying the endianness of the integer.

    Args:
        offset (Tuple[ByteField, int]): the offset of this field
        endianness (Endianness): the endianness of the double, Endianness.NATIVE by default
    '''
    def __init__(self, endianness: Endianness = Endianness.NATIVE, offset: Tuple[ByteField, int] = None, **kwargs):
        prefix = endianness.to_format()
        super().__init__(f'{prefix}d', offset=offset, **kwargs)


class FloatField(SimpleField):
    '''
    FloatField parses 4 byte floating point numbers.
    Supports specifying the endianness of the integer.

    Args:
        offset (Tuple[ByteField, int]): the offset of this field
        endianness (Endianness): the endianness of the double, Endianness.NATIVE by default
    '''
    def __init__(self, endianness: Endianness = Endianness.NATIVE, offset: Tuple[ByteField, int] = None, **kwargs):
        prefix = endianness.to_format()
        super().__init__(f'{prefix}f', offset=offset, **kwargs)


class BooleanField(IntegerField):
    '''
    A BooleanField is an IntegerField, which converts between
    booleans and integers. The conversion is carried out with
    using the bool() and int() casts. That is, a non-zero value
    translates to True and False otherwise.

    By default, BooleanField has a size of 4. You can however
    specify a different size by supplying a size= keyword argument.

    Args:
        offset (Tuple[ByteField, int]): the offset of this field

    Keyword Args:
        size (int): the size of the BooleanField in bytes
    '''
    def __init__(self, offset: Tuple[ByteField, int] = None, **kwargs):
        super().__init__(signed=False, offset=offset, **kwargs)

    def _getvalue(self, byte_struct: ByteStruct) -> bool:
        return bool(super()._getvalue(byte_struct))

    def _setvalue(self, byte_struct: ByteStruct, value: bool):
        return super()._setvalue(byte_struct, int(value))


class StringField(ByteField):
    '''
    A StringField interprets strings of constant or variable size with the
    specified encoding.

    StringField supports variable sized bytearray chunks. To make the field
    dynamically sized, pass None to the length parameter and call
    byte_struct.resize() to resize the field to a new length.

    To decode the string, the encoding type is required. By default, this is "utf-8".
    To view the list of possible encodings,
    visit https://docs.python.org/3/library/codecs.html#standard-encodings.

    If a constant length was passed, the string returned from the field
    will always have this exact length. Setting the field to a string
    different than the constant length, the string will either be
    cut off with a longer string or the field will be set partially,
    with a shorter string.

    Args:
        offset (Tuple[ByteField, int]): the offset of this field
        length (int): the length of the string in bytes
        encoding (str): the encoding of the string, from Standard Encodings of the codecs module
    '''
    def __init__(self, length: int, encoding='utf-8', offset: Tuple[ByteField, int] = None, **kwargs):
        self.encoding = encoding
        self.is_instance = length is None
        super().__init__(offset, length or 0, **kwargs)

        if self.is_instance:
            self.format = '0s'
        else:
            self.format = f'{self.size}s'

    def resize(self, length: int, byte_struct):
        data = byte_struct._get_instance_data(self)
        data['size'] = length

    def _getvalue(self, byte_struct: ByteStruct) -> str:
        size = self.get_size(byte_struct)
        offset = byte_struct.calc_offset(self)
        if offset + size > len(byte_struct.data):
            raise IndexError('failed to get value: field is out of bounds')

        return struct.unpack(f'{size}s', byte_struct.data[offset:offset + size])[0].decode(self.encoding)

    def _setvalue(self, byte_struct: ByteStruct, value: str):
        new_length = len(value)

        offset = byte_struct.calc_offset(self)
        if not self.is_instance:
            assert new_length == self.size, \
                f'failed to set value: string is length is not equal to {self.size} characters'

            byte_struct.data[offset:offset + self.size] = struct.pack(self.format, value.encode(self.encoding))
            return

        data = byte_struct._get_instance_data(self)
        if offset + data['size'] > len(byte_struct.data):
            raise IndexError('failed to set value: field is out of bounds')

        if new_length != data['size']:
            self._resize_with_data(byte_struct, new_length)

        byte_struct.data[offset:offset + data['size']] = struct.pack(f'{data["size"]}s', value.encode(self.encoding))


class ArrayField(ByteField):
    '''
    ArrayField is a general purpose field for parsing arrays of elements.

    You can specify what type of elements the array stores with the
    elem_field_type argument.

    As ArrayField utilizes numpy arrays to handle arrays, the field
    supports parsing multidimensional arrays. The shape argument
    takes a single int (one dimensional array) or a tuple of dimension
    sizes. In case of multi dimensional arrays, the elements are always
    parsed sequentially, but are reshaped into the target shape.

    Accessing an ArrayField does not directly return a numpy array.
    Instead, an ArrayFieldProxy object is returned that allows for
    modifying elements inside the array individually without rewriting
    the entire array data. You can then call to_numpy() on the proxy
    object to retrieve the full array if you need to.

    ArrayFields do not support storing instanced fields (such as dynamically
    sized strings). The contained element type has to have a constant size.
    Storing StructFields is supported, but the struct type cannot store
    dynamically sized fields:

        class Inner(ByteStruct):
            i = IntegerField()
            j = IntegerField()
            # s = StringField(None); unsupported, this field makes the
            # struct dynamically sized and cannot be stored inside the array


        class Struct(ByteStruct):
            arr = ArrayField(None, Inner)

        s = Struct()
        # Resize the array using the resize() method
        s.resize(Struct.arr_field, 1, resize_bytes=True)
        s.arr[0].i = 4
        s.arr[0].j = 10

        # Or set the full array
        s.arr = [Inner(i=4, j=10)]

        print(type(s.arr)) # s.arr is an ArrayFieldProxy object
        print(s.arr.to_numpy()) # call to_numpy() to retrieve the full array

    To make the array dynamically sized, pass None as its shape and
    use byte_struct.resize() to resize the field to a new size.

    You may pass a ByteStruct subclass type to the elem_field_type parameter
    to create an array of structs. ArrayField will then automatically create
    a StructField holding the struct type.

    Attributes:
        shape (tuple): the shape of the array

    Args:
        offset (Tuple[ByteField, int]): the offset of this field
        shape (Tuple[tuple, int]): a tuple or an integer specifying the dimensions of the array
        elem_field_type (type): the type of the field elements inside the array.
                                The type has to be a subclass of ByteField or ByteStruct.

    The rest of keyword arguments are passed to the elem_field_type constructor.
    For example, to create an array field with IntegerField's of size 2, pass the size
    keyword argument:

        class Struct(ByteStruct):
            array = ArrayField(0, None, IntegerField, size=2)
    '''
    def __init__(
        self,
        shape: Tuple[tuple, int],
        elem_field_type: type,
        offset: Tuple[ByteField, int] = None,
        **kwargs
    ):
        if issubclass(elem_field_type, ByteField):
            self._elem_field = elem_field_type(offset=0, **kwargs)
        elif issubclass(elem_field_type, ByteStruct):
            self._elem_field = StructField(elem_field_type, offset=0)
        else:
            raise Exception('elem_field_type has to be a subclass of ByteField or ByteStruct, '
                            f'got {elem_field_type.__name__}')

        # TODO: support instanced fields
        if self._elem_field.is_instance and not isinstance(self._elem_field, StructField):
            raise NotImplementedError('instanced fields (with dynamic size) are currently not supported by ArrayField')

        if shape:
            self.shape = (shape,) if isinstance(shape, int) else shape[:]
            self.is_instance = False
        else:
            self.shape = (0,)
            self.is_instance = True

        super().__init__(offset, self._elem_field.size * int(math.prod(self.shape)), **kwargs)

    def get_size(self, byte_struct):
        shape = self.shape
        if self.is_instance:
            data = byte_struct._get_instance_data(self)
            if 'shape' in data:
                shape = data['shape']

        return self._elem_field.get_size(byte_struct) * int(math.prod(shape))

    def resize(self, shape: Tuple[tuple, int], byte_struct):
        data = byte_struct._get_instance_data(self)
        data['shape'] = (shape,) if isinstance(shape, int) else shape[:]
        data['size'] = self._elem_field.get_size(byte_struct) * int(math.prod(data['shape']))

    def _getvalue(self, byte_struct: ByteStruct) -> np.array:
        return ArrayFieldProxy(byte_struct, self)

    def _setvalue(self, byte_struct: ByteStruct, value: Iterable):
        if isinstance(value, np.ndarray):
            assert value.shape == self.shape, f'array shape {value.shape} not matching, should be {self.shape}'

        data = byte_struct._get_instance_data(self)
        if 'shape' not in data:
            data['shape'] = self.shape

        value_shape = np.array(value).shape
        if value_shape != data['shape']:
            self._resize_with_data(byte_struct, value_shape)

        arr = np.empty(data['shape'], dtype=object)
        arr[:] = value

        arr_offset = byte_struct.calc_field_offset(self)
        for index in np.ndindex(data['shape']):
            self._elem_field.computed_offset = (
                arr_offset + _get_array_index(data['shape'], index) * self._elem_field.size
            )

            self._elem_field._setvalue(byte_struct, arr[index])


class VariableField(ByteField):
    '''
    A VariableField is a field which initially does not hold any
    type of field. To set the type that's being parsed,
    call the resize method of its parent ByteStruct:

        class Struct(ByteStruct):
            variable = VariableField()

        byte_struct = Struct()
        byte_struct.resize(Struct.variable_field, IntegerField(size=2), resize_bytes=True)
        byte_struct.variable = 5

    Note that trying to set the VariableField without resizing it first
    will result in an exception. If the field was not resized, accessing
    the field will return None.

    Attributes:
        child (ByteField): the child field that is currently stored

    Args:
        offset (Tuple[ByteField, int]): the offset of this field
    '''
    def __init__(self, offset: Tuple[ByteField, int] = None, **kwargs):
        self.is_instance = True
        super().__init__(offset, 0, **kwargs)

    def get_size(self, byte_struct):
        data = byte_struct._get_instance_data(self)
        if 'child' not in data:
            data['child'] = None

        return data['child'].get_size(byte_struct) if data['child'] else 0

    def resize(self, child: ByteField, byte_struct):
        data = byte_struct._get_instance_data(self)
        data['child'] = child
        data['child'].computed_offset = self.computed_offset
        data['size'] = child.get_size(byte_struct)
        if child.instance_with_parent:
            child._getvalue(byte_struct)

    def _getvalue(self, byte_struct: ByteStruct):
        data = byte_struct._get_instance_data(self)
        if not data.get('child'):
            return None

        return data['child']._getvalue(byte_struct)

    def _setvalue(self, byte_struct: ByteStruct, value):
        data = byte_struct._get_instance_data(self)
        if not data.get('child'):
            raise Exception('VariableField does not contain any field, '
                            'call resize() with the field instance you want to store')

        data['child']._setvalue(byte_struct, value)


def unpack_bytes(data: Tuple[bytearray, bytes, ByteStruct], field: ByteField):
    '''
    Interprets the bytes inside the data using the supplied field.

    If data is a ByteStruct instance, the value is directly
    retrieved from the bytearray of the ByteStruct.

    Args:
        data (Tuple[bytearray, bytes, ByteStruct]): the data to interpret
        field (ByteField): the field used to interpret the data

    Returns:
        the interpreted value
    '''
    if field.offset is None:
        field.offset = 0

    if isinstance(data, ByteStruct):
        return field._getvalue(data)

    byte_struct = ByteStruct(data)
    return field._getvalue(byte_struct)


def pack_value(value, field: ByteField) -> bytearray:
    '''
    Produces a ByteStruct with the data representing the
    supplied field and value.

    Args:
        value: the value to encode
        field (ByteField): the field used to encode the value

    Returns:
        bytearray: the resulting bytes of the encoding
    '''
    if field.offset is None:
        field.offset = 0

    byte_struct = ByteStruct(bytearray(field.computed_offset + field.size))
    field._setvalue(byte_struct, value)
    return byte_struct.data
