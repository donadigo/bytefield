
from abc import ABC, abstractmethod
import bytefield.fields
from copy import deepcopy
from typing import Iterable, Tuple
import numpy as np
import bytefield.array_proxy
from bytefield.format import _format_bytearray, _format_numpy


class StructBase(type):
    '''
    The metaclass used for creating ByteStruct class types.

    Iterates over the fields available in a ByteStruct subclass
    and creates a new type with necessary static attributes.
    Each field part of the subclass is used as a base to create
    getters and setters for it. The actual field instances can be
    accessed through the *field_name*_field attribute. These field
    instances may be shared across ByteStruct instances.
    Dynamically sized fields such as ArrayField or StringField
    are copied for each new instance to store their state per
    ByteStruct instance.

    StructBase also adds additional attributes to the class type:
    - min_size (int): the minimum size the ByteStruct subclass requires
      to hold all of its members. Dynamically sized arrays and fields
      are counted as having size of 0.
    - last_field (ByteField): the last field in the ByteStruct subclass,
      None if there are no fields in the subclass
    '''
    def __new__(cls, name, bases, attrs):
        last_field = None

        instance_fields = []
        instance_with_parent_field_names = []

        for key, field in attrs.copy().items():
            if not isinstance(field, ByteField):
                continue

            if key in ['data', 'master_offset', 'size', 'instance_with_parent_field_names']:
                raise KeyError(f'a field cannot have a name of "{key}". '
                               'To resolve, rename the field to something else')

            field.property_name = f'{key}_field'
            attrs[key] = property(field._getvalue, field._setvalue)

            if isinstance(field.offset, int):
                field.computed_offset = field.offset
                instance_fields = []
            elif last_field is None:
                assert field.offset is None or isinstance(field.offset, int), \
                    'first field must have an integer offset or None'
                field.computed_offset = field.offset if field.offset else 0
            else:
                field.computed_offset = last_field.computed_offset
                if not last_field.size_includes_computation:
                    field.computed_offset += last_field.size

            field.instance_fields = instance_fields[:]

            attrs[field.property_name] = field

            if field.instance_with_parent:
                instance_with_parent_field_names.append(key)

            if field.is_instance:
                instance_fields.append(field)

            last_field = field

        if last_field:
            new_size = last_field.computed_offset + last_field.size

            if last_field.is_instance:
                instance_fields = instance_fields[:-1]

            new_size += sum(f.size for f in instance_fields if f.size_includes_computation)
            attrs['min_size'] = new_size

        attrs['last_field'] = last_field
        attrs['instance_with_parent_field_names'] = instance_with_parent_field_names
        return super(StructBase, cls).__new__(cls, name, bases, attrs)


class ByteField(ABC):
    '''
    A ByteField is a base abstract class, which is subclassed
    to create a field type used in ByteStruct's. A ByteField implements
    _getvalue and _setvalue methods which operate on the ByteStruct
    instance. These methods usually use the struct module to pack the data into
    the underlying bytearray.

    The ByteField consists of an offset and size, which define the placement
    and sizing of the field. The offset may be a constant integer byte offset,
    or another instance of ByteField, which means that this field directly follows
    that instance offset.

    Attributes:
        offset (Tuple[ByteField, int]): the offset of this field in bytes.
                                        The offset may be None, constant integer byte offset,
                                        or another instance of ByteField, which means that
                                        this field directly follows that instance offset.
        size (int): the size of this field. If the sizing is dynamic, the size is 0 and
                    the field is instance based.
        is_instance (bool): if the field is instance based, it is copied per
                            ByteStruct instance.
        property_name (str): the property name of this field. This is the field name
                             in the ByteStruct class.
    '''
    offset: Tuple[object, int]
    computed_offset: int
    instanced_fields: list
    size: int
    min_size: int
    is_instance: bool = False
    property_name: str

    def __init__(self, _offset, _size, **kwargs):
        self.property_name = None
        self.instance_fields = []
        self.offset = _offset
        self.size = _size
        self.computed_offset = self.offset if isinstance(self.offset, int) else 0
        self.size_includes_computation = False
        self.instance_with_parent = kwargs.pop('instance_with_parent', False)

    def get_size(self, byte_struct):
        '''
        Gets the size of the ByteField, provided the ByteStruct. You may override this method
        if the size of your field depends on the parent ByteStruct.

        Args:
            byte_struct (ByteStruct): the ByteStruct that the size is calculated for

        Returns:
            the size of this field in bytes
        '''
        if self.is_instance:
            data = byte_struct._get_instance_data(self)
            return data['size']

        return self.size

    @abstractmethod
    def _getvalue(self, byte_struct):
        '''
        Gets the value from the provided ByteStruct. To obtain the real byte offset
        into this ByteField, call byte_struct.calc_offset(self).

        Args:
            byte_struct (ByteStruct): the ByteStruct that the value is retrieved from

        Returns:
            the value that was retrieved
        '''
        pass

    @abstractmethod
    def _setvalue(self, byte_struct, value):
        '''
        Sets the value in the provided ByteStruct. To obtain the real byte offset
        into this ByteField, call byte_struct.calc_offset(self).

        Args:
            byte_struct (ByteStruct): the ByteStruct that the value is set in
            value: the value that is being used to set the field
        '''
        pass

    def resize(self, length, byte_struct):
        '''
        The resize method is called when the user requests that this field should
        be resized. Note that you should never call this method directly on the field,
        instead use the resize() API in ByteStruct. This is because the ByteStruct
        API ensures that the field is instanced specifically for the struct instance,
        and that you are not modifying the size of fields in all existing struct instances.

            class Struct(ByteStruct):
                a = ArrayField(None, IntegerField)

            s = Struct([10, 0, 0, 0, 20, 0, 0, 0])
            s.resize(S.a_field, 2)
            print(s.a)  # [10, 20]

        Resizing involves updating the *size* attribute. Note that resizing is only available for
        instance based fields. That is, is_instance has to be set to True.
        '''
        pass

    def _resize_with_data(self, byte_struct, value):
        '''
        Resizes the field including resizing the underlying
        bytearray.

        byte_struct (ByteStruct): the target byte struct that the field is in
        value: the value to resize to
        '''
        old_size = self.get_size(byte_struct)
        self.resize(value, byte_struct)
        if old_size != self.get_size(byte_struct):
            byte_struct._resize_data(self, old_size)


class InstanceData:
    size: int

    def __init__(self) -> None:
        self.size = 0


class ByteStruct(metaclass=StructBase):
    '''
    ByteStruct is the main class used for creating your own
    struct definitions. The definition consists of a list of fields
    that make up the struct. The fields act as a convenient resource
    of defining the structure and also define how you use instances
    of the newly created subclass of ByteStruct.

    ByteStruct's may be constructed out of existing data e.g read from a file, or
    without any arguments. If no data has been provided, ByteStruct will
    automatically resize itself to fit the minimum size required for storing
    all the fields. ByteStruct's can also be offset by a master offset,
    which defines the additional offset of all the fields inside the struct.
    This is mainly used to internally implement nested structs.

    A ByteStruct can contain instanced fields, which contain data specifc to a ByteStruct
    instance. When an instanced field is accessed or written to, ByteStruct
    will make sure to copy the field, so that information can be saved per
    ByteStruct.

    You can also provide initial values for the ByteStruct instance inside the constructor.
    These will be reflected in the underlying bytearray immediately after construction.

    Args:
        data (Iterable): the existing data you want to interpret or modify, None by default
        master_offset (int): the master offset used for the struct, 0 by default
        **kwargs: field=value pairs which describe initial values for defined fields

    Attributes:
        data (bytearray): the underlying bytearray containing the binary data
        master_offset (int): the master offset used for the struct, typically 0
    '''
    def __init__(self, data: Iterable = None, master_offset: int = 0, **kwargs):
        if data is not None:
            if isinstance(data, bytearray):
                self.data = data
            else:
                self.data = bytearray(data)
        else:
            self.data = bytearray(self.min_size)

        self.master_offset = master_offset
        self.instance_data = {}

        if kwargs:
            for key in kwargs:
                setattr(self, key, kwargs[key])

        if self.__class__.instance_with_parent_field_names:
            for field_name in self.__class__.instance_with_parent_field_names:
                getattr(self, field_name)

    @property
    def size(self):
        '''
        Returns the offset in bytes after the last field in the struct.

        This size is not the size of the actual underlying bytearray data
        you may have provided. The size is calculated as calc_field_offset(last_field) + last_field.size.
        To retrieve the data size, access len(struct.data).

        The property does not include hidden fields when calculating the size.

        Note that with e.g:

            class Struct(ByteStruct):
                member = IntegerField(offset=4)

        the size returned by this property is 8 and not 4, this is
        because the last field of this struct ends at offset 8.

        Returns:
            int: the offset after the last field in the struct
        '''
        if not self.__class__.last_field:
            return 0

        return self.calc_field_offset(self.__class__.last_field) + self.__class__.last_field.get_size(self)

    def _get_instance_data(self, field: ByteField) -> InstanceData:
        '''
        Gets the instance data for a field.

        Args:
            field (ByteField): the field to get the instance data for

        Returns:
            InstanceData: the instance data for the field
        '''
        if field not in self.instance_data:
            self.instance_data[field] = {'size': 0}

        return self.instance_data[field]

    def resize(self, field: ByteField, size, resize_bytes: bool = False):
        '''
        Resizes a field inside the struct to a new size.

        By default, this method does not resize the underlying bytearray
        data, changing only the reading size.
        To allocate new space for the struct in the underlying bytearray, specify
        resize_bytes=True.

        If resize_bytes=True, the method adds or removes the bytes
        in the bytearray needed to align the field to its the new size.
        If the field is growing, the field is right padded with zero bytes.
        Similarly, if the field is shrinking, the bytes are removed
        from the end of its data.

            class Struct(ByteStruct):
                arr = ArrayField(None, IntegerField)

            s = Struct()
            # Resize the array field to contain 4 elements, adding the
            # required amount of bytes to the bytearray
            s.resize(S.arr_field, 4, resize_bytes=True)

        Non instance fields such as IntegerField or FloatField cannot be resized:
        calling this method to resize a non instance field will result in an exception.

        Dynamic fields such as ArrayField, StringField and ByteArrayField
        manage their size automatically when changing their contents,
        so it is not required to call this method when setting a new value.

        The size parameter can be an arbitrary object that's type is specified
        by the target field. For example, an ArrayField may be resized with a tuple
        containing the array shape and a VariableField, another ByteField instance.

        Args:
            field (ByteField): the field that is being resized
            size: the new size to resize the field to, type dependant on the field being resized
        '''
        if not field.is_instance:
            raise Exception('non instance fields cannot be resized')

        if resize_bytes:
            old_size = field.get_size(self)
        field.resize(size, self)
        if resize_bytes:
            self._resize_data(field, old_size)

    def check_overflow(self):
        '''
        Checks if the last field definition does not exceed the
        size of the underlying data bytearray. Used to ensure
        integrity when e.g. printing the struct.

        The method throws an exception if self.size > len(self.data).
        '''
        if self.size > len(self.data):
            raise OverflowError('overflow detected: fields after resize take more size than the struct data. '
                                'Make sure to use resize_bytes=True in resize() to resize the underlying bytes')

    def _ensure_is_instanced(self, field: ByteField) -> ByteField:
        '''
        Ensures that a field is instanced within the byte struct.

        If the field is not an instance field, the provided field
        is returned as is. If it is, deepcopy is called on
        the field to perform a deep copy and the field
        is assigned to the struct instance.

        Args:
            field (ByteField): the target field

        Returns:
            ByteField: either the same field, or a deepcopy of the field if it was instanced
        '''
        if not field.is_instance:
            return field

        struct_field = getattr(self, field.property_name)
        if struct_field != field:
            return struct_field

        copy = deepcopy(struct_field)
        setattr(self, copy.property_name, copy)
        return copy

    def _resize_data(self, resizing_field: ByteField, old_size: int):
        '''
        Resizes the underlying data bytearray in-place given the field
        that is being resized and its old size.

        The method adds or removes the bytes in the bytearray needed
        to align the field to its the new size. If the field has
        is growing, the field is right padded with zero bytes.
        Similarly, if the field is shrinking, the bytes are removed
        from the end of its data.

        Args:
            resizing_field (ByteField): the field that is being resized
            old_size (int): the old size of the resizing field, in bytes
        '''
        offset = self.calc_offset(resizing_field)

        new_size = resizing_field.get_size(self)

        # Make sure to modify in-place
        if old_size > new_size:
            rest = self.data[offset:offset + new_size] + self.data[offset + old_size:]
            del self.data[offset:]
            self.data.extend(rest)
        elif old_size < new_size:
            added_bytes = new_size - old_size
            rest = bytearray([0] * added_bytes) + self.data[offset + old_size:]
            del self.data[offset + old_size:]
            self.data.extend(rest)

    def calc_offset(self, byte_field: ByteField) -> int:
        '''
        Calculates the real offset for the provided field.

        This method accounts for the master offset that may be
        present in the ByteStruct that is containing this field,
        returning how many bytes to skip in the bytearray
        to arrive at the beginning of the field.

        Args:
            byte_field (ByteField): the byte field to calculate the offset for

        Returns:
            int: the calculated offset in bytes
        '''
        return self.master_offset + self.calc_field_offset(byte_field)

    def calc_field_offset(self, byte_field: ByteField) -> int:
        '''
        Calculates the real offset for the provided field.

        Same as calc_offset but does not include adding the master offset.

        Args:
            byte_field (ByteField): the byte field to calculate the offset for

        Returns:
            int: the calculated offset in bytes, without the master offset
        '''
        if byte_field.instance_fields:
            return byte_field.computed_offset + sum(
                [instance_field.get_size(self) for instance_field in byte_field.instance_fields]
            )

        return byte_field.computed_offset

    def _print(self, indent_level: int) -> str:
        '''
        Constructs a human readable string that lists all
        the fields of the struct and their current values.

        Args:
            indent_level (int): the indent level, used for printing
                                nested structs

        Returns:
            str: the human readable string
        '''
        tab = '\t' * indent_level
        r = ''
        for varname, value in vars(self.__class__).items():
            if not isinstance(value, property):
                continue

            try:
                field = getattr(self, f'{varname}_field')
            except AttributeError:
                continue

            try:
                field_val = getattr(self, varname)
            except Exception as e:
                r += f'{tab}{varname}: [ reading error ({e.__class__.__name__}) ]\n'
                continue

            if isinstance(field_val, ByteStruct):
                r += f'{tab}{varname} ({field_val.__class__.__name__}):\n{field_val._print(indent_level + 1)}'
            elif isinstance(field_val, (bytearray, bytes, bytefield.array_proxy.ByteArrayFieldProxy)):
                if isinstance(field_val, bytefield.array_proxy.ByteArrayFieldProxy):
                    field_val = field_val.to_bytearray()

                bytes_repr = _format_bytearray(field_val)
                r += f'{tab}{varname} ({field_val.__class__.__name__}): {bytes_repr}'
            elif isinstance(field_val, (np.ndarray, list, bytefield.array_proxy.ArrayFieldProxy)):
                arr_repr = ''

                if isinstance(field_val, bytefield.array_proxy.ArrayFieldProxy):
                    field_val = field_val.to_numpy()

                arr_repr = _format_numpy(field_val)
                if len(field_val) > 0:
                    arr_repr = arr_repr.replace('\n', f'\n{tab}\t')

                r += f'{tab}{varname} ({field_val.__class__.__name__}): {arr_repr}'
            else:
                r += f'{tab}{varname} ({field_val.__class__.__name__}): {field_val}'

            if field.property_name != self.__class__.last_field.property_name:
                r += '\n'

        return r

    def __repr__(self) -> str:
        self.check_overflow()
        return f'[{self.__class__.__name__} object at {hex(id(self))}]\n{self._print(0)}'
