
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Tuple
import numpy as np


class StructBase(type):
    '''
    The metaclass used for creating NativeStruct class types.

    Iterates over the fields available in a NativeStruct subclass
    and creates a new type with necessary static attributes.
    Each field part of the subclass is used as a base to create
    getters and setters for it. The actual field instances can be
    accessed through the *field_name*_field attribute. These field
    instances may be shared across NativeStruct instances.
    If a NativeStruct subclass contains dynamic sized fields, all fields
    are copied for each new instance.

    StructBase also adds additional attributes to the class type:
    - has_instance_fields (bool): whether the type contains instance fields,
      which extract data based on a specific NativeStruct instance
    - min_size (int): the minimum size the NativeStruct subclass requires
      to hold all of its members. Dynamically sized arrays are counted
      as having size of 0.
    - last_field (NativeField): the last field in the NativeStruct subclass,
      None if there are no fields in the subclass
    '''
    def __new__(cls, name, bases, attrs):
        struct_size = 0
        last_field = None
        max_field_offset = -1

        for key, field in attrs.copy().items():
            if not isinstance(field, NativeField):
                continue

            for forbidden in ['data', 'master_offset']:
                if key == forbidden:
                    raise Exception(f'A field cannot have a name of "{forbidden}". '
                                    'To resolve, rename the field to something else')

            field.property_name = f'{key}_field'
            if field.is_instance:
                attrs[key] = property(field._get_instance_value, field._set_instance_value)
            else:
                attrs[key] = property(field._getvalue, field._setvalue)

            attrs[field.property_name] = field
            field_offset = field.get_min_offset(False)
            struct_size = max(struct_size, field_offset + field.size)

            if field_offset > max_field_offset or (field_offset == max_field_offset and field.offset == last_field):
                last_field = field
                max_field_offset = field_offset

        attrs['min_size'] = struct_size
        attrs['last_field'] = last_field
        return super(StructBase, cls).__new__(cls, name, bases, attrs)


class NativeField(ABC):
    '''
    A NativeField is a base abstract class, which is subclassed
    to create a field type used in NativeStruct's. A NativeField implements
    _getvalue and _setvalue methods which operate on the NativeStruct
    instance. These methods usually use the struct module to pack the data into
    the underlying bytearray.

    The NativeField consists of an offset and size, which define the placement
    and sizing of the field. The offset maybe a constant integer byte offset,
    or another instance of NativeField, which means that this field directly follows
    that instance offset.

    Additionally, a field may be invisible. If a field is invisible, it is
    not taken account into offset calculation, therefore any later fields are
    accessed as if the invisible field did not exist.

    Attributes:
        offset (Tuple[NativeField, int]): the offset of this field in bytes.
                                          The offset maybe a constant integer byte offset,
                                          or another instance of NativeField, which means that
                                          this field directly follows that instance offset.
        size (int): the size of this field. If the sizing is dynamic, the size is 0 and
                    the field is instance based.
        is_instance (bool): if the field is instance based, it is copied per
                            NativeStruct instance.
        property_name (str): the property name of this field. This is the field name
                             in the NativeStruct class.
        visible (bool): if the field is visible. By default, this value is None,
                        which means that the value cannot change visibility.
                        Setting the value explicitly to True or False at instantiation
                        will result in the field being instance based and enable
                        toggling its visibility.
    '''
    offset: Tuple[object, int]
    size: int
    is_instance: bool = False
    property_name: str
    visible: bool = None

    def __init__(self, *args, **kwargs):
        self.property_name = None
        self.visible = kwargs.pop('visible', None)
        if self.visible is not None:
            self.is_instance = True

    @abstractmethod
    def _getvalue(self, native_struct):
        '''
        Gets the value from the provided NativeStruct. To obtain the real byte offset
        into this NativeField, call native_struct.calc_offset(self).

        Args:
            native_struct (NativeStruct): the NativeStruct that the value is retrieved from

        Returns:
            the value that was retrieved
        '''
        pass

    @abstractmethod
    def _setvalue(self, native_struct, value):
        '''
        Sets the value in the provided NativeStruct. To obtain the real byte offset
        into this NativeField, call native_struct.calc_offset(self).

        Args:
            native_struct (NativeStruct): the NativeStruct that the value is set in
            value: the value that is being used to set the field
        '''
        pass

    def resize(self, length):
        '''
        The resize method is called when the user requests that this field should
        be resized.

        Resizing involves updating the *size* attribute and calling native_struct._resize_data,
        to actually resize the underlying bytearray. Note that resizing is only available for
        instance based fields. That is, is_instance has to be set to True.
        '''
        pass

    @property
    def min_offset(self):
        '''
        Gets the real offset of this field in bytes, excluding any invisible fields.
        For more details, see NativeField.get_min_offset.

        Returns:
            int: the minimum offset of this field in the parent NativeStruct, in bytes
        '''
        return self.get_min_offset()

    def get_min_offset(self, exclude_invisible: bool = True):
        '''
        Gets the minimum offset of this field in bytes.

        Translates the offset field and recurses until
        a constant offset is encountered. This method
        allows you to count invisible fields as if they were
        visible in the struct. The offset returned may not be
        the real offset with instance fields present in the structure.
        To obtain the real offset, call native_struct.calc_offset(field).

        Args:
            exclude_invisible (bool): whether to exclude invisible fields, True by default

        Returns:
            int: the minimum offset of this field in the parent NativeStruct, in bytes
        '''
        return _translate_offset(self.offset, exclude_invisible)

    def _get_instance_value(self, native_struct):
        '''
        Same as _getvalue, but calls the method on a field inside NativeStruct instance.
        Only used in class type generation.

        Args:
            native_struct (NativeStruct): the NativeStruct that the value is retrieved from

        Returns:
            the value that was retrieved
        '''
        native_struct._ensure_is_instanced(self)
        return getattr(native_struct, self.property_name)._getvalue(native_struct)

    def _set_instance_value(self, native_struct, value):
        '''
        Same as _setvalue, but calls the method on a field inside NativeStruct instance.
        Only used in class type generation.

        Args:
            native_struct (NativeStruct): the NativeStruct that the value is set in
            value: the value that is being used to set the field
        '''
        native_struct._ensure_is_instanced(self)
        return getattr(native_struct, self.property_name)._setvalue(native_struct, value)

    def _resize_with_data(self, native_struct, value):
        old_size = self.size
        self.resize(value)
        if old_size != self.size:
            native_struct._resize_data(self, old_size)


def _translate_offset(offset: Tuple[NativeField, int], exclude_invisible: bool = True):
    '''
    Used by NativeField.get_min_offset to calculate the offset.

    Args:
        offset (Tuple[NativeField, int]): if NativeField, calculates the offset of the field,
                                          if int, returns this argument
        exclude_invisible (bool): whether to exclude invisible fields, True by default
    '''
    if isinstance(offset, int):
        return offset

    if exclude_invisible and offset.visible is False:
        return offset.min_offset

    return offset.min_offset + offset.size


class NativeStruct(metaclass=StructBase):
    '''
    NativeStruct is the main class used for creating your own
    struct definitions. The definition consists of a list of fields
    that make up the struct. The fields act as a convenient resource
    of defining the structure and also define how you use instances
    of the newly created subclass of NativeStruct.

    NativeStruct's may be constructed out of existing data e.g read from a file, or
    without any arguments. If no data has been provided, NativeStruct will
    automatically resize itself to fit the minimum size required for storing
    all the fields. NativeStruct's can also be offset by a master offset,
    which defines the additional offset of all the fields inside the struct.
    This is mainly used to internally implement nested structs.

    If the struct contains any instance fields, all field objects inside the struct are copied
    to the new instance.

    You can also provide initial values for the NativeStruct instance inside the constructor.
    These will be reflected in the underlying bytearray immediately after construction.

    Args:
        data (Tuple[bytearray, bytes]): the existing data you want to interpret or modify, None by default
        master_offset (int): the master offset used for the struct, 0 by default
        **kwargs: field=value pairs which describe initial values for defined fields

    Attributes:
        data (bytearray): the underlying bytearray containing the binary data
        master_offset (int): the master offset used for the struct, typically 0
    '''
    def __init__(self, data: Tuple[bytearray, bytes] = None, master_offset: int = 0, **kwargs):
        if data:
            if isinstance(data, bytes):
                self.data = bytearray(data)
            else:
                self.data = data
        else:
            self.data = bytearray(self.min_size)

        self.master_offset = master_offset

        for key in kwargs:
            setattr(self, key, kwargs[key])

    def setattr_proxy(self, name, value):
        raise Exception('aaaa')

    @property
    def size(self):
        '''
        Returns the offset in bytes after the last field in the struct.

        This size is not the size of the actual underlying bytearray data
        you may have provided. The size is calculated as calc_offset(last_field) + last_field.size.
        To retrieve the data size, access len(struct.data).

        The property does not include hidden fields when calculating the size.

        Note that with e.g:

            class Struct(NativeStruct):
                member = IntegerField(offset=4)

        the size returned by this property is 8 and not 4, this is
        because the last field of this struct ends at offset 8.

        Returns:
            int: the offset after the last field in the struct
        '''
        last_field = getattr(self, self.__class__.last_field.property_name)
        if not last_field:
            return 0

        return self.calc_offset(last_field) + last_field.size

    def resize(self, field_name: str, size, resize_bytes: bool = False):
        '''
        Resizes a field inside the struct to a new size.

        By default, this method does not resize the underlying bytearray
        data, changing only the reading size.
        To allocate new space for the struct in the underlying bytearray, specify
        resize_bytes=True.

        If resize_bytes=True, the method adds or removes the bytes
        in the bytearray needed to align the field to its the new size.
        If the field has is growing, the field is right padded with zero bytes.
        Similarly, if the field is shrinking, the bytes are removed
        from the end of its data.

        Non instance fields such as IntegerField or FloatField cannot be resized:
        calling this method to resize a non instance field will result in an exception.

        Dynamic fields such as ArrayField, StringField and ByteArrayField
        manage their size automatically when changing their contents,
        so it is not required to call this method when setting a new value.

        The size parameter can be an arbitrary object that's type is specified
        by the target field. For example, an ArrayField may be resized with a tuple
        containing the array shape and a VariableField, another NativeField instance.

        Args:
            field_name (str): the field name of the field inside the NativeStruct
            size: the new size to resize the field to, type dependant on the field being resized
            resize_bytes (bool): whether to add or remove bytes to the bytearray based on the field size
                                 False by default
        '''
        try:
            field = getattr(self, f'{field_name}_field')
            if not field.is_instance:
                raise Exception('Non instance fields cannot be resized')

            field = self._ensure_is_instanced(field)
            old_size = field.size
            field.resize(size)
            if resize_bytes:
                self._resize_data(field, old_size)
        except AttributeError:
            raise Exception(f'Field with name "{field_name}" does not exist in class {self.__class__.__name__}')

    def check_overflow(self):
        '''
        Checks if the last field definition does not exceed the
        size of the underlying data bytearray. Used to ensure
        integrity when e.g. printing the struct.

        The method throws an exception if self.size > len(self.data).
        '''
        if self.size > len(self.data):
            raise Exception('Overflow detected: fields after resize take more size than the struct data')

    def _ensure_is_instanced(self, field: NativeField):
        struct_field = getattr(self, field.property_name)
        if field.is_instance and struct_field == field:
            copy = deepcopy(struct_field)
            setattr(self, copy.property_name, copy)
            return copy

        return field

    def _resize_data(self, resizing_field: NativeField, old_size: int):
        '''
        Resizes the underlying data bytearray in-place given the field
        that is being resized and its old size.

        The method adds or removes the bytes in the bytearray needed
        to align the field to its the new size. If the field has
        is growing, the field is right padded with zero bytes.
        Similarly, if the field is shrinking, the bytes are removed
        from the end of its data.

        Args:
            resizing_field (NativeField): the field that is being resized
            old_size (int): the old size of the resizing field, in bytes
        '''
        offset = self.calc_offset(resizing_field)

        # Make sure to modify in-place
        if old_size > resizing_field.size:
            rest = self.data[offset:offset + resizing_field.size] + self.data[offset + old_size:]
            del self.data[offset:]
            self.data.extend(rest)
        elif old_size < resizing_field.size:
            added_bytes = resizing_field.size - old_size
            rest = bytearray([0] * added_bytes) + self.data[offset + old_size:]
            del self.data[offset + old_size:]
            self.data.extend(rest)

    def calc_offset(self, native_field: NativeField):
        '''
        Calculates the real offset for the provided field.

        This method accounts for the master offset that may be
        present in the NativeStruct that is containing this field,
        returning how many bytes to skip in the bytearray
        to arrive at the beginning of the field.

        Args:
            native_field (NativeField): the native field to calculate the offset for

        Returns:
            int: the calculated offset in bytes
        '''
        return self.master_offset + self.calc_field_offset(native_field)

    def calc_field_offset(self, field: NativeField) -> int:
        '''
        Calculates the real offset for the provided field.

        Same as calc_offset but does not include adding the master offset.

        Args:
            native_field (NativeField): the native field to calculate the offset for

        Returns:
            int: the calculated offset in bytes, without the master offset
        '''
        if isinstance(field.offset, int):
            return field.offset

        instance_offset = getattr(self, field.offset.property_name)
        return self.calc_field_offset(instance_offset) + instance_offset.size

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

            if field.visible == False:  # NOQA
                r += f'{tab}{varname}: (hidden)\n'
            else:
                field_val = getattr(self, varname)
                if isinstance(field_val, NativeStruct):
                    r += f'{tab}{varname} ({field_val.__class__.__name__}):\n{field_val._print(indent_level + 1)}'
                elif isinstance(field_val, bytearray) or isinstance(field_val, bytes):
                    if field_val:
                        bytes_repr = f"[ {bytes(field_val[:16]).hex(' ').upper()}"
                        if len(field_val) > 16:
                            bytes_repr += f'  ({len(field_val) - 16} more bytes...)'
                        bytes_repr += ' ]\n'
                    else:
                        bytes_repr = '[ empty ]\n'
                    r += f'{tab}{varname} ({field_val.__class__.__name__}): {bytes_repr}'
                elif isinstance(field_val, np.ndarray) or isinstance(field_val, list):
                    arr_repr = ''

                    if len(field_val) > 0:
                        arr_repr += f'{field_val[:16]}'
                        if len(field_val) > 16:
                            arr_repr += f'   ({len(field_val) - 16} more items...)'

                        if '\n' in arr_repr:
                            arr_repr = '\n' + arr_repr

                        arr_repr = arr_repr.replace('\n', f'\n{tab}\t')
                        arr_repr += '\n'
                    else:
                        arr_repr = '[ empty ]\n'
                    r += f'{tab}{varname} ({field_val.__class__.__name__}): {arr_repr}'
                else:
                    r += f'{tab}{varname} ({field_val.__class__.__name__}): {field_val}\n'

        return r

    def __repr__(self) -> str:
        self.check_overflow()
        return f'[{self.__class__.__name__} object at {hex(id(self))}]\n' + self._print(0)
