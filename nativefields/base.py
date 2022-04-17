
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Tuple


class StructBase(type):
    def __new__(cls, name, bases, attrs):
        struct_size = 0
        last_field = None
        max_field_offset = -1

        has_instance_fields = False
        for field in attrs.copy().values():
            if isinstance(field, NativeField) and field.is_instance:
                has_instance_fields = True
                break

        for key, field in attrs.copy().items():
            if not isinstance(field, NativeField):
                continue

            field.property_name = f'{key}_field'
            if has_instance_fields:
                attrs[key] = property(field._get_instance_value, field._set_instance_value)
            else:
                attrs[key] = property(field._getvalue, field._setvalue)

            attrs[field.property_name] = field
            field_offset = field.get_real_offset(False)
            struct_size = max(struct_size, field_offset + field.size)

            if field_offset > max_field_offset:
                last_field = field
                max_field_offset = field_offset

        attrs['has_instance_fields'] = has_instance_fields
        attrs['min_size'] = struct_size
        attrs['last_field'] = last_field
        return super(StructBase, cls).__new__(cls, name, bases, attrs)


class NativeField(ABC):
    offset: Tuple[object, int]
    size: int
    is_instance: bool = False
    property_name: str
    visible: bool = None

    def __init__(self, *args, **kwargs):
        self.visible = kwargs.pop('visible', True)
        if self.visible is not None:
            self.is_instance = True

    @abstractmethod
    def _getvalue(self, native_struct):
        pass

    @abstractmethod
    def _setvalue(self, native_struct, value):
        pass

    def resize(self, native_struct, length):
        pass

    @property
    def real_offset(self):
        return self.get_real_offset()

    def get_real_offset(self, exclude_invisible: bool = True):
        return _translate_offset(self.offset, exclude_invisible)

    def _get_instance_value(self, native_struct):
        return getattr(native_struct, self.property_name)._getvalue(native_struct)

    def _set_instance_value(self, native_struct, value):
        return getattr(native_struct, self.property_name)._setvalue(native_struct, value)


def _translate_offset(offset: Tuple[NativeField, int], exclude_invisible: bool = True):
    if isinstance(offset, int):
        return offset

    if exclude_invisible and offset.visible is False:
        return offset.real_offset

    return offset.real_offset + offset.size


class NativeStruct(metaclass=StructBase):
    def __init__(self, data: bytearray = None, master_offset: Tuple[NativeField, int] = 0, **kwargs):
        if data:
            self.data = data
        else:
            self.data = bytearray(self.min_size)
        self.master_offset = master_offset

        if type(self).has_instance_fields:
            for varname, value in vars(type(self)).items():
                if isinstance(value, NativeField):
                    field_copy = deepcopy(value)
                    if isinstance(field_copy.offset, NativeField):
                        assert hasattr(
                            field_copy.offset, 'property_name'
                        ), f'Offset of field "{varname}" does not have any parent struct class'
                        field_copy.offset = getattr(self, field_copy.offset.property_name)

                    setattr(self, varname, field_copy)

        for key in kwargs:
            setattr(self, key, kwargs[key])

    @property
    def size(self):
        last_field = getattr(self, type(self).last_field.property_name)
        if not last_field:
            return 0

        return last_field.real_offset + last_field.size

    def resize(self, field_name: str, size):
        try:
            field = getattr(self, f'{field_name}_field')
            if not field.is_instance:
                raise Exception('Non instance fields cannot be resized')

            field.resize(self, size)
        except AttributeError:
            raise Exception(f'Field with name "{field_name}" does not exist in class {type(self).__name__}')

    def _resize_data(self, resizing_field: NativeField, old_size: int):
        offset = self._calc_offset(resizing_field)

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

    def _calc_offset(self, native_field):
        return _translate_offset(self.master_offset, True) + native_field.real_offset
