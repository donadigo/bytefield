import struct
import numpy as np
from nativefields import *


def test_pack_value():
    field = IntegerField(0)
    obj = pack_value(0xDEAD, field)
    assert bytearray(obj.data) == struct.pack('i', 0xDEAD)

    field = DoubleField(4)
    obj = pack_value(4.5, field)
    assert bytearray(obj.data) == bytearray(4) + struct.pack('d', 4.5)

    field = StringField(0, len('string'), encoding='ascii')
    obj = pack_value('string', field)
    assert bytearray(obj.data) == struct.pack(f"{len('string')}s", 'string'.encode('ascii'))


def test_unpack_value():
    assert unpack_bytes(struct.pack('i', 10), IntegerField(0)) == 10
    assert unpack_bytes(bytearray(8) + struct.pack('d', 3.2), DoubleField(8)) == 3.2
    assert unpack_bytes(bytearray(8) + struct.pack('d', 3.2), DoubleField(0)) == 0


def test_struct():
    class InnerStruct(NativeStruct):
        value1 = StringField(offset=0, length=len('abc'))
        value2 = StringField(offset=value1, length=len('def'))
        value3 = ArrayField(offset=value2, shape=3, elem_field_type=FloatField)

    class TestStruct(NativeStruct):
        value1 = IntegerField(offset=0)
        value2 = DoubleField(offset=value1)
        value3 = ArrayField(offset=value2, shape=(3, 3), elem_field_type=IntegerField, size=2, signed=False)
        inner = StructField(value3, InnerStruct)
        ba = ByteArrayField(inner, 10)

    arr = np.arange(1, 10).reshape((3, 3))

    assert TestStruct.min_size == 4 + 8 + (3 * 3 * 2) + InnerStruct.min_size + 10
    inst = TestStruct(value1=15, value2=90.2, value3=arr, ba=[3] * 10)
    assert inst.value1 == 15
    assert inst.value2 == 90.2
    assert np.array_equal(inst.value3, arr)
    assert np.array_equal(inst.ba, [3] * 10)

    inst.inner.value1 = 'abc'
    inst.inner.value2 = 'def'
    inst.inner.value3 = [3, 2, 1]

    assert inst.inner.value1 == 'abc'
    assert inst.inner.value2 == 'def'
    assert np.array_equal(inst.inner.value3, [3, 2, 1])
