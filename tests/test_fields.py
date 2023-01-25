import struct
import numpy as np
from bytefield import *


def test_pack_value():
    field = IntegerField()
    data = pack_value(0xDEAD, field)
    assert data == struct.pack('i', 0xDEAD)

    field = DoubleField(offset=4)
    data = pack_value(4.5, field)
    assert data == bytearray(4) + struct.pack('d', 4.5)

    field = StringField(len('string'), encoding='ascii')
    data = pack_value('string', field)
    assert data == struct.pack(f"{len('string')}s", 'string'.encode('ascii'))


def test_unpack_value():
    assert unpack_bytes(struct.pack('i', 10), IntegerField()) == 10
    assert unpack_bytes(bytearray(8) + struct.pack('d', 3.2), DoubleField(offset=8)) == 3.2
    assert unpack_bytes(bytearray(8) + struct.pack('d', 3.2), DoubleField()) == 0


def test_struct():
    class InnerStruct(ByteStruct):
        value1 = StringField(length=len('abc'))
        value2 = StringField(length=len('def'))
        value3 = ArrayField(shape=3, elem_field_type=FloatField)

    class TestStruct(ByteStruct):
        value1 = IntegerField()
        value2 = DoubleField()
        value3 = ArrayField(shape=(3, 3), elem_field_type=IntegerField, size=2, signed=False)
        inner = StructField(InnerStruct)
        ba = ByteArrayField(10)

    arr = np.arange(1, 10).reshape((3, 3))

    assert TestStruct.min_size == 4 + 8 + (3 * 3 * 2) + InnerStruct.min_size + 10
    inst = TestStruct(value1=15, value2=90.2, value3=arr, ba=[3] * 10)
    assert inst.value1 == 15
    assert inst.value2 == 90.2
    assert np.array_equal(inst.value3.to_numpy(), arr)
    assert np.array_equal(inst.ba.to_bytearray(), [3] * 10)

    inst.inner.value1 = 'abc'
    inst.inner.value2 = 'def'
    inst.inner.value3 = [3, 2, 1]

    assert inst.inner.value1 == 'abc'
    assert inst.inner.value2 == 'def'
    assert np.array_equal(inst.inner.value3.to_numpy(), [3, 2, 1])


def test_instances():
    class InnerStruct(ByteStruct):
        inner_arr = ArrayField(None, StringField, length=3)

    class ElementStruct(ByteStruct):
        elem = IntegerField()

    class TestStruct(ByteStruct):
        static = IntegerField()
        arr = ArrayField(None, IntegerField)
        inner = StructField(InnerStruct)
        elements = ArrayField(None, ElementStruct)

    inst1 = TestStruct()
    inst2 = TestStruct()
    inst3 = TestStruct()

    inst1.arr = [1, 2, 3]
    inst2.arr = [4, 5, 6, 7]

    assert np.array_equal(inst1.arr.to_numpy(), [1, 2, 3])
    assert np.array_equal(inst2.arr.to_numpy(), [4, 5, 6, 7])

    assert inst3.arr.shape == (0,)

    inst1.inner.inner_arr = ['000', '111', '222']
    inst2.inner.inner_arr = ['222', '333', '444', '555']

    inst1.elements = [ElementStruct(elem=5), ElementStruct(elem=1)]
    inst2.resize(TestStruct.elements_field, 4, resize_bytes=True)

    assert np.array_equal(inst1.inner.inner_arr.to_numpy(), ['000', '111', '222'])
    assert inst1.elements[0].elem == 5
    assert inst1.elements[1].elem == 1
    assert np.array_equal(inst2.inner.inner_arr.to_numpy(), ['222', '333', '444', '555'])
    assert inst2.elements[3].elem == 0


def test_variable():
    class Struct(ByteStruct):
        variable = VariableField(0)

    class Inner(ByteStruct):
        arr = ArrayField(None, StringField, length=2)

    class Inner2(ByteStruct):
        arr = ArrayField(None, StringField, length=4)

    s = Struct()
    s.resize(Struct.variable_field, StructField(Inner, offset=0), resize_bytes=True)
    s.variable.arr = ['aa', 'bb', 'cc', 'dd']
    assert len(s.data) == 8
    assert np.array_equal(s.variable.arr.to_numpy(), ['aa', 'bb', 'cc', 'dd'])

    s.resize(Struct.variable_field, StringField(length=8, offset=0))
    assert s.variable == 'aabbccdd'

    s.resize(Struct.variable_field, StructField(Inner2, offset=0))
    s.variable.resize(Inner2.arr_field, 2)
    s.variable.resize(Inner2.arr_field, 4, resize_bytes=True)

    assert s.variable.arr[0] == 'aabb'
    assert s.variable.arr[1] == 'ccdd'
    assert len(s.data) == 16
