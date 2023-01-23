![Build Status](https://github.com/donadigo/bytefields/workflows/Build/badge.svg)
# ByteFields
A Python library for parsing/manipulating binary data with easily accessible Python properties inspired by Django. The library is still in development. ByteFields supports:
* Variable length fields
* Nested structures
* Parsing only accessed fields

## Quick example
ByteFields allows to define binary data layout declaratively which then maps to underlying bytes:
```py
from bytefields import *

class Header(ByteStruct):
    magic = StringField(length=5)
    length = IntegerField()
    array = ArrayField(shape=None, elem_field_type=IntegerField)
    floating = FloatField()

header = Header(magic='bytes', floating=3.14)
header.length = 3
header.array = list(range(1, header.length + 1))
print(header.data)
```

### Output:
```py
bytearray(b'bytes\x03\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\xc3\xf5H@')`
```

## Example: parse a JPEG header
You can embed other structure declarations inside structures:
```py
from bytefields import *

class RGB(ByteStruct):
    r = IntegerField(signed=False, size=1)
    g = IntegerField(signed=False, size=1)
    b = IntegerField(signed=False, size=1)

class Marker(ByteStruct):
    marker = IntegerField(size=2, signed=False)
    length = IntegerField(size=2, signed=False)
    identifier = StringField(length=5, encoding='ascii')
    version = IntegerField(size=2, signed=False)
    density = IntegerField(size=1, signed=False)
    x_density = IntegerField(size=2, signed=False)
    y_density = IntegerField(size=2, signed=False)
    x_thumbnail = IntegerField(size=2, signed=False)
    y_thumbnail = IntegerField(size=2, signed=False)
    thumb_data = ArrayField(shape=None, elem_field_type=RGB)

class JPEGHeader(ByteStruct):
    soi = IntegerField(size=2, signed=False)
    marker = StructField(Marker)

with open('image.jpg', 'rb') as f:
    # Parse the JPEG header
    header = JPEGHeader(f.read())

    # Resize the thumbnail data
    header.marker.resize(
        Marker.thumb_data_field, header.marker.x_thumbnail * header.marker.y_thumbnail
    )

    # Display the thumbnail
    display_thumbnail(header.marker.thumb_data)
```

## Writing custom struct logic
You can create high-level structures which define their own behavior depending on the data contained within the struct:
```py
from bytefields import *

class DynamicFloatArray(ByteStruct):
    length = IntegerField(signed=False)
    array_data = ArrayField(None, FloatField)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # When instantiated, resize the array according to its length
        self.resize(DynamicFloatArray.array_data_field, self.length)

data = bytearray(b'\x03\x00\x00\x00\x00\x00\x80?\x00\x00\x00@\x00\x00@@')
print(DynamicFloatArray(data))
```

### Output:
```py
[DynamicFloatArray object at 0x1c88e709e50]
length (int): 3
array_data (ndarray): [1.0 2.0 3.0]
```

## Variable fields
Bytefields supports fields with unknown type/size:
```py
from bytefields import *

TYPE_INTEGER = 0
TYPE_FLOAT = 1
TYPE_STRING = 2

class DynamicString(ByteStruct):
    length = IntegerField(signed=False)
    str = StringField(None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resize(DynamicString.str_field, self.length)

class Content(ByteStruct):
    content_type = IntegerField(signed=False, size=2)
    content_data = VariableField()  # a variable field that will be resized when parsing the struct

    def __init__(self, data: bytearray = None, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        resize_bytes = not bool(data)
        if self.content_type == TYPE_INTEGER:
            self.resize(Content.content_data_field, IntegerField(), resize_bytes=resize_bytes)
        elif self.content_type == TYPE_FLOAT:
            self.resize(Content.content_data_field, FloatField(), resize_bytes=resize_bytes)
        elif self.content_type == TYPE_STRING:
            self.resize(Content.content_data_field, StructField(DynamicString), resize_bytes=resize_bytes)

write = Content()
write.content_type = TYPE_STRING
write.resize(Content.content_data_field, StructField(DynamicString), resize_bytes=True)
write.content_data.str = 'content'
write.content_data.length = len(write.content_data.str)

read = Content(write.data)
print(f'{write.data} is parsed to:\n{read}')
```

### Output
```
bytearray(b'\x02\x00\x07\x00\x00\x00content') is parsed to:
[Content object at 0x1c1846888b0]
content_type (int): 2
content_data (DynamicString):
        length (int): 7
        str (str): content
```