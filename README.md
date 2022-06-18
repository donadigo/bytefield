![Build Status](https://github.com/donadigo/bytefields/workflows/Build/badge.svg)
# ByteFields
A Python library for parsing/manipulating binary data with easily accessible Python properties. The library is still in development. ByteFields supports:
* Variable length fields
* Optional fields
* Nested structures

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

## Output:
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
        'thumb_data', header.marker.x_thumbnail * header.marker.y_thumbnail
    )

    # Display the thumbnail
    display_thumbnail(header.marker.thumb_data)
```