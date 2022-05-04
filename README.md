![Build Status](https://github.com/donadigo/nativefields/workflows/Build/badge.svg)
# NativeFields
A Python library for parsing/manipulating binary data with easily accessible Python properties. The library is still in development. NativeFields supports:
* Variable length fields
* Optional fields
* Nested structures

## Quick example
NativeFields allows to define binary data layout declaratively which then maps to underlying bytes:
```py
from nativefields import *

class Header(NativeStruct):
    magic = StringField(offset=0, length=6)
    length = IntegerField(offset=magic)
    array = ArrayField(offset=length, shape=None, elem_field_type=IntegerField)
    floating = FloatField(offset=array)

header = Header(magic='native', floating=3.14)
header.length = 3
header.array = list(range(1, header.length + 1))
print(header.data)
```

## Output:
```py
bytearray(b'native\x03\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\xc3\xf5H@')`
```

## Example: parse a JPEG header
You can embed other structure declarations inside structures:
```py
from nativefields import *

class RGB(NativeStruct):
    r = IntegerField(offset=0, signed=False, size=1)
    g = IntegerField(offset=r, signed=False, size=1)
    b = IntegerField(offset=g, signed=False, size=1)

class Marker(NativeStruct):
    marker = IntegerField(offset=0, size=2, signed=False)
    length = IntegerField(marker, size=2, signed=False)
    identifier = StringField(length, length=5, encoding='ascii')
    version = IntegerField(identifier, size=2, signed=False)
    density = IntegerField(version, size=1, signed=False)
    x_density = IntegerField(version, size=2, signed=False)
    y_density = IntegerField(version, size=2, signed=False)
    x_thumbnail = IntegerField(version, size=2, signed=False)
    y_thumbnail = IntegerField(version, size=2, signed=False)
    thumb_data = ArrayField(offset=y_thumbnail, shape=None, elem_field_type=RGB)

class JPEGHeader(NativeStruct):
    soi = IntegerField(offset=0, size=2, signed=False)
    marker = StructField(soi, Marker)

with open('image.jpg', 'rb') as f:
    # Parse the JPEG header
    header = JPEGHeader(f.read())

    # Resize the thumbnail data
    header.marker.resize(
        'thumb_data', header.marker.x_thumbnail * header.marker.y_thumbnail,
        resize_bytes=False
    )

    # Display the thumbnail
    display_thumbnail(header.marker.thumb_data)
```