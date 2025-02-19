# byte-oriented StringIO was moved to io.BytesIO in py3k
from numba import jit, types, int64

try:
    from io import BytesIO
except ImportError:
    from StringIO import StringIO as BytesIO

import sys
import numpy as np

if sys.version > "3":

    def _byte(b):
        return bytes((b,))
else:

    def _byte(b):
        return chr(b)


def encode(number):
    """Pack `number` into varint bytes"""
    buf = b""
    while True:
        towrite = number & 0x7F
        number >>= 7
        if number:
            buf += _byte(towrite | 0x80)
        else:
            buf += _byte(towrite)
            break
    return buf


def decode_stream(stream, return_bytes=False):
    """Read a varint from `stream`"""
    shift = 0
    result = 0
    bytes_read = 0
    while True:
        i = _read_one(stream)
        result |= (i & 0x7F) << shift
        bytes_read += 1
        shift += 7
        if not (i & 0x80):
            break

    if return_bytes:
        return result, bytes_read

    return result


def decode_bytes(buf, return_bytes=False):
    """Read a varint from from `buf` bytes"""
    if isinstance(buf, (bytes, bytearray)):
        buf = BytesIO(buf)
    elif hasattr(buf, "read"):
        pass
    else:
        raise ValueError("Expected bytes or file-like object")

    return decode_stream(buf, return_bytes=return_bytes)


@jit(
    types.UniTuple(int64, 2)(types.Array(types.uint8, 1, "A", readonly=True), int64),
    nopython=True,
)
def decode_bytes_jit(data: np.ndarray, offset: int):
    """Read a varint from from `buf` bytes"""
    shift = 0
    result = 0
    current_offset = offset

    while True:
        byte = data[current_offset]
        result |= (byte & 0x7F) << shift
        current_offset += 1
        shift += 7
        if not (byte & 0x80):
            break

    return result, current_offset


def _read_one(stream):
    """Read a byte from the file (as an integer)

    raises EOFError if the stream ends while reading bytes.
    """
    c = stream.read(1)
    if c == b"":
        raise EOFError("Unexpected EOF while reading bytes")
    return ord(c)
