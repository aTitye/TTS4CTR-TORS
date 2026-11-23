"""Reader-only TFRecord and ``tf.train.Example`` parser.

``iter_tfrecord`` yields payloads without checking CRCs. ``parse_example``
extracts the ``feature`` int64 list and ``label`` float from each payload.
"""
import struct
from typing import Iterator, Tuple

import numpy as np


def iter_tfrecord(path: str) -> Iterator[bytes]:
    """Yield raw payload bytes of each record in a TFRecord file (CRCs skipped)."""
    with open(path, "rb") as f:
        while True:
            header = f.read(8)
            if len(header) < 8:
                return
            (length,) = struct.unpack("<Q", header)
            f.read(4)  # masked crc32 of length
            payload = f.read(length)
            f.read(4)  # masked crc32 of data
            yield payload


def _read_varint(buf: bytes, pos: int) -> Tuple[int, int]:
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def _iter_fields(buf: bytes):
    """Iterate protobuf fields: yields (field_number, wire_type, value).

    value is int for varint/fixed, bytes for length-delimited.
    """
    pos = 0
    n = len(buf)
    while pos < n:
        tag, pos = _read_varint(buf, pos)
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            value, pos = _read_varint(buf, pos)
        elif wire == 1:
            value = buf[pos : pos + 8]
            pos += 8
        elif wire == 2:
            length, pos = _read_varint(buf, pos)
            value = buf[pos : pos + length]
            pos += length
        elif wire == 5:
            value = struct.unpack("<I", buf[pos : pos + 4])[0]
            pos += 4
        else:
            raise ValueError(f"unsupported wire type {wire}")
        yield field, wire, value


def _parse_int64_list(buf: bytes, count: int) -> np.ndarray:
    out = np.empty(count, dtype=np.int64)
    pos_i = 0
    p = 0
    n = len(buf)
    while p < n:
        tag, p = _read_varint(buf, p)
        field, wire = tag >> 3, tag & 7
        if field != 1:
            raise ValueError(f"unexpected field {field} in int64 list")
        if wire == 0:  # unpacked single value
            v, p = _read_varint(buf, p)
            out[pos_i] = v
            pos_i += 1
        elif wire == 2:  # packed run of varints
            length, p = _read_varint(buf, p)
            end = p + length
            while p < end:
                v, p = _read_varint(buf, p)
                out[pos_i] = v
                pos_i += 1
        else:
            raise ValueError("unexpected wire type in int64 list")
    if pos_i != count:
        raise ValueError(f"int64 list length {pos_i} != expected {count}")
    return out


def _parse_float_list(buf: bytes) -> float:
    vals = []
    for field, wire, value in _iter_fields(buf):
        if field == 1:
            if wire == 5:
                vals.append(struct.unpack("<f", struct.pack("<I", value))[0])
            elif wire == 2:  # packed
                for i in range(0, len(value), 4):
                    vals.append(struct.unpack("<f", value[i : i + 4])[0])
    return vals[0]


def parse_example(payload: bytes, num_fields: int) -> Tuple[np.ndarray, float]:
    """Return (feature int64[num_fields], label float)."""
    feature = None
    label = None
    for f1, w1, v1 in _iter_fields(payload):
        if f1 == 1 and w1 == 2:  # Example.features
            for f2, w2, v2 in _iter_fields(v1):
                if f2 == 1 and w2 == 2:  # Features.feature map entry
                    key = None
                    feat_msg = None
                    for f3, w3, v3 in _iter_fields(v2):
                        if f3 == 1 and w3 == 2:
                            key = v3.decode("utf-8")
                        elif f3 == 2 and w3 == 2:
                            feat_msg = v3
                    if key == "feature":
                        for f4, w4, v4 in _iter_fields(feat_msg):
                            if f4 == 3 and w4 == 2:  # Feature.int64_list
                                feature = _parse_int64_list(v4, num_fields)
                    elif key == "label":
                        for f4, w4, v4 in _iter_fields(feat_msg):
                            if f4 == 2 and w4 == 2:  # Feature.float_list
                                label = _parse_float_list(v4)
    if feature is None or label is None:
        raise ValueError("example missing feature/label")
    return feature, label
