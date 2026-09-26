"""ctypes bridge to the compiled Mojo kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a 64-bit
address, so the argtypes below must stay `c_int64` for addresses; `c_int`
truncates them and segfaults.
"""

import ctypes
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-capnproto.so"

_PTR = ctypes.c_int64


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))

    def sig(name, restype, argtypes):
        fn = getattr(lib, name)
        fn.restype = restype
        fn.argtypes = argtypes

    sig("cpn_struct_layout", None,
        [_PTR, _PTR, _PTR, _PTR, _PTR, _PTR])
    sig("cpn_pack", ctypes.c_int64, [_PTR, _PTR, _PTR])
    sig("cpn_unpack", ctypes.c_int64, [_PTR, _PTR, _PTR, _PTR])
    sig("cpn_pack_struct_ptr", None, [_PTR, _PTR, _PTR, _PTR])
    sig("cpn_unpack_struct_ptr", None, [_PTR, _PTR])
    sig("cpn_pack_list_ptr", None, [_PTR, _PTR, _PTR, _PTR])
    sig("cpn_unpack_list_ptr", None, [_PTR, _PTR])
    sig("cpn_pack_far_ptr", None, [_PTR, _PTR, _PTR, _PTR])
    sig("cpn_pack_double_far_ptr", None, [_PTR] * 6)
    sig("cpn_pack_capability_ptr", None, [_PTR, _PTR])
    sig("cpn_word_kind", ctypes.c_int64, [_PTR])
    sig("cpn_frame_message", ctypes.c_int64, [_PTR, _PTR, _PTR, _PTR])
    sig("cpn_parse_frame", ctypes.c_int64, [_PTR, _PTR, _PTR, _PTR, _PTR])
    sig("cpn_read_struct_fields", None, [_PTR, _PTR, _PTR, _PTR, _PTR])
    sig("cpn_resolve_ptrs", None, [_PTR, _PTR, _PTR, _PTR])
    return lib


lib = _load()


# Pointer kind codes, held in the low two bits of a pointer word.
STRUCT = 0
LIST = 1
FAR = 2
OTHER = 3

# List element size codes.
VOID = 0
BIT = 1
BYTE = 2
TWO_BYTES = 3
FOUR_BYTES = 4
EIGHT_BYTES = 5
POINTER = 6
COMPOSITE = 7

ELEM_WIDTH = {VOID: 0, BIT: 0, BYTE: 1, TWO_BYTES: 2, FOUR_BYTES: 4,
              EIGHT_BYTES: 8, POINTER: 8, COMPOSITE: 0}

# Bit widths of the built-in primitive types, used by `struct_layout`.
PRIMITIVE_BITS = {
    "void": 0, "bool": 1, "int8": 8, "uint8": 8, "int16": 16, "uint16": 16,
    "int32": 32, "uint32": 32, "int64": 64, "uint64": 64, "float32": 32,
    "float64": 64, "float16": 16, "enum": 16, "pointer": None,
}


def _i64(v) -> int:
    """Reinterpret an integer as a signed 64-bit value for the C ABI."""
    v = int(v) & 0xFFFFFFFFFFFFFFFF
    return v - 0x10000000000000000 if v >= 0x8000000000000000 else v


class _Scratch:
    """A ctypes byte buffer plus its address, for short-lived output."""

    __slots__ = ("buf", "addr", "size")

    def __init__(self, nbytes: int):
        self.size = max(nbytes, 1)
        self.buf = (ctypes.c_uint8 * self.size)()
        self.addr = ctypes.addressof(self.buf)

    def bytes(self, n=None) -> bytes:
        # ctypes array slicing copies element by element and is orders of
        # magnitude slower than a bulk read, which matters for a 2 MB frame.
        # string_at is bounded, never NUL-terminated.
        return ctypes.string_at(self.addr, self.size if n is None else n)


# ---------------------------------------------------------------------------
# Struct layout
# ---------------------------------------------------------------------------


def struct_layout(fields) -> dict:
    """Lay out a struct from a sequence of field descriptors.

    Each descriptor is either a primitive name from `PRIMITIVE_BITS` or the
    string "pointer" (also spelled "text", "data", "list" or "struct", which are
    all pointer slots in the wire format).

    Returns a dict with `offsets` (bit offset per ordinal, -1 for a pointer),
    `sizes` (bit width per ordinal, -1 for a pointer) and `data_words` (the
    data section size in whole 64-bit words).
    """
    mask = []
    sizes = []
    aligns = []
    for name in fields:
        if name in ("pointer", "text", "data", "list", "struct", "anyPointer"):
            mask.append(1)
            sizes.append(0)
            aligns.append(0)
            continue
        if name not in PRIMITIVE_BITS:
            raise ValueError(f"unknown field type {name!r}")
        width = PRIMITIVE_BITS[name]
        mask.append(0)
        sizes.append(width)
        aligns.append(1 if width <= 1 else width)
    n = len(mask)
    mask_a = np.ascontiguousarray(np.asarray(mask, dtype=np.uint8))
    sizes_a = np.ascontiguousarray(np.asarray(sizes, dtype=np.uint32))
    aligns_a = np.ascontiguousarray(np.asarray(aligns, dtype=np.uint32))
    offs_a = np.zeros(n, dtype=np.uint32)
    words = (ctypes.c_uint32 * 1)()
    lib.cpn_struct_layout(n, mask_a.ctypes.data, sizes_a.ctypes.data,
                          aligns_a.ctypes.data, offs_a.ctypes.data,
                          ctypes.addressof(words))
    offs = [int(x) for x in offs_a]
    return {
        "offsets": [-1 if mask[i] else offs[i] for i in range(n)],
        "sizes": [-1 if mask[i] else sizes[i] for i in range(n)],
        "data_words": int(words[0]),
    }


def read_struct_fields(data, layout) -> list:
    """Read a struct's data section into one raw 64-bit word per ordinal.

    Pointer slots come back as None. Values are zero-extended; the caller
    reinterprets them with the schema, exactly as the real format requires.
    """
    n = len(layout["offsets"])
    # A pointer slot has no size; the kernel only looks at the offset for it.
    sizes = np.ascontiguousarray(
        np.asarray([0 if o < 0 else layout["sizes"][i]
                    for i, o in enumerate(layout["offsets"])], dtype=np.uint32))
    uoffs = np.ascontiguousarray(
        np.asarray([0xFFFFFFFF if o < 0 else o for o in layout["offsets"]],
                   dtype=np.uint32))
    raw = bytes(data)
    if len(raw) < 8 * layout["data_words"]:
        raise ValueError(f"data section of {len(raw)} bytes is shorter than the "
                         f"{layout['data_words']} words the layout needs")
    src = _Scratch(max(len(raw), 1))
    if raw:
        ctypes.memmove(src.addr, raw, len(raw))
    vals = np.zeros(n, dtype=np.uint64)
    lib.cpn_read_struct_fields(src.addr, n, uoffs.ctypes.data,
                               sizes.ctypes.data, vals.ctypes.data)
    return [None if layout["offsets"][i] < 0 else int(vals[i])
            for i in range(n)]


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------


def pack(data) -> bytes:
    """Pack a byte string that is a whole number of 64-bit words.

    Each word becomes a tag byte plus its non-zero bytes; tag 0x00 introduces a
    run of zero words and tag 0xff introduces a run of verbatim non-zero words.
    """
    raw = bytes(data)
    if len(raw) % 8:
        raise ValueError(f"packed input must be a multiple of 8 bytes, got {len(raw)}")
    nwords = len(raw) // 8
    src = _Scratch(max(len(raw), 1))
    if raw:
        ctypes.memmove(src.addr, raw, len(raw))
    out = _Scratch(10 * nwords + 16)
    n = lib.cpn_pack(src.addr, nwords, out.addr)
    return out.bytes(n)


def unpack(data, maxwords: int) -> bytes:
    """Unpack a packed byte stream, stopping at `maxwords` words."""
    raw = bytes(data)
    src = _Scratch(max(len(raw), 1))
    if raw:
        ctypes.memmove(src.addr, raw, len(raw))
    dst = (ctypes.c_uint64 * max(maxwords, 1))()
    n = lib.cpn_unpack(src.addr, len(raw), ctypes.addressof(dst), maxwords)
    return ctypes.string_at(ctypes.addressof(dst), 8 * n)


# ---------------------------------------------------------------------------
# Pointer words
# ---------------------------------------------------------------------------


def pack_struct_pointer(offset_words: int, data_words: int, pointer_count: int) -> int:
    out = _Scratch(8)
    lib.cpn_pack_struct_ptr(_i64(offset_words), _i64(data_words),
                            _i64(pointer_count), out.addr)
    return int.from_bytes(out.bytes(), "little")


def unpack_struct_pointer(word: int) -> tuple:
    out = (ctypes.c_uint32 * 3)()
    lib.cpn_unpack_struct_ptr(_i64(word), ctypes.addressof(out))
    off = out[0] - 0x100000000 if out[0] >= 0x80000000 else out[0]
    return int(off), int(out[1]), int(out[2])


def pack_list_pointer(offset_words: int, elem_size: int, count: int) -> int:
    out = _Scratch(8)
    lib.cpn_pack_list_ptr(_i64(offset_words), _i64(elem_size), _i64(count),
                          out.addr)
    return int.from_bytes(out.bytes(), "little")


def unpack_list_pointer(word: int) -> tuple:
    out = (ctypes.c_uint32 * 3)()
    lib.cpn_unpack_list_ptr(_i64(word), ctypes.addressof(out))
    off = out[0] - 0x100000000 if out[0] >= 0x80000000 else out[0]
    return int(off), int(out[1]), int(out[2])


def pack_far_pointer(offset_words: int, two_word_pad: int, segment_id: int) -> bytes:
    out = _Scratch(16)
    lib.cpn_pack_far_ptr(_i64(offset_words), _i64(two_word_pad),
                         _i64(segment_id), out.addr)
    return out.bytes(16)


def pack_double_far_pointer(pad_segment: int, pad_word: int, tag_word: int,
                            content_segment: int, content_word: int) -> bytes:
    out = _Scratch(32)
    lib.cpn_pack_double_far_ptr(_i64(pad_segment), _i64(pad_word),
                                _i64(tag_word), _i64(content_segment),
                                _i64(content_word), out.addr)
    return out.bytes(32)


def pack_capability_pointer(capability_index: int) -> int:
    out = _Scratch(8)
    lib.cpn_pack_capability_ptr(_i64(capability_index), out.addr)
    return int.from_bytes(out.bytes(), "little")


def word_kind(word: int) -> int:
    return int(lib.cpn_word_kind(_i64(word)))


# ---------------------------------------------------------------------------
# Message framing
# ---------------------------------------------------------------------------


def frame_message(segments) -> bytes:
    """Frame a list of word-aligned segment byte strings into one message.

    A Cap'n Proto segment is always a whole number of 8-byte words, so a
    segment whose length is not a multiple of 8 is rejected rather than
    silently rounded.
    """
    segs = [bytes(s) for s in segments] or [b""]
    for s in segs:
        if len(s) % 8:
            raise ValueError(f"segment of {len(s)} bytes is not word-aligned")
    counts = np.ascontiguousarray([len(s) // 8 for s in segs], dtype=np.uint32)
    payload = b"".join(segs)
    src = _Scratch(max(len(payload), 1))
    if payload:
        ctypes.memmove(src.addr, payload, len(payload))
    out = _Scratch(len(payload) + 8 * len(segs) + 8)
    total = lib.cpn_frame_message(len(segs), counts.ctypes.data, src.addr,
                                  out.addr)
    return out.bytes(total)


def parse_frame(message) -> list:
    """Return [(byte offset, byte size)] per segment of a framed message."""
    data = bytes(message)
    maxseg = 64
    counts = np.zeros(maxseg, dtype=np.uint32)
    offsets = np.zeros(maxseg, dtype=np.uint32)
    src = _Scratch(max(len(data), 1))
    if data:
        ctypes.memmove(src.addr, data, len(data))
    n = lib.cpn_parse_frame(src.addr, len(data), counts.ctypes.data,
                            offsets.ctypes.data, maxseg)
    if n < 0:
        raise ValueError(f"malformed capnp frame (code {n})")
    return [(int(offsets[i]), int(counts[i]) * 8) for i in range(n)]


def resolve_pointers(segment, base_word: int, count: int) -> list:
    """Resolve a run of pointer words inside a segment to target byte offsets.

    Returns -1 for a null pointer and -2 for a far or capability pointer.
    """
    raw = bytes(segment)
    src = _Scratch(max(len(raw), 1))
    if raw:
        ctypes.memmove(src.addr, raw, len(raw))
    tgt = np.zeros(max(count, 1), dtype=np.uint64)
    lib.cpn_resolve_ptrs(src.addr, base_word, count, tgt.ctypes.data)
    out = []
    for i in range(count):
        v = int(tgt[i])
        out.append(-1 if v == 0xFFFFFFFFFFFFFFFF
                   else -2 if v == 0xFFFFFFFFFFFFFFFE else v)
    return out
