"""Conformance of the kernels to the published Cap'n Proto encoding spec.

These need no reference implementation: the format is defined byte for byte at
https://capnproto.org/encoding.html, and the spec's own worked examples are
reproduced here. The pycapnp parity tests live in `test_capnp_parity.py`.
"""

import struct

import pytest

import mojo_capnproto as mcp


# ---------------------------------------------------------------------------
# Pointer words, straight out of the spec's own examples
# ---------------------------------------------------------------------------


def test_struct_pointer_matches_spec_example():
    # encoding.html: "a struct pointer (offset = 2, data size = 3, pointer
    # count = 2)" is the word 08 00 00 00 03 00 02 00.
    word = mcp.pack_struct_pointer(2, 3, 2)
    assert struct.pack("<Q", word) == bytes.fromhex("0800000003000200")
    assert mcp.unpack_struct_pointer(word) == (2, 3, 2)
    assert mcp.word_kind(word) == mcp.STRUCT


def test_list_pointer_matches_spec_example():
    # encoding.html: "a text pointer (offset = 6, length = 53)" is the word
    # 19 00 00 00 aa 01 00 00. Text is a List(UInt8), so element size 2.
    word = mcp.pack_list_pointer(6, mcp.BYTE, 53)
    assert struct.pack("<Q", word) == bytes.fromhex("19000000aa010000")
    assert mcp.unpack_list_pointer(word) == (6, mcp.BYTE, 53)
    assert mcp.word_kind(word) == mcp.LIST


def test_pointer_offset_is_signed_30_bit():
    """A back-pointer is a negative offset; sign extension must survive."""
    for offset in (-1, -5, -1000, 0, 1, 5, 1000):
        word = mcp.pack_struct_pointer(offset, 2, 1)
        assert mcp.unpack_struct_pointer(word)[0] == offset
        word = mcp.pack_list_pointer(offset, mcp.FOUR_BYTES, 7)
        assert mcp.unpack_list_pointer(word)[0] == offset


def test_pointer_kind_is_in_the_low_two_bits():
    """Reading the kind from the wrong end of the word is the classic slip."""
    assert mcp.word_kind(mcp.pack_struct_pointer(3, 1, 1)) == mcp.STRUCT
    assert mcp.word_kind(mcp.pack_list_pointer(3, mcp.EIGHT_BYTES, 1)) == mcp.LIST
    assert mcp.word_kind(mcp.pack_capability_pointer(1)) == mcp.OTHER
    far = struct.unpack("<Q", mcp.pack_far_pointer(9, 0, 3)[:8])[0]
    assert mcp.word_kind(far) == mcp.FAR


def test_list_element_size_and_count_fields():
    word = mcp.pack_list_pointer(0, mcp.POINTER, 0x1FFFFFFF)
    assert (word >> 32) & 0x7 == mcp.POINTER
    assert (word >> 35) & 0x1FFFFFFF == 0x1FFFFFFF
    assert (word >> 2) & 0x3FFFFFFF == 0


def test_far_pointer_field_placement():
    """A far pointer is kind 2, pad bit 2, 29-bit offset, 32-bit segment id."""
    words = struct.unpack("<2Q", mcp.pack_far_pointer(0x1234, 1, 7))
    assert words[0] & 0x3 == mcp.FAR
    assert (words[0] >> 2) & 0x1 == 1
    assert (words[0] >> 3) & 0x1FFFFFFF == 0x1234
    assert words[1] == 7
    words = struct.unpack("<2Q", mcp.pack_far_pointer(0x1234, 0, 7))
    assert (words[0] >> 2) & 0x1 == 0


def test_capability_pointer_field_placement():
    word = mcp.pack_capability_pointer(0xDEADBEEF)
    assert word & 0x3 == mcp.OTHER
    assert (word >> 2) & 0x3FFFFFFF == 0
    assert word >> 32 == 0xDEADBEEF


def test_double_far_pointer_is_far_tag_far_tag():
    tag = mcp.pack_struct_pointer(0, 1, 0)
    words = struct.unpack("<4Q", mcp.pack_double_far_pointer(1, 4, tag, 2, 9))
    assert words[0] & 0x3 == mcp.FAR and (words[0] >> 3) & 0x1FFFFFFF == 4
    assert (words[0] >> 32) & 0xFFFFFFFF == 1
    assert words[1] == tag
    assert words[2] & 0x3 == mcp.FAR and (words[2] >> 3) & 0x1FFFFFFF == 9
    assert (words[2] >> 32) & 0xFFFFFFFF == 2
    assert words[3] == tag


# ---------------------------------------------------------------------------
# Struct layout
# ---------------------------------------------------------------------------


def test_layout_is_empty_for_an_all_pointer_struct():
    layout = mcp.struct_layout(["text", "data", "list"])
    assert layout["offsets"] == [-1, -1, -1]
    assert layout["data_words"] == 0


def test_layout_uses_whole_words():
    layout = mcp.struct_layout(["int8"])
    assert layout["offsets"] == [0]
    assert layout["data_words"] == 1, "a partial word still costs a whole word"


def test_layout_never_reuses_an_occupied_bit():
    """Every field must be fully inside the section and none may overlap."""
    fields = ["int8", "int64", "int16", "bool", "float64", "int32", "bool",
              "int16", "int8", "bool", "float32", "int64", "bool"]
    layout = mcp.struct_layout(fields)
    placed = []
    for name, off, size in zip(fields, layout["offsets"], layout["sizes"]):
        width = mcp.PRIMITIVE_BITS[name]
        assert off % max(width, 1) == 0, f"{name} at {off} is misaligned"
        placed.append((off, off + size))
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            a0, a1 = placed[i]
            b0, b1 = placed[j]
            assert a1 <= b0 or b1 <= a0, f"fields {i} and {j} overlap"
    assert max(e for _, e in placed) <= layout["data_words"] * 64


def test_layout_puts_a_bool_in_a_hole():
    """The whole point of the padding rule: a UInt8 then Int64 then Bool."""
    layout = mcp.struct_layout(["uint8", "int64", "bool"])
    assert layout["offsets"] == [0, 64, 8]
    assert layout["data_words"] == 2


def test_layout_rejects_an_unknown_type():
    with pytest.raises(ValueError):
        mcp.struct_layout(["int32", "complex128"])


# ---------------------------------------------------------------------------
# Reading a data section
# ---------------------------------------------------------------------------


def test_read_fields_extracts_each_width():
    """A data section laid out by hand, read back through the same rule."""
    layout = mcp.struct_layout(["int32", "int64", "uint8", "bool"])
    # Int32 at bit 0, Int64 at bit 64, UInt8 in the byte after the Int32, and
    # the Bool in the bit after the UInt8.
    assert layout["offsets"] == [0, 64, 32, 40]
    data = bytearray(16)
    data[0:4] = struct.pack("<I", 0xDEADBEEF)
    data[8:16] = struct.pack("<Q", 0x0123456789ABCDEF)
    data[4] = 0xBE
    data[5] = 0b0000_0011
    values = mcp.read_struct_fields(bytes(data), layout)
    assert values[0] == 0xDEADBEEF
    assert values[1] == 0x0123456789ABCDEF
    assert values[2] == 0xBE
    assert values[3] == 1


def test_read_fields_preserves_the_full_64_bit_pattern():
    """A negative Int64 arrives as its unsigned pattern, not sign-extended."""
    layout = mcp.struct_layout(["int64"])
    values = mcp.read_struct_fields(struct.pack("<q", -(2 ** 63)), layout)
    assert values[0] == 0x8000000000000000
    assert struct.unpack("<q", struct.pack("<Q", values[0]))[0] == -(2 ** 63)


def test_read_fields_marks_pointer_slots():
    layout = mcp.struct_layout(["int8", "text", "int16"])
    assert layout["offsets"] == [0, -1, 16]
    assert mcp.read_struct_fields(b"\x01\x00\x02\x00\x00\x00\x00\x00",
                                  layout) == [1, None, 2]


def test_read_fields_rejects_a_short_data_section():
    with pytest.raises(ValueError):
        mcp.read_struct_fields(b"\x00", mcp.struct_layout(["int64", "int64"]))


# ---------------------------------------------------------------------------
# Pointer resolution
# ---------------------------------------------------------------------------


def test_resolve_reports_a_null_pointer():
    assert mcp.resolve_pointers(bytes(16), 0, 2) == [-1, -1]


def test_resolve_handles_a_back_pointer():
    """A pointer at word 2 with offset -2 reaches word 1, i.e. byte 8."""
    text_ptr = mcp.pack_list_pointer(-2, mcp.BYTE, 6)
    seg = struct.pack("<3Q", 0, int.from_bytes(b"hello\x00\x00\x00", "little"),
                      text_ptr)
    assert mcp.resolve_pointers(seg, 2, 1) == [8]


def test_resolve_uses_the_pointer_word_not_the_next_one():
    """The target is measured from the word *after* the pointer, not from it."""
    ptr = mcp.pack_list_pointer(0, mcp.EIGHT_BYTES, 1)
    seg = struct.pack("<3Q", 0, ptr, 0x1122334455667788)
    assert mcp.resolve_pointers(seg, 1, 1) == [16]


def test_resolve_marks_far_and_capability_pointers():
    far = struct.unpack("<Q", mcp.pack_far_pointer(12, 0, 1)[:8])[0]
    seg = struct.pack("<2Q", far, 0)
    assert mcp.resolve_pointers(seg, 0, 1) == [-2]
    cap = mcp.pack_capability_pointer(3)
    seg = struct.pack("<2Q", cap, 0)
    assert mcp.resolve_pointers(seg, 0, 1) == [-2]


# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------


def test_frame_header_is_the_segment_count_minus_one():
    assert mcp.frame_message([b"\x00" * 8])[0:4] == struct.pack("<I", 0)
    assert mcp.frame_message([b"\x00" * 8, b"\x00" * 8])[0:4] == struct.pack("<I", 1)


def test_frame_pads_the_table_to_a_word_boundary():
    """Two segments need 4 + 8 = 12 header bytes, so 4 of padding."""
    framed = mcp.frame_message([b"\x01" * 8, b"\x02" * 8])
    assert framed[12:16] == b"\x00\x00\x00\x00"
    assert mcp.parse_frame(framed) == [(16, 8), (24, 8)]


def test_frame_round_trips_multiple_segments():
    segs = [bytes(range(8)), b"\xaa" * 8, b"\x01" * 16]
    framed = mcp.frame_message(segs)
    assert mcp.parse_frame(framed) == [(16, 8), (24, 8), (32, 16)]
    for (off, size), seg in zip(mcp.parse_frame(framed), segs):
        assert framed[off:off + size] == seg


def test_frame_sizes_are_in_words_not_bytes():
    """A 24-byte segment is three words; the header must say 3."""
    framed = mcp.frame_message([b"\x07" * 24])
    assert struct.unpack("<I", framed[4:8])[0] == 3
    assert mcp.parse_frame(framed) == [(8, 24)]


def test_frame_rejects_a_segment_that_is_not_word_aligned():
    with pytest.raises(ValueError):
        mcp.frame_message([b"12345"])


def test_parse_frame_rejects_a_truncated_header():
    with pytest.raises(ValueError):
        mcp.parse_frame(b"\x01\x00\x00\x00")


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------


def test_pack_matches_the_spec_zero_word_example():
    # encoding.html: 32 zero bytes pack to "00 03".
    assert mcp.pack(b"\x00" * 32) == bytes([0x00, 0x03])


def test_pack_matches_the_spec_verbatim_run_example():
    # encoding.html: 32 bytes of 0x8a pack to "ff 8a x8 03 8a x24".
    assert mcp.pack(b"\x8a" * 32) == bytes([0xFF]) + b"\x8a" * 8 + bytes([0x03]) \
        + b"\x8a" * 24


def test_pack_matches_the_spec_tagged_word_example():
    # encoding.html: the two words 08 00 00 00 03 00 02 00 and
    # 19 00 00 00 aa 01 00 00 pack to 51 08 03 02 31 19 aa 01.
    word = bytes.fromhex("0800000003000200") + bytes.fromhex("19000000aa010000")
    assert mcp.pack(word) == bytes.fromhex("51080302") + bytes.fromhex("3119aa01")


def test_pack_emits_nonzero_bytes_in_ascending_byte_order():
    word = bytes.fromhex("0100000000000002")
    assert mcp.pack(word) == bytes([0b1000_0001, 0x01, 0x02])


def test_pack_caps_a_zero_run_at_256_words():
    packed = mcp.pack(bytes(8 * 300))
    assert packed == bytes([0x00, 0xFF, 0x00, 0x2B])


def test_unpack_reproduces_the_spec_examples():
    assert mcp.unpack(bytes([0x00, 0x03]), 4) == b"\x00" * 32
    assert mcp.unpack(bytes([0x51, 0x08, 0x03, 0x02]), 1) == \
        bytes.fromhex("0800000003000200")
    assert mcp.unpack(bytes([0x31, 0x19, 0xAA, 0x01]), 1) == \
        bytes.fromhex("19000000aa010000")
    assert mcp.unpack(bytes([0xFF]) + b"\x8a" * 8 + bytes([0x03]) + b"\x8a" * 24,
                      4) == b"\x8a" * 32


def test_unpack_is_the_inverse_of_pack():
    raw = bytearray(4096)
    for i in range(0, 4096, 97):
        raw[i] = i % 251 + 1
    raw = bytes(raw)
    assert mcp.unpack(mcp.pack(raw), len(raw) // 8) == raw


def test_unpack_stops_at_the_word_budget():
    """A hostile stream must not be able to overrun the caller's buffer."""
    packed = mcp.pack(b"\xff" * 8 * 300)
    assert len(mcp.unpack(packed, 4)) == 32
    assert len(mcp.unpack(packed, 300)) == 300 * 8


def test_unpack_of_an_empty_stream_is_empty():
    assert mcp.unpack(b"", 16) == b""


def test_pack_rejects_a_partial_word():
    with pytest.raises(ValueError):
        mcp.pack(b"1234567")


def test_pack_round_trips_all_byte_patterns():
    for pattern in (b"\x00" * 64, b"\xff" * 64, b"\x01\x00\x00\x00\x00\x00\x00\x00" * 8,
                    bytes(range(64)), b"\x00" * 24 + b"\xff" * 8 + b"\x00" * 32):
        assert mcp.unpack(mcp.pack(pattern), len(pattern) // 8) == pattern
