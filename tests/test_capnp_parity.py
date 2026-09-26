"""Byte-exact parity between the Mojo kernels and pycapnp.

Cap'n Proto is a bit-level format: the pointer words, the segment table and the
packing stream are defined byte for byte, so every assertion here is exact
equality. A wrong shift, a mis-sized data section or a dropped run marker all
show up immediately as a diff rather than as a small numerical drift.
"""

import struct

import pytest

from conftest import _capnp, requires_capnp, segments_of

import mojo_capnproto as mcp

pytestmark = requires_capnp


# ---------------------------------------------------------------------------
# Struct layout, against what the Cap'n Proto compiler actually emits
# ---------------------------------------------------------------------------


def test_point_layout_matches_pycapnp_data_section_size(schema):
    msg = schema.Point.new_message(x=1, y=2, name="a", flags=b"", vals=[])
    seg = segments_of(msg)[0]
    _, data_words, ptrs = mcp.unpack_struct_pointer(struct.unpack("<Q", seg[0:8])[0])
    layout = mcp.struct_layout(["int32", "int64", "text", "data", "list"])
    assert layout["data_words"] == data_words
    assert ptrs == 3
    assert layout["offsets"] == [0, 64, -1, -1, -1]


def test_scene_layout_matches_pycapnp(schema):
    """UInt8 then Int64 then Bool: the Bool must land in the 8-bit hole."""
    msg = schema.Scene.new_message()
    msg.origin.x, msg.origin.y, msg.origin.name, msg.origin.flags = 1, 2, "a", b"\xaa"
    msg.origin.vals = [9]
    msg.pts = []
    msg.label = "zz"
    msg.count, msg.big, msg.tiny = 200, -(2 ** 40), True
    seg = segments_of(msg)[0]
    _, data_words, _ = mcp.unpack_struct_pointer(struct.unpack("<Q", seg[0:8])[0])
    layout = mcp.struct_layout(
        ["pointer", "pointer", "pointer", "uint8", "int64", "bool"])
    assert layout["data_words"] == data_words
    # count at bit 0, big needs 64-bit alignment so it goes to bit 64, and the
    # Bool then fills the hole the UInt8 left at bit 8.
    assert layout["offsets"] == [-1, -1, -1, 0, 64, 8]


def test_wide_layout_reads_every_field_back_correctly(schema):
    """A field order that forces several padding holes.

    Asserting only on data_words would be too weak: a layout that put a field
    one bit off still rounds to the same word count. Reading every field back
    out of pycapnp's own bytes and comparing to what was written is the check
    that would actually fail.
    """
    msg = schema.Wide.new_message(a=-3, b=-(2 ** 60), c=-300, d=True,
                                  e=-1.5, f=70000, g=False)
    seg = segments_of(msg)[0]
    _, data_words, ptrs = mcp.unpack_struct_pointer(struct.unpack("<Q", seg[0:8])[0])
    layout = mcp.struct_layout(
        ["int8", "int64", "int16", "bool", "float64", "int32", "bool"])
    assert ptrs == 0
    assert layout["data_words"] == data_words
    offsets = layout["offsets"]
    assert offsets[1] == 64, "Int64 must be 64-bit aligned"
    assert offsets[4] % 64 == 0, "Float64 must be 64-bit aligned"
    assert offsets[5] % 32 == 0
    assert max(offsets[i] + layout["sizes"][i] for i in range(7)) <= data_words * 64

    values = mcp.read_struct_fields(seg[8:8 + 8 * data_words], layout)
    assert struct.unpack("<b", struct.pack("<B", values[0] & 0xFF))[0] == -3
    assert struct.unpack("<q", struct.pack("<Q", values[1]))[0] == -(2 ** 60)
    assert struct.unpack("<h", struct.pack("<H", values[2] & 0xFFFF))[0] == -300
    assert values[3] == 1
    assert struct.unpack("<d", struct.pack("<Q", values[4]))[0] == -1.5
    assert struct.unpack("<i", struct.pack("<I", values[5] & 0xFFFFFFFF))[0] == 70000
    assert values[6] == 0


# ---------------------------------------------------------------------------
# Reading real field values back out
# ---------------------------------------------------------------------------


def test_read_scene_fields_matches_pycapnp_values(schema):
    msg = schema.Scene.new_message()
    msg.origin.x, msg.origin.y, msg.origin.name = 7, -(2 ** 40), "n"
    msg.origin.flags = b"\x01"
    msg.origin.vals = [1]
    msg.pts = []
    msg.label = "L"
    msg.count, msg.big, msg.tiny = 200, -(2 ** 40), True
    seg = segments_of(msg)[0]
    layout = mcp.struct_layout(
        ["pointer", "pointer", "pointer", "uint8", "int64", "bool"])
    data = seg[8:8 + 8 * layout["data_words"]]
    values = mcp.read_struct_fields(data, layout)
    assert values[3] == 200
    # Int64 is read as a raw little-endian 64-bit pattern, so reinterpreting it
    # the way the schema does has to give the negative number back.
    assert struct.unpack("<q", struct.pack("<Q", values[4]))[0] == -(2 ** 40)
    assert values[5] == 1
    assert values[:3] == [None, None, None]


def test_read_point_int32_and_int64(schema):
    msg = schema.Point.new_message(x=-7, y=-(2 ** 62), name="", flags=b"", vals=[])
    seg = segments_of(msg)[0]
    layout = mcp.struct_layout(["int32", "int64", "text", "data", "list"])
    data = seg[8:8 + 8 * layout["data_words"]]
    values = mcp.read_struct_fields(data, layout)
    assert struct.unpack("<i", struct.pack("<I", values[0] & 0xFFFFFFFF))[0] == -7
    assert struct.unpack("<q", struct.pack("<Q", values[1]))[0] == -(2 ** 62)


def test_read_bool_uses_a_single_bit(schema):
    """A Bool wider than one bit would still 'work' for True, so check False."""
    msg = schema.Scene.new_message()
    msg.origin.x, msg.origin.y, msg.origin.name, msg.origin.flags = 0, 0, "", b""
    msg.origin.vals = []
    msg.pts = []
    msg.label = ""
    msg.count, msg.big, msg.tiny = 1, 0, False
    seg = segments_of(msg)[0]
    layout = mcp.struct_layout(
        ["pointer", "pointer", "pointer", "uint8", "int64", "bool"])
    data = seg[8:8 + 8 * layout["data_words"]]
    assert mcp.read_struct_fields(data, layout)[5] == 0


# ---------------------------------------------------------------------------
# Framing, byte for byte
# ---------------------------------------------------------------------------


def test_frame_message_is_byte_identical_to_pycapnp(schema):
    msg = schema.Point.new_message(x=7, y=-1234567890123, name="hello",
                                   flags=b"\x01\x02\xff", vals=[1, 2, 3])
    segs = segments_of(msg)
    framed = msg.to_bytes()
    assert mcp.frame_message(segs) == framed


def test_frame_matches_for_a_large_text_field(schema):
    text = "the quick brown fox " * 40
    msg = schema.Point.new_message(x=1, y=2, name=text, flags=b"\x00" * 300,
                                   vals=list(range(200)))
    segs = segments_of(msg)
    framed = msg.to_bytes()
    msg.clear_write_flag()
    assert mcp.frame_message(segs) == framed


def test_frame_matches_for_a_list_of_structs(schema):
    msg = schema.Scene.new_message()
    msg.origin.x, msg.origin.y, msg.origin.name, msg.origin.flags = 1, 2, "a", b"\xaa"
    msg.origin.vals = [1, 2, 3]
    msg.label = "scene"
    msg.count, msg.big, msg.tiny = 7, -(2 ** 40), True
    pts = msg.init("pts", 4)
    for i in range(4):
        pts[i].x, pts[i].y = i, i * 2
        pts[i].name = f"p{i}"
        pts[i].flags = bytes([i])
        pts[i].vals = [i, i + 1]
    segs = segments_of(msg)
    framed = msg.to_bytes()
    msg.clear_write_flag()
    assert mcp.frame_message(segs) == framed


def test_parse_frame_recovers_pycapnp_segments(schema):
    msg = schema.Point.new_message(x=3, y=4, name="abc", flags=b"\xff", vals=[9, 8])
    segs = segments_of(msg)
    raw = msg.to_bytes()
    msg.clear_write_flag()
    offsets = mcp.parse_frame(raw)
    assert [size for _, size in offsets] == [len(s) for s in segs]
    for (off, size), seg in zip(offsets, segs):
        assert raw[off:off + size] == seg


# ---------------------------------------------------------------------------
# Pointer resolution against a real message
# ---------------------------------------------------------------------------


def test_resolve_pointers_finds_the_text_data_and_list(schema):
    msg = schema.Point.new_message(x=7, y=8, name="hello", flags=b"\x01\x02\xff",
                                   vals=[1, 2, 3])
    seg = segments_of(msg)[0]
    root_off, root_size, root_ptrs = mcp.unpack_struct_pointer(
        struct.unpack("<Q", seg[0:8])[0])
    assert (root_off, root_ptrs) == (0, 3)
    # The pointer section starts one word past the end of the data section.
    targets = mcp.resolve_pointers(seg, 1 + root_size, root_ptrs)
    assert seg[targets[0]:targets[0] + 6] == b"hello\x00"
    assert seg[targets[1]:targets[1] + 3] == b"\x01\x02\xff"
    assert seg[targets[2]:targets[2] + 12] == struct.pack("<3i", 1, 2, 3)


def test_resolve_pointers_reads_the_actual_list_pointer(schema):
    """The resolved target's own list pointer must describe what is there."""
    msg = schema.Point.new_message(x=1, y=2, name="hi", flags=b"\x01\x02",
                                   vals=[5, 6, 7])
    seg = segments_of(msg)[0]
    _, root_size, root_ptrs = mcp.unpack_struct_pointer(
        struct.unpack("<Q", seg[0:8])[0])
    targets = mcp.resolve_pointers(seg, 1 + root_size, root_ptrs)
    assert seg[targets[0]:targets[0] + 3] == b"hi\x00"
    assert seg[targets[1]:targets[1] + 2] == b"\x01\x02"
    # targets[2] is where the list content starts, and the pointer describing
    # it is the third word of the pointer section.
    ptr_word = struct.unpack("<Q", seg[(1 + root_size + 2) * 8:
                                      (1 + root_size + 3) * 8])[0]
    off, elem, count = mcp.unpack_list_pointer(ptr_word)
    assert (elem, count) == (mcp.FOUR_BYTES, 3)
    base = (1 + root_size + 2 + 1 + off) * 8
    assert base == targets[2]
    assert seg[base:base + 12] == struct.pack("<3i", 5, 6, 7)


def test_resolve_pointers_reports_nulls_for_unset_fields(schema):
    """A Text/Data/List field that is never assigned stays a null pointer."""
    msg = schema.Point.new_message(x=1, y=2)
    seg = segments_of(msg)[0]
    _, data_words, ptrs = mcp.unpack_struct_pointer(struct.unpack("<Q", seg[0:8])[0])
    assert mcp.resolve_pointers(seg, 1 + data_words, ptrs) == [-1, -1, -1]


def test_resolve_pointers_finds_nested_structs(schema):
    """A List(Point) is a composite list; its tag word carries the layout."""
    msg = schema.Scene.new_message()
    msg.origin.x, msg.origin.y, msg.origin.name, msg.origin.flags = 1, 2, "a", b"\xaa"
    msg.origin.vals = [1]
    msg.label = "L"
    msg.count, msg.big, msg.tiny = 1, 0, False
    pts = msg.init("pts", 2)
    for i in range(2):
        pts[i].x, pts[i].y = 10 + i, 20 + i
        pts[i].name = "p"
        pts[i].flags = b"\x00"
        pts[i].vals = []
    seg = segments_of(msg)[0]
    _, data_words, ptrs = mcp.unpack_struct_pointer(struct.unpack("<Q", seg[0:8])[0])
    targets = mcp.resolve_pointers(seg, 1 + data_words, ptrs)
    # pts is the second declared pointer field.
    # A struct list is a composite list: the pointer lands on a tag word that
    # has the same shape as a struct pointer but counts elements in its offset.
    ptr_word = struct.unpack("<Q", seg[(1 + data_words + 1) * 8:
                                       (1 + data_words + 2) * 8])[0]
    _, elem_code, words_total = mcp.unpack_list_pointer(ptr_word)
    assert elem_code == mcp.COMPOSITE
    tag = struct.unpack("<Q", seg[targets[1]:targets[1] + 8])[0]
    # For a composite list the tag's offset field counts elements and its size
    # fields give each element's data and pointer sections, so the list holds
    # (2 + 3) * 2 = 10 words, which is what the pointer's D field recorded.
    tag_count, elem_data, tag_ptrs = mcp.unpack_struct_pointer(tag)
    assert (tag_count, elem_data, tag_ptrs) == (2, 2, 3)
    assert words_total == (2 + 3) * 2
    # Each element is a whole struct in place: its data words then its pointer
    # words, so consecutive elements are (2 + 3) words apart.
    stride = (elem_data + tag_ptrs) * 8
    body = targets[1] + 8
    point = mcp.struct_layout(["int32", "int64", "text", "data", "list"])
    assert point["data_words"] == elem_data
    for i, (want_x, want_y) in enumerate([(10, 20), (11, 21)]):
        start = body + i * stride
        section = seg[start:start + elem_data * 8]
        fields = mcp.read_struct_fields(section, point)
        assert struct.unpack("<i", struct.pack("<I", fields[0]))[0] == want_x
        assert struct.unpack("<q", struct.pack("<Q", fields[1]))[0] == want_y
        # The element's three pointers follow its data section, the first
        # being the Text field, which here is a single NUL-terminated byte.
        ptr_word = struct.unpack("<Q", seg[start + elem_data * 8:
                                         start + elem_data * 8 + 8])[0]
        _, elem, count = mcp.unpack_list_pointer(ptr_word)
        assert (elem, count) == (mcp.BYTE, 2)



# ---------------------------------------------------------------------------
# Packing, checked against pycapnp's C++ reference decoder
# ---------------------------------------------------------------------------


def test_packed_message_is_readable_by_pycapnp(schema):
    """The strongest packing test: hand our bytes to the C++ reference reader."""
    msg = schema.Point.new_message(x=7, y=-1234567890123, name="hello",
                                   flags=b"\x01\x02\xff", vals=[1, 2, 3])
    raw = msg.to_bytes()
    msg.clear_write_flag()
    readers = list(_capnp.read_multiple_bytes_packed(mcp.pack(raw)))
    assert len(readers) == 1
    got = readers[0].as_struct(schema.Point)
    assert got.x == 7
    assert got.y == -1234567890123
    assert got.name == "hello"
    assert bytes(got.flags) == b"\x01\x02\xff"
    assert list(got.vals) == [1, 2, 3]


def test_pack_round_trips_through_pycapnp_built_messages(schema):
    """Several shapes, packed by Mojo and decompressed by the reference."""
    for name, kwargs in [
        ("empty", dict(x=0, y=0, name="", flags=b"", vals=[])),
        ("text", dict(x=-1, y=1, name="x" * 500, flags=b"", vals=[])),
        ("data", dict(x=1, y=-1, name="", flags=bytes(range(256)), vals=[])),
        ("list", dict(x=1, y=2, name="n", flags=b"", vals=list(range(500)))),
    ]:
        msg = schema.Point.new_message(**kwargs)
        raw = msg.to_bytes()
        msg.clear_write_flag()
        got = list(_capnp.read_multiple_bytes_packed(mcp.pack(raw)))[0]
        got = got.as_struct(schema.Point)
        assert got.x == kwargs["x"] and got.y == kwargs["y"], name
        assert got.name == kwargs["name"], name
        assert bytes(got.flags) == kwargs["flags"], name
        assert list(got.vals) == kwargs["vals"], name


def test_pack_round_trips_a_message_with_many_zero_words(schema):
    """A long run of zero words is where a wrong run length shows up."""
    msg = schema.Point.new_message(x=1, y=2, name="q", flags=b"\x01",
                                   vals=list(range(2000)))
    raw = msg.to_bytes()
    msg.clear_write_flag()
    packed = mcp.pack(raw)
    assert len(packed) < len(raw)
    got = list(_capnp.read_multiple_bytes_packed(packed))[0].as_struct(schema.Point)
    assert list(got.vals) == list(range(2000))
