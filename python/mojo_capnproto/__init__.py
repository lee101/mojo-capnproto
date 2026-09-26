"""mojo_capnproto — the Cap'n Proto wire format, compiled.

Cap'n Proto is a binary serialisation format, so the surface worth compiling is
the byte-level one: the struct layout rule, the packing codec, the 64-bit
pointer words, message framing, and the pointer chase that reads a struct back
out. All of it is pure integer and bitwise work, so it is bit-exact against
`capnp` rather than merely close.

This is not a Cap'n Proto implementation. It does not parse `.capnp` schemas,
generate classes, build messages from typed fields, do RPC, or manage a
traversal limit. Use the real `capnp` package for all of that. What you get here
is the arithmetic underneath, so a hand-rolled reader, a bulk codec or a
verification pass can run it compiled.

    >>> import mojo_capnproto as mcp
    >>> layout = mcp.struct_layout(["int32", "int64", "text", "bool"])
    >>> layout["offsets"], layout["data_words"]
    ([0, 64, -1, 8], 2)
    >>> mcp.unpack_struct_pointer(mcp.pack_struct_pointer(3, 2, 1))
    (3, 2, 1)
    >>> mcp.pack(b"\\x00" * 24) == b"\\x00\\x02"
    True
"""

from ._lib import (  # noqa: F401
    BIT,
    BYTE,
    COMPOSITE,
    EIGHT_BYTES,
    ELEM_WIDTH,
    FAR,
    FOUR_BYTES,
    LIST,
    OTHER,
    POINTER,
    PRIMITIVE_BITS,
    STRUCT,
    TWO_BYTES,
    VOID,
    frame_message,
    pack,
    pack_capability_pointer,
    pack_double_far_pointer,
    pack_far_pointer,
    pack_list_pointer,
    pack_struct_pointer,
    parse_frame,
    read_struct_fields,
    resolve_pointers,
    struct_layout,
    unpack,
    unpack_list_pointer,
    unpack_struct_pointer,
    word_kind,
)

__version__ = "0.1.0"

__all__ = [
    "struct_layout",
    "read_struct_fields",
    "pack",
    "unpack",
    "pack_struct_pointer",
    "unpack_struct_pointer",
    "pack_list_pointer",
    "unpack_list_pointer",
    "pack_far_pointer",
    "pack_double_far_pointer",
    "pack_capability_pointer",
    "word_kind",
    "frame_message",
    "parse_frame",
    "resolve_pointers",
    "PRIMITIVE_BITS",
    "ELEM_WIDTH",
    "STRUCT",
    "LIST",
    "FAR",
    "OTHER",
    "VOID",
    "BIT",
    "BYTE",
    "TWO_BYTES",
    "FOUR_BYTES",
    "EIGHT_BYTES",
    "POINTER",
    "COMPOSITE",
]
