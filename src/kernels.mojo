"""Cap'n Proto wire-format kernels: struct layout, the packing codec, the 64-bit
pointer words, message framing and the pointer chase that reads a struct back
out.

Every exported symbol takes buffer addresses as plain `Int` values and rebuilds
the pointer inside the body, because `@export` rejects parametric functions and
an inferred pointer origin would make the symbol parametric.

Everything here is exact integer / bitwise work, so the Python tests compare
these against pycapnp byte for byte rather than with a tolerance.
"""

comptime U8Ptr = Pointer[UInt8, AnyOrigin[mut=True]]
comptime U32Ptr = Pointer[UInt32, AnyOrigin[mut=True]]
comptime U64Ptr = Pointer[UInt64, AnyOrigin[mut=True]]

# Marker written into a field's bit offset when the slot is a pointer, which
# takes no space in the data section at all.
comptime NO_OFFSET = 0xFFFFFFFF


def u8p(addr: Int) -> U8Ptr:
    return U8Ptr(unsafe_from_address=addr)


def u32p(addr: Int) -> U32Ptr:
    return U32Ptr(unsafe_from_address=addr)


def u64p(addr: Int) -> U64Ptr:
    return U64Ptr(unsafe_from_address=addr)


# ---------------------------------------------------------------------------
# Struct layout
# ---------------------------------------------------------------------------
# A struct's data section is a bit field. Each data field is placed at the
# lowest bit position that is a multiple of its own size and that does not
# collide with a lower-numbered field, so later fields land in the padding left
# behind by earlier ones. Pointer fields occupy no bits at all. This is the
# rule the Cap'n Proto compiler implements, and it is what makes a struct never
# need more than 63 bits of padding.


@export("cpn_struct_layout")
def cpn_struct_layout(nfields: Int, ptr_mask_addr: Int, bit_sizes_addr: Int,
                      bit_aligns_addr: Int, bit_offsets_addr: Int,
                      words_addr: Int) abi("C"):
    """Lay out a struct. Writes each field's bit offset and the section size.

    `ptr_mask` is a byte per ordinal, non-zero for a pointer slot. `bit_sizes`
    and `bit_aligns` give each data field's width and required alignment in
    bits; a width of zero is a Void field, which is still placed. Offsets for
    pointer slots come back as `NO_OFFSET`. The data-section size in whole words
    is written to `words_addr`.
    """
    var mask = u8p(ptr_mask_addr)
    var sizes = u32p(bit_sizes_addr)
    var aligns = u32p(bit_aligns_addr)
    var offs = u32p(bit_offsets_addr)
    var total = 0
    for i in range(nfields):
        if mask[unsafe_offset=i] != 0:
            offs[unsafe_offset=i] = NO_OFFSET
            continue
        var s = Int(sizes[unsafe_offset=i])
        var a = Int(aligns[unsafe_offset=i])
        if a <= 0:
            a = 1 if s <= 1 else s
        if s == 0:
            offs[unsafe_offset=i] = 0
            continue
        var p = 0
        var q = 0
        while True:
            # Round the candidate up to the field's own alignment.
            q = ((p + a - 1) // a) * a
            var moved = False
            # Re-scan the fields already placed; there are never many of them,
            # and this keeps the kernel allocation-free.
            for j in range(i):
                if offs[unsafe_offset=j] == NO_OFFSET:
                    continue
                var sj = Int(sizes[unsafe_offset=j])
                if sj == 0:
                    continue
                var st = Int(offs[unsafe_offset=j])
                if q < st + sj and st < q + s:
                    p = st + sj
                    moved = True
                    break
            if not moved:
                break
        offs[unsafe_offset=i] = UInt32(q)
        if q + s > total:
            total = q + s
    u32p(words_addr)[unsafe_offset=0] = UInt32((total + 63) >> 6)


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------
# Each 8-byte word becomes a tag byte plus only its non-zero bytes; bit b of the
# tag is set when byte b of the word is non-zero. Tag 0x00 is followed by a run
# length for consecutive zero words, and tag 0xff is followed by the word itself
# and then a run length for consecutive non-zero words copied verbatim.


@export("cpn_pack")
def cpn_pack(src_addr: Int, nwords: Int, out_addr: Int) abi("C") -> Int:
    """Pack `nwords` words. Returns the number of bytes written.

    A run of at most 256 zero words, or 255 verbatim words after a 0xff tag, is
    emitted per marker, so a caller must allow 10 bytes per input word.
    """
    var src = u64p(src_addr)
    var out = u8p(out_addr)
    var pos = 0
    var i = 0
    while i < nwords:
        var w = src[unsafe_offset=i]
        var tag = 0
        for b in range(8):
            if ((w >> UInt64(8 * b)) & 0xFF) != 0:
                tag = tag | (1 << b)
        if tag == 0:
            var run = 1
            while i + run < nwords and run < 256:
                if src[unsafe_offset=i + run] != 0:
                    break
                run += 1
            out[unsafe_offset=pos] = UInt8(0)
            out[unsafe_offset=pos + 1] = UInt8(run - 1)
            pos += 2
            i += run
            continue
        out[unsafe_offset=pos] = UInt8(tag)
        if tag == 0xFF:
            pos += 1
            for b in range(8):
                out[unsafe_offset=pos + b] = UInt8((w >> UInt64(8 * b)) & 0xFF)
            pos += 8
            var run = 0
            while i + 1 + run < nwords and run < 255:
                if src[unsafe_offset=i + 1 + run] == 0:
                    break
                run += 1
            out[unsafe_offset=pos] = UInt8(run)
            pos += 1
            for k in range(run):
                var v = src[unsafe_offset=i + 1 + k]
                for b in range(8):
                    out[unsafe_offset=pos + 8 * k + b] = UInt8(
                        (v >> UInt64(8 * b)) & 0xFF
                    )
            pos += 8 * run
            i += 1 + run
            continue
        pos += 1
        for b in range(8):
            if ((tag >> b) & 1) != 0:
                out[unsafe_offset=pos] = UInt8((w >> UInt64(8 * b)) & 0xFF)
                pos += 1
        i += 1
    return pos


@export("cpn_unpack")
def cpn_unpack(src_addr: Int, nbytes: Int, out_addr: Int,
               maxwords: Int) abi("C") -> Int:
    """Unpack a packed byte stream into words. Returns the words written.

    Stops early rather than writing past `maxwords`, so a hostile or truncated
    stream cannot overrun the caller's buffer.
    """
    var src = u8p(src_addr)
    var out = u64p(out_addr)
    var pos = 0
    var n = 0
    while pos < nbytes and n < maxwords:
        var tag = Int(src[unsafe_offset=pos])
        pos += 1
        var w = UInt64(0)
        for b in range(8):
            if ((tag >> b) & 1) != 0:
                if pos >= nbytes:
                    return n
                w = w | (UInt64(src[unsafe_offset=pos]) << UInt64(8 * b))
                pos += 1
        if n >= maxwords:
            return n
        out[unsafe_offset=n] = w
        n += 1
        if tag == 0:
            # A 0x00 tag is the first of a run of zero words.
            if pos >= nbytes:
                return n
            var run = Int(src[unsafe_offset=pos])
            pos += 1
            while run > 0 and n < maxwords:
                out[unsafe_offset=n] = UInt64(0)
                n += 1
                run -= 1
            continue
        if tag == 0xFF:
            # The eight bytes of the word have already been read above, because
            # 0xff means "all eight bytes follow" as well as "verbatim run".
            if pos >= nbytes:
                return n
            var run = Int(src[unsafe_offset=pos])
            pos += 1
            while run > 0 and pos + 8 <= nbytes and n < maxwords:
                var x = UInt64(0)
                for b in range(8):
                    x = x | (UInt64(src[unsafe_offset=pos + b]) << UInt64(8 * b))
                pos += 8
                out[unsafe_offset=n] = x
                n += 1
                run -= 1
    return n


# ---------------------------------------------------------------------------
# Pointer words
# ---------------------------------------------------------------------------
# Every pointer word puts its kind in the two least significant bits, so the
# offset field starts at bit 2 rather than at bit 0.
# Struct pointer: bits 0..1 kind (0 = struct), bits 2..31 signed word offset from
# the word after the pointer, bits 32..47 data-section size in words, bits 48..63
# pointer-section size in words.
# List pointer: bits 0..1 kind (1 = list), bits 2..31 signed word offset, bits
# 32..34 element size code, bits 35..63 element count.
# Far pointer: bits 0..1 kind 2, bit 2 one/two-word landing pad, bits 3..31
# unsigned word offset into the target segment, bits 32..63 that segment's id.
# Capability (other) pointer: bits 0..1 kind 3, bits 2..31 zero, bits 32..63 the
# capability's index in the message's capability table.


@export("cpn_pack_struct_ptr")
def cpn_pack_struct_ptr(offset_words: Int, data_words: Int, ptr_count: Int,
                        out_addr: Int) abi("C"):
    var out = u64p(out_addr)
    out[unsafe_offset=0] = (
        (UInt64(offset_words & 0x3FFFFFFF) << UInt64(2))
        | (UInt64(data_words & 0xFFFF) << UInt64(32))
        | (UInt64(ptr_count & 0xFFFF) << UInt64(48))
    )


@export("cpn_unpack_struct_ptr")
def cpn_unpack_struct_ptr(word: Int, out_addr: Int) abi("C"):
    """Write (sign-extended offset, data words, pointer count) as 3 u32s."""
    var out = u32p(out_addr)
    var w = UInt64(word)
    out[unsafe_offset=0] = UInt32(_sign30(w >> UInt64(2)))
    out[unsafe_offset=1] = UInt32((w >> UInt64(32)) & 0xFFFF)
    out[unsafe_offset=2] = UInt32((w >> UInt64(48)) & 0xFFFF)


@export("cpn_pack_list_ptr")
def cpn_pack_list_ptr(offset_words: Int, elem_size: Int, count: Int,
                      out_addr: Int) abi("C"):
    var out = u64p(out_addr)
    out[unsafe_offset=0] = (
        (UInt64(offset_words & 0x3FFFFFFF) << UInt64(2))
        | UInt64(0x1)
        | (UInt64(elem_size & 0x7) << UInt64(32))
        | (UInt64(count & 0x1FFFFFFF) << UInt64(35))
    )


@export("cpn_unpack_list_ptr")
def cpn_unpack_list_ptr(word: Int, out_addr: Int) abi("C"):
    """Write (sign-extended offset, element size code, count) as 3 u32s."""
    var out = u32p(out_addr)
    var w = UInt64(word)
    out[unsafe_offset=0] = UInt32(_sign30(w >> UInt64(2)))
    out[unsafe_offset=1] = UInt32((w >> UInt64(32)) & 0x7)
    out[unsafe_offset=2] = UInt32((w >> UInt64(35)) & 0x1FFFFFFF)


@export("cpn_pack_far_ptr")
def cpn_pack_far_ptr(offset_words: Int, two_word_pad: Int, segment_id: Int,
                     out_addr: Int) abi("C"):
    """Write the two words of a far pointer."""
    var out = u64p(out_addr)
    var b = 1 if two_word_pad != 0 else 0
    out[unsafe_offset=0] = (
        UInt64(0x2) | (UInt64(b) << UInt64(2)) | (UInt64(offset_words & 0x1FFFFFFF) << UInt64(3))
    )
    out[unsafe_offset=1] = UInt64(segment_id & 0xFFFFFFFF)


@export("cpn_pack_double_far_ptr")
def cpn_pack_double_far_ptr(pad_segment: Int, pad_word: Int, tag_word: Int,
                            content_segment: Int, content_word: Int,
                            out_addr: Int) abi("C"):
    """Write the four words of a double-far pointer.

    The tag word looks exactly like the intra-segment pointer to the target
    object except that its offset is zero, and the same tag follows both far
    pointers.
    """
    var out = u64p(out_addr)
    out[unsafe_offset=0] = (
        UInt64(0x2) | (UInt64(pad_word & 0x1FFFFFFF) << UInt64(3))
        | (UInt64(pad_segment & 0xFFFFFFFF) << UInt64(32))
    )
    out[unsafe_offset=1] = UInt64(tag_word)
    out[unsafe_offset=2] = (
        UInt64(0x2) | (UInt64(content_word & 0x1FFFFFFF) << UInt64(3))
        | (UInt64(content_segment & 0xFFFFFFFF) << UInt64(32))
    )
    out[unsafe_offset=3] = UInt64(tag_word)


@export("cpn_pack_capability_ptr")
def cpn_pack_capability_ptr(capability_index: Int, out_addr: Int) abi("C"):
    """Write a capability ("other") pointer: kind 3, zero offset, 32-bit index."""
    var out = u64p(out_addr)
    out[unsafe_offset=0] = (
        UInt64(0x3) | (UInt64(capability_index & 0xFFFFFFFF) << UInt64(32))
    )


@export("cpn_word_kind")
def cpn_word_kind(word: Int) abi("C") -> Int:
    return Int(UInt64(word) & 0x3)


# ---------------------------------------------------------------------------
# Message framing
# ---------------------------------------------------------------------------
# (4 bytes) the segment count minus one, (4 bytes per segment) that segment's
# size in words, (0 or 4 bytes) padding up to a word boundary, then the segments
# in order. All of it is unsigned little-endian.


@export("cpn_frame_message")
def cpn_frame_message(nseg: Int, words_addr: Int, payload_addr: Int,
                      out_addr: Int) abi("C") -> Int:
    """Frame `nseg` segments. `words` holds each segment's length in words.

    Returns the total framed length in bytes.
    """
    var counts = u32p(words_addr)
    var out = u8p(out_addr)
    var table = 4 + 4 * nseg
    if (table % 8) != 0:
        table += 4
    var pos = 0
    # The header word is just (segment count - 1); there is no shift.
    var head = UInt32(nseg - 1)
    for b in range(4):
        out[unsafe_offset=pos + b] = UInt8((head >> UInt32(8 * b)) & 0xFF)
    pos += 4
    for s in range(nseg):
        var sz = counts[unsafe_offset=s]
        for b in range(4):
            out[unsafe_offset=pos + b] = UInt8((sz >> UInt32(8 * b)) & 0xFF)
        pos += 4
    while pos < table:
        out[unsafe_offset=pos] = UInt8(0)
        pos += 1
    # The payload is a whole number of words, so copy it a word at a time; a
    # byte loop here is several times slower than the move the format needs.
    var src = u64p(payload_addr)
    var dst = u64p(out_addr)
    var wbase = 0
    for s in range(nseg):
        var szw = Int(counts[unsafe_offset=s])
        var at = pos >> 3
        for w in range(szw):
            dst[unsafe_offset=at + w] = src[unsafe_offset=wbase + w]
        pos += szw * 8
        wbase += szw
    return pos


@export("cpn_parse_frame")
def cpn_parse_frame(msg_addr: Int, msglen: Int, words_addr: Int,
                    offsets_addr: Int, maxseg: Int) abi("C") -> Int:
    """Parse a framed message's segment table.

    Writes each segment's length in words and its byte offset within the
    message. Returns the segment count, or a negative code if the header is
    truncated (-1) or claims more segments than the caller can hold (-2).
    """
    var msg = u8p(msg_addr)
    var counts = u32p(words_addr)
    var offsets = u32p(offsets_addr)
    if msglen < 8:
        return -1
    var nseg = 1 + Int(
        Int(msg[unsafe_offset=0])
        | (Int(msg[unsafe_offset=1]) << 8)
        | (Int(msg[unsafe_offset=2]) << 16)
    )
    if nseg > maxseg:
        return -2
    var table = 4 + 4 * nseg
    if (table % 8) != 0:
        table += 4
    if msglen < table:
        return -1
    var pos = 4
    for s in range(nseg):
        var sz = UInt32(0)
        for b in range(4):
            sz = sz | (UInt32(msg[unsafe_offset=pos + b]) << UInt32(8 * b))
        counts[unsafe_offset=s] = sz
        pos += 4
    pos = table
    var total = 0
    for s in range(nseg):
        offsets[unsafe_offset=s] = UInt32(pos + total)
        total += Int(counts[unsafe_offset=s]) * 8
    return nseg


# ---------------------------------------------------------------------------
# Struct traversal
# ---------------------------------------------------------------------------


@export("cpn_read_struct_fields")
def cpn_read_struct_fields(data_addr: Int, nfields: Int, bit_offsets_addr: Int,
                           bit_sizes_addr: Int, values_addr: Int) abi("C"):
    """Read every data field of a struct out of its data section.

    Offsets and sizes are in bits, as `cpn_struct_layout` produced them. Values
    are zero-extended into 64-bit slots, because a struct's default values are
    xor'd in and reinterpreting a field is the schema's job, exactly as in the
    real format. Pointer slots come back as 0xFFFFFFFFFFFFFFFF.
    """
    var offs = u32p(bit_offsets_addr)
    var sizes = u32p(bit_sizes_addr)
    var data = u8p(data_addr)
    var vals = u64p(values_addr)
    for i in range(nfields):
        var off = Int(offs[unsafe_offset=i])
        if off == NO_OFFSET:
            vals[unsafe_offset=i] = UInt64(0xFFFFFFFFFFFFFFFF)
            continue
        var s = Int(sizes[unsafe_offset=i])
        if s == 0:
            vals[unsafe_offset=i] = UInt64(0)
            continue
        if s == 1:
            var byte = UInt64(data[unsafe_offset=off >> 3])
            vals[unsafe_offset=i] = (byte >> UInt64(off & 7)) & 0x1
            continue
        var v = UInt64(0)
        for b in range(s // 8):
            v = v | (UInt64(data[unsafe_offset=(off >> 3) + b]) << UInt64(8 * b))
        vals[unsafe_offset=i] = v


@export("cpn_resolve_ptrs")
def cpn_resolve_ptrs(seg_addr: Int, base_word: Int, nptrs: Int,
                     targets_addr: Int) abi("C"):
    """Resolve a run of pointer words inside a segment to target byte offsets.

    The pointer words are read straight out of the segment at words
    `base_word .. base_word+nptrs-1`, so this is the real pointer chase rather
    than a copy. A pointer at word w with signed offset o targets word
    w + 1 + o. Writes the target byte offset into `targets`; -1 marks a null
    pointer and -2 a far or capability pointer needing multi-word handling.
    """
    var seg = u8p(seg_addr)
    var tgt = u64p(targets_addr)
    for i in range(nptrs):
        var byte = (base_word + i) * 8
        var w = UInt64(0)
        for b in range(8):
            w = w | (UInt64(seg[unsafe_offset=byte + b]) << UInt64(8 * b))
        if w == 0:
            # Only an all-zero word is null. A list pointer with offset zero is
            # a perfectly ordinary pointer at the word right after itself.
            tgt[unsafe_offset=i] = UInt64(0xFFFFFFFFFFFFFFFF)
            continue
        var kind = Int(w & 0x3)
        if kind == 2 or kind == 3:
            tgt[unsafe_offset=i] = UInt64(0xFFFFFFFFFFFFFFFE)
            continue
        var raw = Int((w >> UInt64(2)) & 0x3FFFFFFF)
        var signed_off = raw
        if raw >= 0x20000000:
            signed_off = raw - 0x40000000
        tgt[unsafe_offset=i] = UInt64((base_word + i + 1 + signed_off) * 8)


def _sign30(w: UInt64) -> Int:
    """Sign-extend the 30-bit offset field shared by struct and list pointers."""
    var raw = Int(w & 0x3FFFFFFF)
    if raw >= 0x20000000:
        return raw - 0x40000000
    return raw
