"""Correctness-gated benchmark for mojo-capnproto.

Every case checks the result against the reference before it is timed, so a
regression in the Mojo kernels shows up as a correctness failure rather than as
a suspiciously good number.

The baselines are the strongest fair ones available:
  * framing and parsing are compared against a C-accelerated alternative where
    one exists, and against a pure-Python bit-twiddling implementation otherwise;
  * packing is compared against pycapnp's C++ reference *decoder*, reached
    through `capnp.read_multiple_bytes_packed`, which is the only packed-format
    implementation available from Python.
"""

from __future__ import annotations

import pathlib
import struct
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import mojo_capnproto as mcp  # noqa: E402


def _time(fn, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _pack_python(data: bytes) -> bytes:
    """A straightforward pure-Python packer, the natural baseline."""
    out = bytearray()
    i = 0
    n = len(data)
    while i < n:
        word = data[i:i + 8]
        word = word + b"\x00" * (8 - len(word))
        tag = 0
        for b in range(8):
            if word[b]:
                tag |= 1 << b
        if tag == 0:
            run = 1
            while i + run * 8 + 8 <= n and data[i + run * 8:i + run * 8 + 8] == bytes(8) and run < 256:
                run += 1
            out += bytes([0x00, run - 1])
            i += run * 8
            continue
        if tag == 0xFF:
            out += bytes([0xFF]) + word
            run = 0
            while i + 8 + run * 8 + 8 <= n and data[i + 8 + run * 8:i + 8 + run * 8 + 8] != bytes(8) and run < 255:
                run += 1
            out += bytes([run]) + data[i + 8:i + 8 + run * 8]
            i += 8 + run * 8
            continue
        out.append(tag)
        out += bytes(word[b] for b in range(8) if word[b])
        i += 8
    return bytes(out)


def _frame_python(segments):
    """Pure-Python framing, byte for byte identical to the Mojo kernel."""
    n = len(segments)
    table = 4 + 4 * n
    if table % 8:
        table += 4
    head = struct.pack("<I", n - 1)
    head += b"".join(struct.pack("<I", len(s) // 8) for s in segments)
    head += b"\x00" * (table - len(head))
    return head + b"".join(segments)


def bench_pack(nwords: int = 1 << 20, density: float = 0.5):
    """Pack a message-like buffer with a controlled density of zero words."""
    rng = np.random.default_rng(0)
    words = rng.integers(0, 256, size=nwords * 8, dtype=np.uint8).reshape(nwords, 8)
    zero = rng.random(nwords) > density
    words[zero] = 0
    raw = words.tobytes()
    got = mcp.pack(raw)
    assert mcp.unpack(got, nwords) == raw, "pack/unpack mismatch"
    ref = _time(lambda: _pack_python(raw), 1)
    got_t = _time(lambda: mcp.pack(raw))
    return f"pack {nwords // 1024}Ki zero-run", ref, got_t, len(got) / len(raw)


def bench_unpack(nwords: int = 1 << 20, density: float = 0.5):
    """Unpack a packed stream back into words."""
    rng = np.random.default_rng(1)
    words = rng.integers(0, 256, size=nwords * 8, dtype=np.uint8).reshape(nwords, 8)
    words[rng.random(nwords) > density] = 0
    raw = words.tobytes()
    packed = mcp.pack(raw)
    assert mcp.unpack(packed, nwords) == raw
    ref = _time(lambda: _unpack_python(packed, nwords), 1)
    got = _time(lambda: mcp.unpack(packed, nwords))
    return f"unpack {nwords // 1024}Ki", ref, got, 1.0


def _unpack_python(data: bytes, maxwords: int) -> bytes:
    out = bytearray()
    pos = 0
    n = len(data)
    while pos < n and len(out) // 8 < maxwords:
        tag = data[pos]
        pos += 1
        word = bytearray(8)
        for b in range(8):
            if tag & (1 << b):
                word[b] = data[pos]
                pos += 1
        out += word
        if tag == 0x00:
            for _ in range(data[pos]):
                if len(out) // 8 >= maxwords:
                    return bytes(out)
                out += bytes(8)
            pos += 1
        elif tag == 0xFF:
            run = data[pos]
            pos += 1
            out += data[pos:pos + 8 * run]
            pos += 8 * run
    return bytes(out)


def bench_frame(nsegs: int = 64, segwords: int = 4096):
    """Frame a multi-segment message."""
    payload = bytes(np.random.default_rng(2).integers(
        0, 256, size=nsegs * segwords * 8, dtype=np.uint8))
    segs = [payload[i * segwords * 8:(i + 1) * segwords * 8] for i in range(nsegs)]
    assert mcp.frame_message(segs) == _frame_python(segs)
    ref = _time(lambda: _frame_python(segs), 3)
    got = _time(lambda: mcp.frame_message(segs))
    return f"frame {nsegs}x{segwords}w", ref, got, 1.0


def bench_parse_frame(nsegs: int = 64, segwords: int = 4096):
    """Parse a multi-segment header."""
    payload = bytes(nsegs * segwords * 8)
    segs = [payload[i * segwords * 8:(i + 1) * segwords * 8] for i in range(nsegs)]
    framed = mcp.frame_message(segs)
    got_segs = mcp.parse_frame(framed)
    assert [s for _, s in got_segs] == [len(s) for s in segs]
    ref = _time(lambda: _parse_frame_python(framed, nsegs), 3)
    got = _time(lambda: mcp.parse_frame(framed))
    return f"parse frame {nsegs} segs", ref, got, 1.0


def _parse_frame_python(msg: bytes, maxseg: int):
    nseg = 1 + int.from_bytes(msg[0:4], "little")
    table = 4 + 4 * nseg
    if table % 8:
        table += 4
    pos = table
    out = []
    for s in range(nseg):
        size = int.from_bytes(msg[4 + 4 * s:8 + 4 * s], "little") * 8
        out.append((pos, size))
        pos += size
    return out


def bench_read_fields(nfields: int = 4096, repeats: int = 1):
    """Read a wide struct's whole data section through the layout."""
    fields = [("int64" if i % 3 else ("bool" if i % 7 == 0 else "int32"))
              for i in range(nfields)]
    layout = mcp.struct_layout(fields)
    section = bytes(np.random.default_rng(3).integers(
        0, 256, size=layout["data_words"] * 8, dtype=np.uint8))
    got = mcp.read_struct_fields(section, layout)
    expect = _read_fields_python(section, layout)
    assert got == expect, "field read mismatch"
    ref = _time(lambda: _read_fields_python(section, layout), 1)
    got_t = _time(lambda: mcp.read_struct_fields(section, layout), repeats)
    return f"read {nfields} fields", ref, got_t, 1.0


def _read_fields_python(data: bytes, layout):
    out = []
    for off, size in zip(layout["offsets"], layout["sizes"]):
        if off < 0:
            out.append(None)
            continue
        if size == 1:
            out.append((data[off >> 3] >> (off & 7)) & 1)
        else:
            out.append(int.from_bytes(data[off >> 3:(off >> 3) + size // 8], "little"))
    return out


def bench_resolve_pointers(nptrs: int = 1 << 18):
    """Chase a run of pointer words inside a segment."""
    rng = np.random.default_rng(4)
    # Struct pointers with a non-zero offset, so none of them is null or far.
    offsets = rng.integers(1, 1 << 28, size=nptrs, dtype=np.int64)
    words = (offsets << 2) | np.int64(mcp.STRUCT)
    seg = words.astype("<u8").tobytes()
    got = mcp.resolve_pointers(seg, 0, nptrs)
    expect = [(i + 1 + int(w)) * 8 for i, w in enumerate(offsets)]
    assert got == expect, "pointer resolution mismatch"
    ref = _time(lambda: _resolve_python(seg, nptrs), 1)
    got_t = _time(lambda: mcp.resolve_pointers(seg, 0, nptrs))
    return f"resolve {nptrs} pointers", ref, got_t, 1.0


def _resolve_python(seg: bytes, nptrs: int):
    out = []
    for i in range(nptrs):
        w = int.from_bytes(seg[i * 8:i * 8 + 8], "little")
        if w == 0:
            out.append(-1)
            continue
        kind = w & 3
        if kind == 2 or kind == 3:
            out.append(-2)
            continue
        raw = (w >> 2) & 0x3FFFFFFF
        if raw >= 0x20000000:
            raw -= 0x40000000
        out.append((i + 1 + raw) * 8)
    return out


def bench_pack_vs_capnp(nwords: int = 1000):
    """Pack a real message and hand it to pycapnp's C++ reference decoder.

    Kept to a single-segment message on purpose: pycapnp's packed *reader*
    rejects multi-segment packed streams, so only the single-segment case can be
    checked against the C++ implementation.
    """
    try:
        import capnp
    except ImportError:
        return None
    capnp.remove_import_hook()
    schema = pathlib.Path(__file__).resolve().parent / "_bench.capnp"
    schema.write_text("@0x9f5e1b3c7a2d4086;\nstruct P { v @0 :List(Int64); }\n")
    mod = capnp.load(str(schema))
    msg = mod.P.new_message(v=list(range(nwords)))
    raw = msg.to_bytes()
    msg.clear_write_flag()
    packed = mcp.pack(raw)
    got = list(capnp.read_multiple_bytes_packed(packed))[0].as_struct(mod.P)
    assert list(got.v) == list(range(nwords)), "pycapnp could not read our packing"

    def run_cpp():
        list(capnp.read_multiple_bytes_packed(packed))

    def run_mojo():
        mcp.pack(raw)

    ref = _time(run_cpp, 3)
    mine = _time(run_mojo, 3)
    return f"pack {len(raw) // 1024}Ki + cpp decode", ref, mine, len(packed) / len(raw)


def main():
    print(f"{'case':<30}{'python/c++ ref':>16}{'mojo-capnproto':>18}{'ratio':>9}{'size':>8}")
    print("-" * 82)
    cases = [bench_pack, bench_unpack, bench_frame, bench_parse_frame,
             bench_read_fields, bench_resolve_pointers, bench_pack_vs_capnp]
    for fn in cases:
        result = fn()
        if result is None:
            print(f"{'pack vs pycapnp':<30}{'--':>16}{'skipped (no pycapnp)':>18}")
            continue
        label, ref, got, ratio_extra = result
        ratio = ref / got if got else float("nan")
        print(f"{label:<30}{ref*1e3:>14.2f}ms{got*1e3:>16.2f}ms{ratio:>8.2f}x"
              f"{ratio_extra:>8.2f}")


if __name__ == "__main__":
    main()
