# mojo-capnproto

`mojo-capnproto` is the byte-level core of the [Cap'n Proto](https://capnproto.org)
wire format, written in Mojo and callable from Python.

Cap'n Proto is a binary serialisation format, so there is no floating-point
arithmetic in it. The compute that exists is integer and bitwise: the struct
layout rule, the packing codec, the 64-bit pointer words, the stream framing,
and the pointer chase that reads a struct back out. All of that is here, and all
of it is **bit-exact** against `capnp` rather than merely close.

This is not a Cap'n Proto implementation. It does not parse `.capnp` schemas,
generate classes, build messages from typed fields, do RPC, canonicalise, or
enforce a traversal limit. Use the real `capnp` package for all of that. What
this gives you is the arithmetic underneath, so a hand-rolled reader, a bulk
codec or a verification pass can run it compiled.

```python
import mojo_capnproto as mcp

mcp.struct_layout(["int32", "int64", "text", "bool"])
# {'offsets': [0, 64, -1, 32], 'sizes': [32, 64, -1, 1], 'data_words': 2}

mcp.unpack_struct_pointer(mcp.pack_struct_pointer(3, 2, 1))   # (3, 2, 1)
mcp.pack(bytes(24))                                          # b'\x00\x02'
mcp.parse_frame(framed_message)                              # [(16, 816)]
```

## Covered subset

| area | implemented API |
| --- | --- |
| Struct layout | `struct_layout` (bit offsets, widths, data-section size in words) |
| Field access | `read_struct_fields` (reads a whole data section by layout) |
| Packing | `pack`, `unpack` (tag byte, `0x00` zero runs, `0xff` verbatim runs) |
| Pointer words | `pack_struct_pointer` / `unpack_struct_pointer`, `pack_list_pointer` / `unpack_list_pointer`, `pack_far_pointer`, `pack_double_far_pointer`, `pack_capability_pointer`, `word_kind` |
| Framing | `frame_message`, `parse_frame` |
| Pointer chase | `resolve_pointers` |
| Constants | `STRUCT`/`LIST`/`FAR`/`OTHER`, the element-size codes, `ELEM_WIDTH`, `PRIMITIVE_BITS` |

## Not implemented

Everything schema-driven, and everything that is policy rather than arithmetic:

- `.capnp` schema parsing, code generation, and the generated `_capnp` types
- building or reading typed messages, defaults, unions, groups, orphans,
  generics, and the compile-time evolution rules
- list content accessors: a composite (struct) list's tag word is packed and
  unpacked, but the elements are not decoded into records
- the canonicalization rules, LZ4/zlib compression, RPC, and the capability
  table (a capability *pointer* is packed; the table behind it is not)
- pointer validation, the traversal limit, and the nesting limit
- `AnyPointer`, generics, and `interface(Capability)` dispatch

## Install

```bash
pixi install
pixi run build      # -> dist/libmojo-capnproto.so
pixi run test
```

Set `PYTHONPATH=python` when using the package outside a Pixi task. The Python
package is `mojo_capnproto`, so it imports alongside the real `capnp`.

## Tests

```
55 passed
```

Two files:

- `tests/test_wire_format.py` reproduces the encoding spec's own worked
  examples byte for byte — the struct and list pointer words, the three packing
  examples, the zero-run and verbatim-run markers — and needs no reference
  implementation.
- `tests/test_capnp_parity.py` is the real parity suite. It loads a small
  schema through `capnp.load`, builds messages with pycapnp, and checks that
  `frame_message` is **byte identical** to `MessageReader.to_bytes()`, that the
  layout produces the same data-section size the compiler emitted, that every
  field reads back as the value that was written, and that the pointer chase
  finds the text, data, list and composite-list objects at the right byte
  offsets. The packing tests hand our bytes to pycapnp's C++ reference decoder
  (`capnp.read_multiple_bytes_packed`) and check the fields come back.

Because pycapnp is not in the shared test venv, the parity file is skipped when
it is missing. To run it, extract the wheel into `.refdeps/` (git-ignored):

```bash
python -m pip download --no-deps -d /tmp/capnp pycapnp
python -c "import zipfile,glob; zipfile.ZipFile(glob.glob('/tmp/capnp/*.whl')[0]).extractall('.refdeps')"
```

## Performance

Best-of-N wall clock in one process, correctness gated first. Baselines are the
strongest fair ones: pure-Python bit-twiddling for the packer and the layout
reader, and pycapnp's C++ reference *decoder* for the packing round trip.

| case | reference | mojo-capnproto | result |
| --- | ---: | ---: | ---: |
| pack 1 MiB, 50% zero words | 3066.82 ms | 31.74 ms | 96.6x faster |
| unpack 1 MiB | 2306.18 ms | 80.25 ms | 28.7x faster |
| frame 64 x 4096 words (2 MiB) | 0.60 ms | 2.88 ms | 3.4x slower |
| parse 64-segment header | 0.04 ms | 0.36 ms | 8.2x slower |
| read 4096 fields | 4.48 ms | 3.96 ms | 1.13x faster |
| resolve 262144 pointers | 460.68 ms | 168.85 ms | 2.7x faster |
| pack 7 KiB + C++ decode | 0.05 ms | 0.05 ms | 1.11x faster |

The two losses are real and worth being precise about. **Framing** and **header
parsing** are dominated by fixed per-call cost, not by arithmetic: the Mojo path
pays a ctypes boundary, two buffer allocations and a bulk copy, while the Python
baseline for framing is `b"".join()`, which is a single `memcpy` over segments
that are already contiguous. At 64 segments the header is 260 bytes and the
payload 2 MiB, so the kernel is essentially a copy engine and a copy engine is
not what Mojo wins at. At one segment and a small payload the two are level
(bench case seven). The same fixed cost is why parsing a 260-byte header takes
0.36 ms here: the parse itself is a few dozen integer operations.

The packing wins are the large ones because packing is a genuine per-word loop
with a branch on the tag byte, and the Python baseline pays an interpreter
dispatch for every byte.

`read 4096 fields` at 1.13x is close to parity, which is the honest answer for a
loop that is a single load and a shift per field.

Reproduce with `pixi run bench`.

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit, because shared
library build cost is largely fixed. `build/build.sh` compiles it with
`mojo build --emit shared-lib` into `dist/libmojo-capnproto.so`.

The Python layer in `python/mojo_capnproto` owns every array. Buffers cross the
C ABI as 64-bit addresses and are rebuilt in Mojo as
`Pointer[T, AnyOrigin[mut=True]]`, which keeps the exported symbols
non-parametric. A few details that are easy to get wrong and are pinned by
tests:

- The pointer **kind lives in the two least significant bits**, so the 30-bit
  offset field starts at bit 2, not bit 0. `pack_struct_pointer(2, 3, 2)`
  reproduces the spec's own word `08 00 00 00 03 00 02 00`.
- The framing header word is the segment count **minus one**, with no shift, and
  the per-segment sizes are in **words**, not bytes.
- An all-zero word is null. A *list* pointer with offset zero is a perfectly
  ordinary pointer at the word right after itself, so the null test has to be on
  the whole word.
- A `0xff` packing tag means both "all eight bytes follow" and "verbatim run
  follows", so the decoder must not read the word twice.
- In a composite (struct) list, the tag word has a struct pointer's shape but
  its offset field counts elements, and each element is laid out in place as
  its own data words followed by its own pointer words.

There is no FMA in this port — every operation is integer or bitwise — so exact
equality is the right assertion and the tests use it. The only floating-point
anywhere is in the benchmark's data generation.

## License

MIT
