import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))

_LIB = _ROOT / "dist" / "libmojo-capnproto.so"

if not _LIB.exists():
    pytest.skip(
        "libmojo-capnproto.so not built; run `bash build/build.sh`",
        allow_module_level=True,
    )

# pycapnp is not in the shared test venv, so the parity tests look for an
# extracted wheel under .refdeps/ first:
#
#   python -m pip download --no-deps -d /tmp/capnp pycapnp
#   python -c "import zipfile,glob; zipfile.ZipFile(glob.glob('/tmp/capnp/*.whl')[0]).extractall('.refdeps')"
#
# When it is missing, the byte-exact parity tests are skipped and the
# spec-conformance and round-trip tests still run.
_REFDEPS = _ROOT / ".refdeps"
if (_REFDEPS / "capnp").is_dir():
    sys.path.append(str(_REFDEPS))

try:
    import capnp as _capnp

    HAS_CAPNP = True
except ImportError:  # pragma: no cover - depends on the environment
    _capnp = None
    HAS_CAPNP = False

requires_capnp = pytest.mark.skipif(
    not HAS_CAPNP, reason="pycapnp not available; see tests/conftest.py"
)

SCHEMA = """
@0x9f5e1b3c7a2d4086;

struct Point {
  x @0 :Int32;
  y @1 :Int64;
  name @2 :Text;
  flags @3 :Data;
  vals @4 :List(Int32);
}

struct Scene {
  origin @0 :Point;
  pts @1 :List(Point);
  label @2 :Text;
  count @3 :UInt8;
  big @4 :Int64;
  tiny @5 :Bool;
}

struct Wide {
  a @0 :Int8;
  b @1 :Int64;
  c @2 :Int16;
  d @3 :Bool;
  e @4 :Float64;
  f @5 :Int32;
  g @6 :Bool;
}
"""


@pytest.fixture(scope="session")
def schema(tmp_path_factory):
    """A loaded pycapnp module for the schemas used by the parity tests."""
    if not HAS_CAPNP:
        pytest.skip("pycapnp not available")
    _capnp.remove_import_hook()
    path = tmp_path_factory.mktemp("schema") / "t.capnp"
    path.write_text(SCHEMA)
    return _capnp.load(str(path))


def segments_of(message) -> list:
    """The raw (unframed) segment byte strings of a pycapnp message."""
    out = []
    for seg in message.to_segments():
        if isinstance(seg, bytes):
            out.append(seg)
        else:
            out.append(b"".join(bytes(w) for w in seg))
    return out
