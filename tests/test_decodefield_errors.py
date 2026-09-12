"""Tests that a failed decode thread is not mistaken for end-of-input.

decodefield() pre-populates its result dict with field/offset None sentinels
before doing any work. Those sentinels are also exactly what a genuine EOF
returns, so when decodefield() runs on a worker thread (the default path) and
raises, readfield() used to read the untouched sentinels back as EOF: the decode
stopped early but exited 0 and wrote a .tbc.json whose field count described the
truncated output as complete.

decodefield() now records the exception on the shared dict and readfield()
re-raises it on the main thread, while a real EOF still reads as EOF.

This is not specific to any exception type -- the sentinels are pre-populated
before any code that can raise, so anything a field's process() throws is
swallowed the same way. The tests therefore use a meaningless exception class,
with the ZeroDivisionError from the original report kept as a single named case.

These drive the two functions against stub objects: constructing a real LDdecode
requires an RF source and a populated DemodCache, which is far more scaffolding
than the None-sentinel/exception contract under test needs.
"""

import types

import pytest

from lddecode.core import LDdecode


class _DecodeBoom(Exception):
    """An arbitrary decode-time failure.

    Deliberately not a builtin: the sentinel/EOF collision swallows *any*
    exception raised on the decode thread, so the tests use an exception with no
    meaning of its own. The ZeroDivisionError that surfaced this in the field is
    covered separately below, as one instance of the general case rather than the
    thing being fixed.
    """


class _BoomField:
    """Stands in for FieldClass; blows up in process() like a real bad field."""

    def __init__(self, *args, **kwargs):
        self.valid = False

    def process(self):
        raise _DecodeBoom("field failed to process")


def _decodefield_stub(rawdecode, field_class=_BoomField):
    """Minimal `self` carrying only what decodefield() touches before process()."""
    return types.SimpleNamespace(
        rf=types.SimpleNamespace(blockcut=0),
        blocksize=32768,
        readlen=65536,
        demodcache=types.SimpleNamespace(read=lambda *a, **kw: rawdecode),
        FieldClass=field_class,
        fields_written=0,
        wow_level_adjust_smoothing=None,
        wow_interpolation_method=None,
        curfield=None,
        use_profiler=False,
        system="NTSC",
    )


def _readfield_stub(threadreturn):
    """Minimal `self` carrying only what readfield() touches before the raise."""
    return types.SimpleNamespace(
        fieldstack=[],
        second_decode=None,
        fields_written=0,
        decodethread=None,
        threadreturn=threadreturn,
        fdoffset=0,
        mtf_level=0,
        numthreads=0,
        decodefield=lambda *a, **kw: (None, None),
    )


def test_decodefield_records_exception_on_rv_and_still_raises():
    """A field that fails to process leaves the exception on the shared dict."""
    rv = {}
    stub = _decodefield_stub({"startloc": 0})

    with pytest.raises(_DecodeBoom):
        LDdecode.decodefield(stub, 0, 0, rv=rv)

    assert isinstance(rv["exception"], _DecodeBoom)
    # The sentinels the caller would otherwise read as EOF are still in place --
    # the exception is what distinguishes this dict from a real end-of-input.
    assert rv["field"] is None
    assert rv["offset"] is None


def test_decodefield_leaves_exception_none_at_eof():
    """A real EOF (no rawdecode) returns the bare sentinels and no exception."""
    rv = {}
    stub = _decodefield_stub(None)

    assert LDdecode.decodefield(stub, 0, 0, rv=rv) == (None, None)
    assert rv["exception"] is None


def test_readfield_reraises_a_dead_decode_thread():
    """The failure surfaces on the main thread instead of reading as EOF."""
    boom = _DecodeBoom("field failed to process")
    stub = _readfield_stub({"field": None, "offset": None, "exception": boom})

    with pytest.raises(_DecodeBoom) as excinfo:
        LDdecode.readfield(stub)

    # The original exception object, so its traceback points at the real cause.
    assert excinfo.value is boom


def test_readfield_reraises_the_reported_zerodivisionerror():
    """The specific failure this was found through, as one case of the general one.

    A ZeroDivisionError out of a field's process() is what truncated a real
    capture at 43% while exiting 0. It gets its own test only so the report has a
    regression test that names it -- nothing here is specific to the exception
    type, which is the whole point of the fix.
    """
    boom = ZeroDivisionError("float division by zero")
    stub = _readfield_stub({"field": None, "offset": None, "exception": boom})

    with pytest.raises(ZeroDivisionError) as excinfo:
        LDdecode.readfield(stub)

    assert excinfo.value is boom


def test_readfield_still_reports_eof_as_eof():
    """A clean (None, None) with no exception must keep meaning end-of-input."""
    stub = _readfield_stub({"field": None, "offset": None, "exception": None})

    assert LDdecode.readfield(stub) is None
