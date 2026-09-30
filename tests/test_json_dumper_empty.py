"""Tests that a decode producing no fields still shuts down cleanly.

build_json() returns None until a valid field has been decoded. JSONDumper
handed that None straight to its writer thread, which died on None.items() --
printing an AttributeError traceback over the "Completed without handling any
frames" message that explains what actually went wrong, and leaving a
truncated .tbc.json.tmp behind.

Reaching this needs only a decode that yields nothing; see test_s8_loader.py
for the .s8 loader gap that was one way to get there.
"""

import types

from lddecode.utils import JSONDumper


class _EmptyDecode:
    """Stands in for LDdecode after a decode that produced no valid field.

    Only the three attributes JSONDumper touches are provided; building a real
    LDdecode needs an RF source and a populated DemodCache, none of which the
    None-handling under test depends on.
    """

    verboseVITS = False

    def __init__(self):
        self.field_info_reads = 0

    def build_json(self):
        return None

    def _read_field_info(self):
        self.field_info_reads += 1
        return []

    @property
    def fieldinfo(self):
        return types.SimpleNamespace(read=self._read_field_info)


def test_empty_decode_writes_no_json_and_leaves_the_thread_alive(tmp_path):
    """close() must survive build_json() returning None, and write no files."""
    outname = str(tmp_path / "out")
    ldd = _EmptyDecode()

    dumper = JSONDumper(ldd, outname)
    dumper.write()
    dumper.close()

    assert not dumper._dumper.is_alive()
    # Neither the final file nor the partial one the writer opens first.
    assert list(tmp_path.iterdir()) == []


def test_empty_decode_does_not_drain_field_info(tmp_path):
    """Skipping the write must not consume field info that was never written.

    fieldinfo.read() empties the unsent list, so calling it and then dropping
    the result would silently lose fields if build_json() ever returned None
    with fields pending.
    """
    ldd = _EmptyDecode()

    dumper = JSONDumper(ldd, str(tmp_path / "out"))
    dumper.write()
    dumper.close()

    assert ldd.field_info_reads == 0
