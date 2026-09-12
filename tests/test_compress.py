"""Tests for lddecode.compress, the .lds <-> .ldf compressor.

The round trip tests need a flac 1.5.0 or later to encode with, the same one
ld-compress itself requires, and are skipped when there is not one to hand.
"""

import hashlib
import io
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from lddecode import compress, lds


def _have_usable_flac():
    path = shutil.which("flac")
    if path is None:
        return False
    try:
        out = subprocess.run(
            [path, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        ).stdout
    except OSError:
        return False

    fields = out.split()
    if len(fields) < 2:
        return False
    version = compress._parse_version(fields[1])
    return version is not None and version >= compress.FLAC_MIN_VERSION


needs_flac = pytest.mark.skipif(
    not _have_usable_flac(), reason="flac 1.5.0 or later is not installed"
)


@pytest.fixture
def sample_lds(tmp_path):
    """A .lds file holding a second and a bit of pseudo-random 10-bit samples."""
    rng = np.random.RandomState(1234)
    tenbit = rng.randint(0, 1024, size=40000 * 12).astype(np.int32)
    samples = ((tenbit - 512) * 64).astype(np.int16)

    path = tmp_path / "sample.lds"
    with lds.LdsWriter(str(path)) as writer:
        writer.write(samples)
    return path


# ---------------------------------------------------------------------------
# Version parsing - this is what keeps a flac too old for -j out
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1.5.0", (1, 5)),
        ("1.5.1", (1, 5)),
        ("1.4.3", (1, 4)),
        ("2.0.0", (2, 0)),
        ("1.10.0", (1, 10)),
        ("1.5.0-git", (1, 5)),
        ("", None),
        ("1", None),
        ("abc", None),
        ("1.x", None),
    ],
)
def test_parse_version(text, expected):
    assert compress._parse_version(text) == expected


def test_version_ordering_rejects_pre_1_5():
    assert compress._parse_version("1.4.3") < compress.FLAC_MIN_VERSION
    assert compress._parse_version("1.10.0") > compress.FLAC_MIN_VERSION
    assert compress._parse_version("1.5.0") >= compress.FLAC_MIN_VERSION


# ---------------------------------------------------------------------------
# Output naming.  Output always lands in the current directory, whatever
# directory the input came from.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path,suffixes,expected",
    [
        ("sample.lds", (".lds",), "sample"),
        (os.path.join("some", "dir", "sample.lds"), (".lds",), "sample"),
        ("sample.ldf", compress.COMPRESSED_SUFFIXES, "sample"),
        ("sample.raw.oga", compress.COMPRESSED_SUFFIXES, "sample"),
        # .raw.oga has to be tried before .ldf, or the name keeps a stray .raw
        ("disc.raw.oga", (".raw.oga", ".ldf"), "disc"),
        ("no-suffix", (".lds",), "no-suffix"),
    ],
)
def test_strip_suffix(path, suffixes, expected):
    assert compress._strip_suffix(path, suffixes) == expected


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


@needs_flac
def test_round_trip_is_lossless(tmp_path, sample_lds, monkeypatch):
    monkeypatch.chdir(tmp_path)
    flac = compress.find_flac()

    assert compress.compress(str(sample_lds), flac, 8, "ldf", False)
    ldf = tmp_path / "sample.ldf"
    assert ldf.exists()

    # Uncompress somewhere else, so the .lds being checked is a new file
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.chdir(out_dir)
    assert compress.uncompress(str(ldf), False)

    assert (out_dir / "sample.lds").read_bytes() == sample_lds.read_bytes()


@needs_flac
def test_oga_extension(tmp_path, sample_lds, monkeypatch):
    monkeypatch.chdir(tmp_path)
    flac = compress.find_flac()

    assert compress.compress(str(sample_lds), flac, 1, "raw.oga", False)
    assert (tmp_path / "sample.raw.oga").exists()


@needs_flac
def test_verify_reports_both_checksums(tmp_path, sample_lds, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    flac = compress.find_flac()
    assert compress.compress(str(sample_lds), flac, 8, "ldf", False)

    ldf = tmp_path / "sample.ldf"
    assert compress.verify(str(ldf), False)

    printed = capsys.readouterr().out.split("\n")
    digests = [line.split()[0] for line in printed if line.strip()]

    assert digests[0] == hashlib.md5(ldf.read_bytes()).hexdigest()
    assert digests[1] == hashlib.md5(sample_lds.read_bytes()).hexdigest()


@needs_flac
def test_failed_compression_leaves_nothing_behind(tmp_path, monkeypatch):
    """A truncated .ldf must never be left looking like a good capture."""
    monkeypatch.chdir(tmp_path)
    missing = tmp_path / "missing.lds"

    assert not compress.compress(str(missing), compress.find_flac(), 8, "ldf", False)
    assert sorted(p.name for p in tmp_path.iterdir()) == []


@needs_flac
def test_failed_compression_keeps_a_preexisting_output(tmp_path, monkeypatch):
    """flac -f truncates its output as it starts, so compression has to go via
    a temporary file or a failure destroys a .ldf that was already there."""
    monkeypatch.chdir(tmp_path)
    existing = tmp_path / "missing.ldf"
    existing.write_bytes(b"not ours to touch")

    assert not compress.compress(
        str(tmp_path / "missing.lds"), compress.find_flac(), 8, "ldf", False
    )
    assert existing.read_bytes() == b"not ours to touch"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["missing.ldf"]


@needs_flac
def test_compression_over_a_preexisting_output_succeeds(tmp_path, sample_lds, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sample.ldf").write_bytes(b"stale")

    assert compress.compress(str(sample_lds), compress.find_flac(), 8, "ldf", False)
    assert (tmp_path / "sample.ldf").read_bytes()[:4] == b"OggS"
    assert not (tmp_path / "sample.ldf.part").exists()


def test_failed_uncompression_leaves_nothing_behind(tmp_path, monkeypatch):
    broken = tmp_path / "broken.ldf"
    broken.write_bytes(b"\x00" * 4096)

    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)

    assert not compress.uncompress(str(broken), False)
    assert sorted(p.name for p in work.iterdir()) == []


def test_failed_uncompression_keeps_a_preexisting_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    broken = tmp_path / "broken.ldf"
    broken.write_bytes(b"\x00" * 4096)
    existing = tmp_path / "broken.lds"
    existing.write_bytes(b"not ours to touch")

    assert not compress.uncompress(str(broken), False)
    assert existing.read_bytes() == b"not ours to touch"


# ---------------------------------------------------------------------------
# Locating flac
# ---------------------------------------------------------------------------


def _fake_flac(directory, version, name="flac"):
    """Write a stub flac that reports the given version and does nothing else."""
    directory.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        path = directory / (name + ".bat")
        path.write_text("@echo off\r\necho flac %s\r\n" % version)
    else:
        path = directory / name
        path.write_text('#!/bin/sh\necho "flac %s"\n' % version)
        path.chmod(0o755)
    return path


def test_flac_beside_the_command_wins_over_path(tmp_path, monkeypatch):
    """The packages ship flac next to ld-compress, in a directory that is not
    necessarily on PATH, and it has to be preferred to any other copy."""
    bundled = tmp_path / "bundle"
    elsewhere = tmp_path / "elsewhere"
    _fake_flac(bundled, "1.5.0")
    _fake_flac(elsewhere, "1.5.0")

    monkeypatch.setattr(sys, "argv", [str(bundled / "ld-compress")])
    monkeypatch.setenv("PATH", str(elsewhere))

    assert os.path.dirname(compress.find_flac()) == str(bundled)


def test_missing_flac_exits(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(tmp_path / "ld-compress")])
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))
    monkeypatch.setenv("PATH", str(tmp_path / "nothing-here"))

    with pytest.raises(SystemExit) as exit_info:
        compress.find_flac()
    assert exit_info.value.code == 1


def test_too_old_flac_exits(tmp_path, monkeypatch):
    bundled = tmp_path / "bundle"
    _fake_flac(bundled, "1.4.3")

    monkeypatch.setattr(sys, "argv", [str(bundled / "ld-compress")])
    monkeypatch.setattr(sys, "executable", str(bundled / "python"))
    monkeypatch.setenv("PATH", str(tmp_path / "nothing-here"))

    with pytest.raises(SystemExit) as exit_info:
        compress.find_flac()
    assert exit_info.value.code == 1


# ---------------------------------------------------------------------------
# Progress display
# ---------------------------------------------------------------------------


def test_progress_file_reads_transparently(tmp_path):
    data = os.urandom(5000)
    path = tmp_path / "data.bin"
    path.write_bytes(data)

    with compress.ProgressFile(str(path), show_progress=False) as f:
        assert f.read() == data


def test_progress_file_draws_without_a_terminal(tmp_path, capsys):
    """The bar is measured against stderr, which is not a terminal under
    pytest; drawing must still work rather than raise."""
    path = tmp_path / "data.bin"
    path.write_bytes(os.urandom(5000))

    with compress.ProgressFile(str(path), show_progress=True) as f:
        while f.read(1024):
            pass

    assert "data.bin" in capsys.readouterr().err


def test_hash_sink_matches_a_real_file(tmp_path):
    samples = np.arange(-2048, 2048, dtype=np.int16)

    digest = hashlib.md5()
    with lds.LdsWriter(compress._HashSink(digest)) as writer:
        writer.write(samples)

    path = tmp_path / "reference.lds"
    with lds.LdsWriter(str(path)) as writer:
        writer.write(samples)

    assert digest.hexdigest() == hashlib.md5(path.read_bytes()).hexdigest()
