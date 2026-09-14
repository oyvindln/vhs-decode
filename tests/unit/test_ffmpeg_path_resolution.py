from __future__ import annotations

# import sys
# import types
#
# import lddecode.utils as utils
#
#
# def test_ensure_ffmpeg_on_path_uses_static_ffmpeg_add_paths(monkeypatch):
#     calls: list[bool] = []
#
#     def _add_paths(*, weak: bool = False) -> None:
#         calls.append(weak)
#
#     fake_static_ffmpeg = types.SimpleNamespace(add_paths=_add_paths)
#     monkeypatch.setitem(sys.modules, "static_ffmpeg", fake_static_ffmpeg)
#
#     assert utils._ensure_ffmpeg_on_path() is True
#     assert calls == [True]
#
#
# def test_ensure_ffmpeg_on_path_returns_false_when_add_paths_fails(monkeypatch):
#     def _add_paths(*, weak: bool = False) -> None:
#         raise RuntimeError("boom")
#
#     fake_static_ffmpeg = types.SimpleNamespace(add_paths=_add_paths)
#     monkeypatch.setitem(sys.modules, "static_ffmpeg", fake_static_ffmpeg)
#
#     assert utils._ensure_ffmpeg_on_path() is False
