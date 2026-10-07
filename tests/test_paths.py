"""Tests for sprite/state path resolution and version helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from jaxatari import paths


def test_prefers_canonical_when_present(tmp_path, monkeypatch):
    canonical = tmp_path / "jaxtari"
    legacy = tmp_path / "jaxatari"
    (canonical / "sprites" / "pong").mkdir(parents=True)
    (canonical / "sprites" / "pong" / "x.npy").write_bytes(b"x")
    (legacy / "sprites" / "pong").mkdir(parents=True)
    (legacy / "sprites" / "pong" / "x.npy").write_bytes(b"y")

    monkeypatch.setattr(paths, "canonical_storage_dir", lambda: canonical)
    monkeypatch.setattr(paths, "legacy_storage_dir", lambda: legacy)

    assert paths.get_storage_dir() == canonical
    assert paths.get_base_sprite_dir() == str(canonical / "sprites")


def test_falls_back_to_legacy(tmp_path, monkeypatch):
    canonical = tmp_path / "jaxtari"
    legacy = tmp_path / "jaxatari"
    (legacy / "sprites" / "pong").mkdir(parents=True)
    (legacy / "sprites" / "pong" / "x.npy").write_bytes(b"y")
    (legacy / paths.OWNERSHIP_MARKER_NAME).write_text("", encoding="utf-8")

    monkeypatch.setattr(paths, "canonical_storage_dir", lambda: canonical)
    monkeypatch.setattr(paths, "legacy_storage_dir", lambda: legacy)

    assert paths.get_storage_dir() == legacy
    assert paths.ownership_or_alt_markers_present()


def test_new_install_targets_canonical(tmp_path, monkeypatch):
    canonical = tmp_path / "jaxtari"
    legacy = tmp_path / "jaxatari"
    monkeypatch.setattr(paths, "canonical_storage_dir", lambda: canonical)
    monkeypatch.setattr(paths, "legacy_storage_dir", lambda: legacy)
    assert paths.get_storage_dir() == canonical


def test_version_read_write_and_update_needed(tmp_path, monkeypatch):
    storage = tmp_path / "jaxtari"
    (storage / "sprites" / "pong").mkdir(parents=True)
    (storage / "sprites" / "pong" / "x.npy").write_bytes(b"x")
    monkeypatch.setattr(paths, "get_storage_dir", lambda: storage)
    monkeypatch.setattr(paths, "REQUIRED_SPRITE_VERSION", 2)

    assert paths.read_installed_sprite_version(storage) is None
    assert paths.sprite_update_needed(storage)

    paths.write_sprite_version(storage, 1)
    assert paths.read_installed_sprite_version(storage) == 1
    assert paths.sprite_update_needed(storage)

    paths.write_sprite_version(storage, 2)
    assert not paths.sprite_update_needed(storage)


def test_declined_update_marker(tmp_path, monkeypatch):
    storage = tmp_path / "jaxtari"
    storage.mkdir()
    monkeypatch.setattr(paths, "get_storage_dir", lambda: storage)
    monkeypatch.setattr(paths, "REQUIRED_SPRITE_VERSION", 1)

    assert not paths.declined_update_for_required_version(storage)
    paths.record_declined_update(storage)
    assert paths.declined_update_for_required_version(storage)

    monkeypatch.setattr(paths, "REQUIRED_SPRITE_VERSION", 2)
    assert not paths.declined_update_for_required_version(storage)
