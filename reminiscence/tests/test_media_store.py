"""Sprint 2 tests: content-addressed MediaStore + media asset references."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from reminiscence.storage.database import Database
from reminiscence.storage.media_store import (
    MediaIntegrityError,
    MediaStore,
    UnsafePathError,
    sha256_of_file,
)


@pytest.fixture()
def store(tmp_path: Path) -> MediaStore:
    return MediaStore(tmp_path / "media")


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


# ---------------------------------------------------------------- identity


def test_same_bytes_different_paths_one_object(store: MediaStore, tmp_path: Path):
    """Acceptance: identical bytes from different paths produce one object."""
    a = _write(tmp_path / "photos" / "a.jpg", b"IMAGEBYTES" * 100)
    b = _write(tmp_path / "backup" / "copy-of-a.jpg", b"IMAGEBYTES" * 100)
    obj_a = store.put(a)
    obj_b = store.put(b)
    assert obj_a.key == obj_b.key
    assert obj_a.sha256 == obj_b.sha256
    # exactly one canonical file on disk
    stats = store.stats()
    assert stats["objects"]["count"] == 1


def test_content_hash_matches_streamed_sha256(store: MediaStore, tmp_path: Path):
    f = _write(tmp_path / "doc.pdf", b"%PDF-1.4 fake pdf payload")
    obj = store.put(f)
    import hashlib

    assert obj.sha256 == hashlib.sha256(b"%PDF-1.4 fake pdf payload").hexdigest()
    assert sha256_of_file(obj.local_path if False else store.local_path(obj.key)) == obj.sha256


def test_get_roundtrip_and_exists(store: MediaStore, tmp_path: Path):
    f = _write(tmp_path / "note.txt", b"hello memory")
    obj = store.put(f)
    assert store.exists(obj.key)
    assert store.exists_hash(obj.sha256)
    assert store.get(obj.key) == b"hello memory"
    with store.open(obj.key) as fh:
        assert fh.read() == b"hello memory"


# ---------------------------------------------------------------- security


def test_path_traversal_rejected(store: MediaStore):
    for bad in ("../outside", "objects/../../etc/passwd", "/absolute/key", "..", "a/./b/../c"):
        with pytest.raises(UnsafePathError):
            store.get(bad)
    assert store.exists("../outside") is False


def test_null_byte_rejected(store: MediaStore):
    with pytest.raises(UnsafePathError):
        store.get("objects/ab/ab\x00cd")


def test_atomic_write_leaves_no_temp_on_success(store: MediaStore, tmp_path: Path):
    f = _write(tmp_path / "x.bin", os.urandom(4096))
    obj = store.put(f)
    parent = store.local_path(obj.key).parent
    temps = [p for p in parent.iterdir() if p.name.startswith(".tmp-")]
    assert temps == []


# ---------------------------------------------------------------- integrity


def test_verify_detects_corruption(store: MediaStore, tmp_path: Path):
    f = _write(tmp_path / "img.png", b"\x89PNG fake image bytes")
    obj = store.put(f)
    assert store.verify(obj.key) is True
    # corrupt the stored object in place
    store.local_path(obj.key).write_bytes(b"\x89PNG TAMPERED BYTES")
    with pytest.raises(MediaIntegrityError):
        store.verify(obj.key)
    results = list(store.verify_all())
    assert any(ok is False for _, ok, _ in results)


# ---------------------------------------------------------------- deletion


def test_delete_explicit_and_secure(store: MediaStore, tmp_path: Path):
    f = _write(tmp_path / "secret.txt", b"private journal entry")
    obj = store.put(f)
    assert store.delete(obj.key, secure=True) is True
    assert not store.exists(obj.key)
    assert store.delete(obj.key) is False  # idempotent second delete


# ---------------------------------------------------------------- derived


def test_thumbnails_and_derivatives_namespaced(store: MediaStore, tmp_path: Path):
    f = _write(tmp_path / "clip.mp4", b"fake video bytes")
    obj = store.put(f)
    thumb = store.create_thumbnail(obj.sha256, b"tinyjpegbytes")
    kf = store.create_derivative(obj.sha256, b"frame-bytes", "keyframe-3", extension=".jpg")
    assert thumb.key.startswith("thumbnails/")
    assert obj.sha256 in kf.key  # namespaced under source hash
    assert store.thumbnails_for(obj.sha256) == [thumb.key]
    assert store.get(kf.key) == b"frame-bytes"


# ---------------------------------------------------------------- temp area


def test_cleanup_temp_removes_only_stale_entries(store: MediaStore):
    stale = store.temp_file(suffix=".wav")
    fresh = store.temp_file(suffix=".wav")
    old = time.time() - 7200
    os.utime(stale, (old, old))
    removed = store.cleanup_temp(max_age_seconds=3600)
    assert removed >= 1
    assert not stale.exists()
    assert fresh.exists()


# ------------------------------------------------- DB references + pipeline


def test_media_asset_reference_persists(tmp_path: Path):
    db = Database(tmp_path / "db.sqlite")
    store = MediaStore(tmp_path / "media")
    obj = store.put(_write(tmp_path / "src.txt", b"content identity test"))
    db.upsert_source(
        "s1", str(tmp_path / "src.txt"), "src.txt", "document", content_hash=obj.sha256
    )
    db.add_media_asset("a1", obj.key, obj.sha256, "original", source_id="s1", size_bytes=obj.size)
    assets = db.get_media_assets("s1")
    assert len(assets) == 1
    assert assets[0]["media_key"] == obj.key
    src_row = db.find_source_by_media_key(obj.key)
    assert src_row is not None and src_row["id"] == "s1"
    assert obj.key in db.list_media_keys()
    db.close()


def test_pipeline_ingest_registers_canonical_media(tmp_path: Path):
    """End-to-end: ingest writes the original into the MediaStore and records
    the reference; re-import of copied bytes dedups to one object."""
    from reminiscence.app.services.engine import EnginePaths, ReminiscenceEngine

    paths = EnginePaths(tmp_path / "data", tmp_path / "data" / "db.sqlite", tmp_path / "models")
    engine = ReminiscenceEngine(paths=paths)
    db = engine.db
    text = (
        "# Attention\n\nThe transformer architecture relies on scaled dot-product "
        "attention mechanisms which allow parallel training over sequences. " + "word " * 50
    )
    orig = _write(tmp_path / "notes" / "attn.md", text.encode())
    res = engine.pipeline.ingest(orig)
    assert res.events_created > 0
    assets = db.get_media_assets(res.source_id)
    assert len(assets) == 1
    assert assets[0]["media_type"] == "original"
    key = assets[0]["media_key"]
    assert key.startswith("objects/")
    assert engine.pipeline.media_store.verify(key) is True

    # Copy same bytes elsewhere and re-ingest: no duplicate media object.
    copy = _write(tmp_path / "backup" / "attn-copy.md", text.encode())
    res2 = engine.pipeline.ingest(copy)
    assert res2.events_created == 0  # moved/copied file detection
    assert engine.pipeline.media_store.stats()["objects"]["count"] == 1
    db.close()


def test_schema_version_4_migrations_applied(tmp_path: Path):
    db = Database(tmp_path / "db.sqlite")
    assert db.get_schema_version() >= 4
    cols = {r[1] for r in db._conn.execute("PRAGMA table_info(media_assets)")}
    assert {"asset_id", "media_key", "sha256", "media_type", "parent_asset_id"} <= cols
    src_cols = {r[1] for r in db._conn.execute("PRAGMA table_info(sources)")}
    assert "media_key" in src_cols
    db.close()
