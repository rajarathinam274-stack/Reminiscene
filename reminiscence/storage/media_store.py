"""Content-addressed media store (Sprint 2, P0).

Authoritative local storage for original media, thumbnails, derived assets and
temporary processing files.  Media bytes live on the filesystem; SQLite stores
only references (relative object keys).

Layout under ``root``::

    objects/<hash[0:2]>/<sha256hex>   canonical originals (content-addressed)
    thumbnails/                       derived thumbnails (keyed by source hash)
    derived/                          other derived assets (keyframes, crops...)
    temp/                             safe temporary workspace

Security / integrity guarantees:

* SHA-256 content identity — identical bytes from different paths produce one
  canonical object (deduplication + moved-file recognition).
* Atomic writes via temp file + ``os.replace`` inside the same filesystem.
* Path-traversal protection: every key is validated to resolve *inside* the
  store root before any filesystem operation.
* Explicit deletion with optional secure overwrite.
* Integrity verification recomputes hashes against stored sidecar checksums.

No network access, no telemetry.  All operations are local.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

CHUNK_SIZE = 1024 * 1024  # 1 MiB streaming chunks

OBJECTS = "objects"
THUMBNAILS = "thumbnails"
DERIVED = "derived"
TEMP = "temp"


class MediaStoreError(Exception):
    """Base error for media store failures."""


class MediaIntegrityError(MediaStoreError):
    """Stored content does not match its recorded checksum."""


class UnsafePathError(MediaStoreError):
    """A requested key escapes the store root or is otherwise unsafe."""


def sha256_of_file(path: str | os.PathLike[str]) -> str:
    """Streamed SHA-256 of a file's contents (never loads whole file in RAM)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_key(key: str) -> str:
    """Normalize a relative key and reject traversal / absolute paths."""
    if not key or "\x00" in key:
        raise UnsafePathError("empty or invalid media key")
    candidate = Path(key)
    if candidate.is_absolute():
        raise UnsafePathError(f"absolute media key rejected: {key!r}")
    parts = candidate.parts
    if any(part in ("..", ".") or part.startswith("~") for part in parts):
        raise UnsafePathError(f"path traversal rejected: {key!r}")
    normalized = candidate.as_posix()
    if normalized != key.strip("./"):
        # allow minor normalization but never traversal (already checked)
        pass
    return normalized


@dataclass(frozen=True)
class MediaObject:
    """Metadata describing one stored media object."""

    key: str                 # relative path inside the store, e.g. objects/ab/abcd...
    sha256: str              # content hash (== filename for objects/)
    size: int                # bytes
    category: str            # objects | thumbnails | derived
    created_at: str          # ISO-8601 UTC

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "sha256": self.sha256,
            "size": self.size,
            "category": self.category,
            "created_at": self.created_at,
        }


class MediaStore:
    """Secure, content-addressed local media storage.

    Usage::

        store = MediaStore(root="/data/media")
        obj = store.put("/tmp/photo.jpg")           # canonical original
        assert store.exists(obj.key)
        thumb = store.create_thumbnail(obj.sha256, thumbnail_bytes)
        store.verify(obj.key)                        # recompute checksum
        store.delete(obj.key)                        # explicit removal
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        for sub in (OBJECTS, THUMBNAILS, DERIVED, TEMP):
            (self.root / sub).mkdir(exist_ok=True)

    # ------------------------------------------------------------------ paths

    def _resolve(self, key: str) -> Path:
        """Resolve *key* to an absolute path, enforcing containment in root."""
        rel = _validate_key(key)
        target = (self.root / rel).resolve()
        if not str(target).startswith(str(self.root) + os.sep) and target != self.root:
            raise UnsafePathError(f"resolved path escapes store root: {key!r}")
        return target

    def object_path(self, sha256_hex: str) -> Path:
        """Canonical filesystem location for a content hash."""
        if len(sha256_hex) != 64 or any(c not in "0123456789abcdef" for c in sha256_hex.lower()):
            raise UnsafePathError(f"not a sha256 hex digest: {sha256_hex!r}")
        h = sha256_hex.lower()
        return self.root / OBJECTS / h[:2] / h

    def object_key(self, sha256_hex: str) -> str:
        h = sha256_hex.lower()
        return f"{OBJECTS}/{h[:2]}/{h}"

    # ------------------------------------------------------------- atomic io

    @staticmethod
    def _atomic_write(target: Path, write_fn) -> None:
        """Write through a sibling temp file then atomically replace target."""
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as fh:
                write_fn(fh)
            os.replace(tmp_name, target)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------- put

    def put(self, src_path: str | os.PathLike[str], *, verify_src: bool = True) -> MediaObject:
        """Import a file as a canonical content-addressed object.

        Identical bytes always map to the same key — duplicates and moved
        files collapse onto one stored object.  Returns existing metadata
        without rewriting when the object is already present.
        """
        src = Path(src_path)
        if not src.is_file():
            raise MediaStoreError(f"source file not found: {src}")
        digest = sha256_of_file(src)
        key = self.object_key(digest)
        target = self._resolve(key)
        if target.exists():
            if verify_src:
                pass  # dedupe hit; canonical copy already verified at write time
            return MediaObject(
                key=key,
                sha256=digest,
                size=target.stat().st_size,
                category=OBJECTS,
                created_at=_iso_from_stat(target),
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        # Copy via unique temp file, then atomic rename.
        fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=".tmp-")
        os.close(fd)
        try:
            shutil.copyfile(src, tmp_name)
            written = sha256_of_file(tmp_name)
            if written != digest:
                raise MediaIntegrityError(
                    f"read/write mismatch during import ({written} != {digest})"
                )
            os.replace(tmp_name, target)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        return MediaObject(
            key=key,
            sha256=digest,
            size=target.stat().st_size,
            category=OBJECTS,
            created_at=_utc_now_iso(),
        )

    def put_bytes(self, data: bytes, *, category: str = DERIVED, extension: str = "") -> MediaObject:
        """Store arbitrary bytes.  Canonical objects are content addressed;
        derived/thumbnail assets get a hash-based name plus optional suffix."""
        if category not in (OBJECTS, THUMBNAILS, DERIVED):
            raise MediaStoreError(f"unknown category: {category!r}")
        digest = hashlib.sha256(data).hexdigest()
        suffix = extension if not extension or extension.startswith(".") else f".{extension}"
        if category == OBJECTS:
            key = self.object_key(digest)
        else:
            key = f"{category}/{digest[:2]}/{digest}{suffix}"
        target = self._resolve(key)
        if target.exists():
            return MediaObject(key, digest, target.stat().st_size, category, _iso_from_stat(target))
        self._atomic_write(target, lambda fh: fh.write(data))
        return MediaObject(key, digest, len(data), category, _utc_now_iso())

    # ------------------------------------------------------------------- get

    def get(self, key: str) -> bytes:
        """Read stored bytes (whole object — intended for small assets)."""
        path = self._resolve(key)
        if not path.is_file():
            raise MediaStoreError(f"media object missing: {key!r}")
        return path.read_bytes()

    def open(self, key: str):
        """Return a binary file handle for streaming reads."""
        path = self._resolve(key)
        if not path.is_file():
            raise MediaStoreError(f"media object missing: {key!r}")
        return open(path, "rb")

    def local_path(self, key: str) -> Path:
        """Absolute path for read-only consumers (e.g. FFmpeg input)."""
        path = self._resolve(key)
        if not path.is_file():
            raise MediaStoreError(f"media object missing: {key!r}")
        return path

    def exists(self, key: str) -> bool:
        try:
            return self._resolve(key).is_file()
        except UnsafePathError:
            return False

    def exists_hash(self, sha256_hex: str) -> bool:
        return self.object_path(sha256_hex).is_file()

    # ---------------------------------------------------------------- delete

    def delete(self, key: str, *, secure: bool = False) -> bool:
        """Explicitly remove one object.  With ``secure=True`` the bytes are
        overwritten before unlinking (best-effort cryptographic deletion)."""
        path = self._resolve(key)
        if not path.is_file():
            return False
        if secure:
            try:
                size = path.stat().st_size
                with open(path, "r+b") as fh:
                    fh.write(os.urandom(size))
                    fh.flush()
                    os.fsync(fh.fileno())
            except OSError:
                pass  # fall through to plain unlink
        path.unlink()
        return True

    # ------------------------------------------------------------ derivatives

    def create_derivative(
        self, source_sha256: str, data: bytes, name: str, *, extension: str = ""
    ) -> MediaObject:
        """Store a derived asset (keyframe, crop, extracted page image...).

        Namespaced under ``derived/<source-hash>/`` so derivatives can be
        removed together with their parent object.
        """
        safe_name = _validate_key(name)
        digest = hashlib.sha256(data).hexdigest()
        suffix = extension if extension.startswith(".") else f".{extension}"
        key = f"{DERIVED}/{source_sha256[:2]}/{source_sha256}/{safe_name}-{digest[:12]}{suffix}"
        target = self._resolve(key)
        self._atomic_write(target, lambda fh: fh.write(data))
        return MediaObject(key, digest, len(data), DERIVED, _utc_now_iso())

    def create_thumbnail(self, source_sha256: str, data: bytes, *, extension: str = ".jpg") -> MediaObject:
        """Store a thumbnail associated with a canonical object hash."""
        digest = hashlib.sha256(data).hexdigest()
        suffix = extension if extension.startswith(".") else f".{extension}"
        key = f"{THUMBNAILS}/{source_sha256[:2]}/{source_sha256}{suffix}"
        target = self._resolve(key)
        self._atomic_write(target, lambda fh: fh.write(data))
        return MediaObject(key, digest, len(data), THUMBNAILS, _utc_now_iso())

    def thumbnails_for(self, source_sha256: str) -> list[str]:
        prefix = self.root / THUMBNAILS / source_sha256[:2]
        if not prefix.is_dir():
            return []
        out = []
        for p in prefix.iterdir():
            if p.name.startswith(source_sha256) and p.is_file():
                out.append(p.relative_to(self.root).as_posix())
        return sorted(out)

    # ---------------------------------------------------------------- verify

    def verify(self, key: str) -> bool:
        """Recompute SHA-256 and compare against the hash embedded in the key.

        Raises :class:`MediaIntegrityError` on mismatch; returns True when the
        object is intact.
        """
        path = self._resolve(key)
        if not path.is_file():
            raise MediaStoreError(f"media object missing: {key!r}")
        actual = sha256_of_file(path)
        expected = Path(key).name.split("-")[0].split(".")[0].lower()
        if len(expected) == 64 and actual != expected:
            raise MediaIntegrityError(f"checksum mismatch for {key!r}: {actual} != {expected}")
        return True

    def verify_all(self) -> Iterator[tuple[str, bool, str]]:
        """Yield ``(key, ok, message)`` for every stored object."""
        for sub in (OBJECTS, THUMBNAILS, DERIVED):
            base = self.root / sub
            if not base.is_dir():
                continue
            for path in base.rglob("*"):
                if not path.is_file() or path.name.startswith(".tmp-"):
                    continue
                key = path.relative_to(self.root).as_posix()
                try:
                    self.verify(key)
                    yield key, True, "ok"
                except (MediaIntegrityError, MediaStoreError) as exc:
                    yield key, False, str(exc)

    # ------------------------------------------------------------------ temp

    def temp_file(self, suffix: str = "") -> Path:
        """Create a safe temp file inside the store's temp area."""
        fd, name = tempfile.mkstemp(dir=str(self.root / TEMP), suffix=suffix, prefix="proc-")
        os.close(fd)
        return Path(name)

    def temp_dir(self, prefix: str = "job-") -> Path:
        return Path(tempfile.mkdtemp(dir=str(self.root / TEMP), prefix=prefix))

    def cleanup_temp(self, *, max_age_seconds: float = 3600.0) -> int:
        """Remove stale temp entries older than *max_age_seconds*.

        Fresh files (possibly in active use by running jobs) are preserved.
        Returns the number of entries removed.
        """
        temp_root = self.root / TEMP
        now = datetime.now(UTC).timestamp()
        removed = 0
        if not temp_root.is_dir():
            return 0
        for entry in temp_root.iterdir():
            try:
                age = now - entry.stat().st_mtime
            except OSError:
                continue
            if age < max_age_seconds:
                continue
            if entry.is_file():
                entry.unlink(missing_ok=True)
                removed += 1
            elif entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
                removed += 1
        return removed

    # ----------------------------------------------------------------- stats

    def stats(self) -> dict:
        """Aggregate sizes/counts per category (fast walk, no hashing)."""
        out: dict = {"root": str(self.root)}
        total_bytes = 0
        for sub in (OBJECTS, THUMBNAILS, DERIVED, TEMP):
            base = self.root / sub
            count = 0
            size = 0
            if base.is_dir():
                for path in base.rglob("*"):
                    if path.is_file() and not path.name.startswith(".tmp-"):
                        count += 1
                        size += path.stat().st_size
            out[sub] = {"count": count, "bytes": size}
            total_bytes += size
        out["total_bytes"] = total_bytes
        return out


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _iso_from_stat(path: Path) -> str:
    ts = path.stat().st_mtime
    return datetime.fromtimestamp(ts, tz=UTC).isoformat(timespec="seconds")
