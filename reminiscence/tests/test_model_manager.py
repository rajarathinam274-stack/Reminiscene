"""Sprint 3 tests: model manager lifecycle, checksums, cache, production embeddings."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from reminiscence.ai.embeddings.embedder import (
    HashingTFIDFEmbedder,
    OnnxMiniLMEmbedder,
    get_embedder,
)
from reminiscence.ai.model_manager import (
    ModelCache,
    ModelDownloader,
    ModelManager,
    ModelManagerError,
    ModelManifest,
    ModelState,
    ModelVerifier,
)


def _manifest(
    root: Path, *, blob: bytes = b"fake-onnx-bytes", file: str = "model.onnx"
) -> ModelManifest:
    return ModelManifest(
        model_id="embed-minilm-l6-v2",
        version="1.0.0",
        task="embeddings",
        format="onnx",
        quantization="fp32",
        file=file,
        sha256=hashlib.sha256(blob).hexdigest(),
        size=len(blob),
        minimum_ram_mb=256,
        supported_providers=["CPUExecutionProvider"],
        license="MIT",
        dim=384,
        model_path=root / file,
    )


# ---------------------------------------------------------------------------
# verification / states
# ---------------------------------------------------------------------------
class TestModelVerifier:
    def test_missing_artifact(self, tmp_path):
        m = _manifest(tmp_path)
        assert ModelVerifier.verify(m) == ModelState.MISSING

    def test_verified_roundtrip(self, tmp_path):
        blob = b"fake-onnx-bytes"
        (tmp_path / "model.onnx").write_bytes(blob)
        assert ModelVerifier.verify(_manifest(tmp_path, blob=blob)) == ModelState.VERIFIED

    def test_checksum_mismatch_detected(self, tmp_path):
        (tmp_path / "model.onnx").write_bytes(b"tampered-payload!!")  # same length, different bytes
        m = _manifest(tmp_path)  # expects sha256 of original blob
        assert ModelVerifier.verify(m) == ModelState.CHECKSUM_FAILED

    def test_size_mismatch_detected(self, tmp_path):
        (tmp_path / "model.onnx").write_bytes(b"short")
        assert ModelVerifier.verify(_manifest(tmp_path)) == ModelState.CHECKSUM_FAILED


class TestModelManagerLifecycle:
    def test_register_discovered_states(self, tmp_path):
        mgr = ModelManager(tmp_path / "models")
        mm = mgr.register(_manifest(tmp_path / "models"))
        assert mm.state == ModelState.MISSING
        (tmp_path / "models" / "model.onnx").write_bytes(b"fake-onnx-bytes")
        mm2 = mgr.register(_manifest(tmp_path / "models"))
        assert mm2.state == ModelState.VERIFIED

    def test_ensure_available_cpu_provider(self, tmp_path):
        mgr = ModelManager(tmp_path / "models")
        (tmp_path / "models").mkdir(parents=True, exist_ok=True)
        (tmp_path / "models" / "model.onnx").write_bytes(b"fake-onnx-bytes")
        key = _manifest(tmp_path / "models").key
        mgr.register(_manifest(tmp_path / "models"))
        mm = mgr.ensure_available(key, ["CPUExecutionProvider"])
        assert mm.state == ModelState.AVAILABLE

    def test_unsupported_when_no_provider(self, tmp_path):
        mgr = ModelManager(tmp_path / "models")
        (tmp_path / "models").mkdir(parents=True, exist_ok=True)
        (tmp_path / "models" / "model.onnx").write_bytes(b"fake-onnx-bytes")
        m = _manifest(tmp_path / "models")
        m = ModelManifest(**{**m.__dict__, "supported_providers": ["QNNExecutionProvider"]})
        mgr.register(m)
        mm = mgr.ensure_available(m.key, [])  # no providers at all
        assert mm.state == ModelState.UNSUPPORTED

    def test_load_blocked_after_checksum_failure(self, tmp_path):
        """Acceptance: checksum mismatch prevents model activation."""
        mgr = ModelManager(tmp_path / "models")
        (tmp_path / "models").mkdir(parents=True, exist_ok=True)
        (tmp_path / "models" / "model.onnx").write_bytes(b"tampered-payload!!")
        m = _manifest(tmp_path / "models")
        mgr.register(m)
        mm = mgr.ensure_available(m.key, ["CPUExecutionProvider"])
        assert mm.state == ModelState.CHECKSUM_FAILED
        with pytest.raises(ModelManagerError):
            mgr.load(m.key)
        with pytest.raises(ModelManagerError):
            mgr.activate(m.key)

    def test_load_requires_available_state(self, tmp_path):
        mgr = ModelManager(tmp_path / "models")
        m = _manifest(tmp_path / "models")
        mgr.register(m)  # MISSING
        with pytest.raises(ModelManagerError):
            mgr.load(m.key)

    def test_network_disabled_by_default(self, tmp_path):
        mgr = ModelManager(tmp_path / "models")  # allow_network defaults False
        m = ModelManifest(
            **{
                **_manifest(tmp_path / "models").__dict__,
                "source": "https://example.invalid/model.onnx",
            }
        )
        mgr.register(m)
        mm = mgr.ensure_available(m.key, ["CPUExecutionProvider"])
        assert mm.state == ModelState.MISSING  # never fetched silently

    def test_discover_reloads_manifests(self, tmp_path):
        root = tmp_path / "models"
        root.mkdir()
        (root / "model.onnx").write_bytes(b"fake-onnx-bytes")
        mgr = ModelManager(root)
        mgr.register(_manifest(root))
        mgr2 = ModelManager(root)
        found = mgr2.discover()
        assert len(found) == 1
        assert found[0].state == ModelState.VERIFIED


class TestModelCache:
    def test_manifest_json_roundtrip(self, tmp_path):
        cache = ModelCache(tmp_path)
        m = _manifest(tmp_path)
        cache.save_manifest(m)
        loaded = cache.load_manifests()
        assert loaded and loaded[0].model_id == m.model_id
        assert loaded[0].sha256 == m.sha256

    def test_path_traversal_rejected(self, tmp_path):
        cache = ModelCache(tmp_path)
        m = _manifest(tmp_path, file="../../escape.onnx")
        with pytest.raises(ModelManagerError):
            cache.artifact_path(m)

    def test_corrupt_manifest_skipped(self, tmp_path):
        (tmp_path / "bad.manifest.json").write_text("{not json", encoding="utf-8")
        assert ModelCache(tmp_path).load_manifests() == []


class TestDownloader:
    def test_refuses_non_http(self, tmp_path):
        with pytest.raises(ModelManagerError):
            ModelDownloader.download("file:///etc/passwd", tmp_path / "x.onnx")


# ---------------------------------------------------------------------------
# production embeddings (P0 item 5)
# ---------------------------------------------------------------------------
class TestProductionEmbeddings:
    def test_tokenizer_pairing_required(self, tmp_path):
        emb = OnnxMiniLMEmbedder(str(tmp_path / "model.onnx"), tokenizer=None)
        with pytest.raises(RuntimeError, match="tokenizer"):
            emb.embed(["hello"])

    def test_dimension_validation(self, monkeypatch, tmp_path):
        emb = OnnxMiniLMEmbedder(
            str(tmp_path / "model.onnx"),
            tokenizer=lambda texts: {
                "input_ids": np.ones((len(texts), 4)),
                "attention_mask": np.ones((len(texts), 4)),
            },
            expected_dim=384,
        )

        class FakeAdapter:
            info = None

            def run(self, feed):
                b = feed["input_ids"].shape[0]
                return [np.zeros((b, 4, 999), dtype=np.float32)]  # wrong hidden dim

        emb._adapter = FakeAdapter()
        with pytest.raises(RuntimeError, match="dimension validation failed"):
            emb.embed(["hello world"])

    def test_mean_pool_and_normalize(self, monkeypatch, tmp_path):
        emb = OnnxMiniLMEmbedder(
            str(tmp_path / "model.onnx"),
            tokenizer=lambda texts: {
                "input_ids": np.ones((len(texts), 3)),
                "attention_mask": np.array([[1, 1, 0]] * len(texts)),
            },
            expected_dim=384,
        )

        class FakeAdapter:
            info = None

            def run(self, feed):
                b = feed["input_ids"].shape[0]
                out = np.zeros((b, 3, 384), dtype=np.float32)
                out[:, :, 0] = 2.0  # tokens 0,1 active via mask
                return [out]

        emb._adapter = FakeAdapter()
        v = emb.embed(["a b c"])
        assert v.shape == (1, 384)
        assert abs(float(np.linalg.norm(v[0])) - 1.0) < 1e-5

    def test_metadata_traceable(self, tmp_path):
        emb = OnnxMiniLMEmbedder(
            str(tmp_path / "m.onnx"),
            model_version="2.0.1",
            quantization="int8",
        )
        md = emb.metadata
        assert md["model_id"] == "embed-minilm-l6-v2"
        assert md["model_version"] == "2.0.1"
        assert md["quantization"] == "int8"

    def test_factory_honest_fallback(self, tmp_path):
        """Configured-but-unloadable path returns labeled fallback, not fake MiniLM."""
        emb = get_embedder(prefer_model_path=str(tmp_path / "nonexistent.onnx"), dim=512)
        assert isinstance(emb, HashingTFIDFEmbedder)
        assert emb.model_id != OnnxMiniLMEmbedder.model_id

    def test_factory_without_path_is_fallback(self):
        assert isinstance(get_embedder(), HashingTFIDFEmbedder)
