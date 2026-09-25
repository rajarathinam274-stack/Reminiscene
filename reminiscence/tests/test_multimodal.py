"""Sprint 4 tests: OCR/ASR backends, availability honesty, pipeline adapter wiring."""

from __future__ import annotations

import pytest

from reminiscence.ai.asr.asr import (
    TranscriptResult,
    UnavailableTranscriber,
    WhisperOnnxTranscriber,
    get_transcriber,
)
from reminiscence.ai.vision.ocr import (
    OnnxOCRPipeline,
    OcrRegion,
    UnavailableOCR,
    get_ocr_backend,
)
from reminiscence.ingestion.pipeline import AsrAdapter, OcrAdapter


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------
class TestOcrPipeline:
    def _pipeline(self):
        detector = lambda path: [([0.1, 0.1, 0.9, 0.1, 0.9, 0.3, 0.1, 0.3], 0.97)]
        recognizer = lambda path, box: ("BOARD MEETING NOTES", 0.93)
        return OnnxOCRPipeline(detector, recognizer, image_size_fn=lambda p: (2000, 1000))

    def test_extract_regions_with_bbox_evidence(self, tmp_path):
        img = tmp_path / "photo.png"
        img.write_bytes(b"\x89PNG fake")
        result = self._pipeline().recognize(img)
        assert len(result.regions) == 1
        r = result.regions[0]
        assert r.text == "BOARD MEETING NOTES"
        assert r.confidence == pytest.approx(0.93)
        # normalized coords scaled to image size -> pixel bbox for evidence UI
        assert r.bbox == pytest.approx((200.0, 100.0, 1800.0, 300.0))
        assert r.page == 1

    def test_searchable_text_join(self, tmp_path):
        img = tmp_path / "p.png"
        img.write_bytes(b"x")
        res = self._pipeline().recognize(img, page=3)
        assert "BOARD MEETING NOTES" in res.text
        assert res.model_id == "ocr-onnx-pipeline"

    def test_unwired_pipeline_reports_uninstalled(self):
        pipe = OnnxOCRPipeline(None, None)
        h = pipe.health()
        assert not h.installed and "not installed" in h.detail
        with pytest.raises(RuntimeError, match="OCR unavailable"):
            pipe.recognize("anything.png")

    def test_unavailable_placeholder_honest(self):
        o = UnavailableOCR("no model")
        assert o.health().backend == "none"
        with pytest.raises(RuntimeError):
            o.recognize("x.png")

    def test_factory_without_models_is_honest(self, tmp_path):
        backend = get_ocr_backend(None)
        assert isinstance(backend, (UnavailableOCR,)) or backend.health().installed is False

    def test_region_to_dict_shape(self):
        d = OcrRegion(text="hi", page=2, bbox=(1, 2, 3, 4), confidence=0.5).to_dict()
        assert d == {"text": "hi", "page": 2, "bbox": [1, 2, 3, 4], "confidence": 0.5}


class TestOcrAdapterForPipeline:
    """Acceptance: image text becomes searchable MemoryEvents through the pipeline."""

    def test_adapter_emits_pipeline_record_shape(self, tmp_path):
        from reminiscence.ai.vision.ocr import OcrResult

        class FakeBackend:
            def recognize(self, path, *, page=1):
                return OcrResult(page=page, text="hello ocr",
                                 regions=[OcrRegion("hello ocr", page, (1, 2, 3, 4), 0.9)])

        adapter = OcrAdapter(FakeBackend())
        recs = adapter.recognize(tmp_path / "img.png")
        rec = recs[0]
        assert rec["text"] == "hello ocr"
        assert rec["page"] == 1
        assert rec["bbox"] == [1, 2, 3, 4]
        assert rec["confidence"] == 0.9


# ---------------------------------------------------------------------------
# ASR
# ---------------------------------------------------------------------------
class TestAsrAvailability:
    def test_missing_artifacts_report_precise_reason(self, tmp_path):
        t = WhisperOnnxTranscriber(tmp_path)
        h = t.health()
        assert not h.installed
        assert "encoder_model.onnx" in h.detail

    def test_factory_falls_back_to_honest_stub(self, tmp_path):
        stub = get_transcriber(tmp_path / "nowhere")
        assert isinstance(stub, UnavailableTranscriber)
        with pytest.raises(RuntimeError, match="ASR unavailable"):
            stub.transcribe("audio.wav")

    def test_stub_health_names_reason(self):
        stub = UnavailableTranscriber("whisper artifact missing")
        h = stub.health()
        assert h.backend == "none" and "whisper" in h.detail

    def test_result_carries_provider_for_honest_benchmarks(self):
        res = TranscriptResult(segments=[], provider_used="CPUExecutionProvider")
        assert res.provider_used == "CPUExecutionProvider"


class TestAsrAdapterForPipeline:
    """Acceptance: local audio becomes timestamped searchable memory segments."""

    def test_adapter_maps_segments(self, tmp_path):
        from reminiscence.ai.asr.asr import TranscriptResult
        from reminiscence.memory.chunking import Segment

        class FakeTranscriber:
            def transcribe_segments(self, wav):
                return TranscriptResult(
                    segments=[Segment(0.0, 4.2, "first utterance"),
                              Segment(4.2, 9.0, "second utterance")],
                    language="en", provider_used="CPUExecutionProvider")

        adapter = AsrAdapter(FakeTranscriber())
        segs = adapter.transcribe(tmp_path / "a.wav")
        assert [(s.start, s.end, s.text) for s in segs] == [
            (0.0, 4.2, "first utterance"), (4.2, 9.0, "second utterance")]

    def test_adapter_rejects_legacy_interface(self):
        class Legacy:
            def transcribe(self, wav):
                return []

        with pytest.raises(TypeError, match="transcribe_segments"):
            AsrAdapter(Legacy())
