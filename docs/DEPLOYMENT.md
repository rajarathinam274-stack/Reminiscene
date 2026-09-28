# Reminiscence — Deployment (Desktop Product)

Target: Windows 11 ARM64 on Snapdragon X-series HP PCs; CPU-only Linux/macOS
development also supported. No server tier is required or planned for the core.

Packaging: PyInstaller (native single-app bundle) initially.
- Core app + SQLite + FTS5 + NumPy vector index run offline.
- Models live under REMINISCENCE_MODEL_DIR, verified by manifest+checksum.
- QNN/NPU path is optional and isolated: enabled only when the ONNX Runtime
  reports QNNExecutionProvider actually loaded. Otherwise CPU fallback is
  labeled explicitly (never hidden).

Release checklist (from Definition of Done): reproducible install, CI green,
offline test pass, benchmarks recorded from real hardware, security tests,
export/delete verified. See docs/MISSING_IMPLEMENTATION_PLAN.log for current
gaps — do not ship claiming features that are marked 🟡/🔴 there.
