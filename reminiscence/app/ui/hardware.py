"""Runtime hardware/provider status helpers."""
from __future__ import annotations
import platform
from dataclasses import dataclass

@dataclass(frozen=True)
class HardwareStatus:
    machine: str
    system: str
    npu_verified: bool
    provider: str
    @property
    def headline(self) -> str:
        return "Snapdragon NPU ready" if self.npu_verified else "CPU / provider fallback"

def detect_hardware(engine) -> HardwareStatus:
    verified = bool(getattr(engine.scheduler, "npu_available", False))
    return HardwareStatus(platform.machine() or "unknown", platform.system() or "unknown",
                          verified, "QNN / NPU" if verified else "CPU")
