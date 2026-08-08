"""Uniform ASR interface so the app and the benchmark harness treat every
model identically. Add a model = add one subclass. Nothing else changes."""

from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class Transcript:
    text: str
    model_name: str
    latency_s: float
    language_hint: Optional[str] = None
    raw: dict = field(default_factory=dict)


class ASRModel:
    """Subclass and implement _transcribe(). Everything else is free."""

    name: str = "unnamed"
    # Free-text notes that flow straight into your benchmark report's
    # pros/cons table. Fill these in honestly as you test.
    notes: str = ""

    def transcribe(self, audio_path: str, language_hint: Optional[str] = None) -> Transcript:
        t0 = time.perf_counter()
        text, raw = self._transcribe(audio_path, language_hint)
        return Transcript(
            text=(text or "").strip(),
            model_name=self.name,
            latency_s=round(time.perf_counter() - t0, 3),
            language_hint=language_hint,
            raw=raw or {},
        )

    def _transcribe(self, audio_path: str, language_hint: Optional[str]):
        raise NotImplementedError

    def available(self) -> bool:
        """Return False when creds/weights are missing so the harness can
        skip cleanly instead of crashing mid-run."""
        return True
