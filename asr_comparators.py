"""The two (or more) comparison models.

Pick a lineup that makes the report interesting:
  1. Sahara            -- required, African-optimised
  2. Whisper           -- global frontier baseline, English-first
  3. MMS / Seamless    -- explicitly multilingual, African language coverage

Three English-first models would produce a boring table. The contrast is
the point.
"""

import os
from typing import Optional
from asr_base import ASRModel


class FasterWhisperASR(ASRModel):
    """CPU-friendly Whisper via CTranslate2. Good default for a free Space."""

    name = "faster-whisper-small"
    notes = (
        "Strong on English matrix, degrades on embedded African-language spans. "
        "Tends to 'translate away' code-switches into fluent English -- watch for "
        "hallucinated fluency rather than honest errors."
    )

    def __init__(self, size: str = "small", compute_type: str = "int8"):
        self.size = size
        self.compute_type = compute_type
        self._model = None
        self.name = f"faster-whisper-{size}"

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.size, device="cpu", compute_type=self.compute_type
            )
        return self._model

    def available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401

            return True
        except ImportError:
            return False

    def _transcribe(self, audio_path: str, language_hint: Optional[str]):
        model = self._load()
        segments, info = model.transcribe(
            audio_path,
            language=language_hint if language_hint not in (None, "auto") else None,
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join(s.text.strip() for s in segments)
        return text, {"detected_language": info.language}


class HFPipelineASR(ASRModel):
    """Any HF ASR checkpoint. Swap model_id to change your third model.

    Worth trying:
      openai/whisper-large-v3
      facebook/mms-1b-all              (needs target lang adapter)
      facebook/seamless-m4t-v2-large
      any Naija/Yoruba fine-tune on the Hub
    """

    def __init__(self, model_id: str, name: Optional[str] = None, notes: str = ""):
        self.model_id = model_id
        self.name = name or model_id.split("/")[-1]
        self.notes = notes
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            from transformers import pipeline

            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_id,
                chunk_length_s=30,
                device=-1,  # CPU; set 0 on a GPU Space
            )
        return self._pipe

    def available(self) -> bool:
        try:
            import transformers  # noqa: F401

            return True
        except ImportError:
            return False

    def _transcribe(self, audio_path: str, language_hint: Optional[str]):
        out = self._load()(audio_path)
        return out.get("text", ""), {}


def build_registry(include_heavy: bool = False):
    """Central place the app and harness both pull models from."""
    from asr_sahara import SaharaASR

    models = [SaharaASR(api_key=os.environ.get("INTRON_API_KEY"))]
    models.append(FasterWhisperASR(size="small"))
    if include_heavy:
        models.append(
            HFPipelineASR(
                "facebook/seamless-m4t-v2-large",
                name="seamless-m4t-v2",
                notes="Explicit multilingual coverage; heavier, slower on CPU.",
            )
        )
    return [m for m in models if m.available()]
