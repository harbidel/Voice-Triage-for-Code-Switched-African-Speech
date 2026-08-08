"""Intron Sahara STT adapter.

Docs: https://docs.voice.intron.io/docs/stt/file-upload-sync
Endpoint: POST https://infer.voice.intron.io/file/v1/upload/sync
Constraints that bite: audio <= 120s, 30 req/min, sync route needs an
Integrator Account. A 503 is NOT a failure -- it returns a file_id you poll.
"""

import os
import time
import requests
from typing import Optional
from asr_base import ASRModel

SYNC_URL = "https://infer.voice.intron.io/file/v1/upload/sync"
STATUS_URL = "https://infer.voice.intron.io/file/v1/status"  # verify path in docs
DEFAULT_TIMEOUT = 180


class SaharaASR(ASRModel):
    name = "sahara-intron"
    notes = (
        "African-accent optimised. Native telehealth post-processing "
        "(entities, SOAP, ICD). Constraints: 120s audio cap, 30 rpm."
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        category: str = "file_category_telehealth",
        language: str = "en",
        native_entities: bool = False,
    ):
        self.api_key = api_key or os.environ.get("INTRON_API_KEY", "")
        self.category = category
        self.language = language
        # Leave False for the headline benchmark so ASR is the only variable.
        # Flip True only for the "Sahara native extraction" side-experiment.
        self.native_entities = native_entities

    def available(self) -> bool:
        return bool(self.api_key)

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}"}

    def _transcribe(self, audio_path: str, language_hint: Optional[str]):
        form = {
            "audio_file_name": (None, os.path.basename(audio_path)),
            "use_category": (None, self.category),
            "use_language_asr_input": (None, language_hint or self.language),
        }
        if self.native_entities:
            form["get_entity_list"] = (None, "TRUE")

        with open(audio_path, "rb") as fh:
            files = {**form, "audio_file_blob": (os.path.basename(audio_path), fh)}
            resp = requests.post(
                SYNC_URL, headers=self._headers(), files=files, timeout=DEFAULT_TIMEOUT
            )

        # Rate limited -- respect Retry-After and try once more.
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", "5"))
            time.sleep(wait + 1)
            return self._transcribe(audio_path, language_hint)

        payload = _safe_json(resp)

        # 503 = still processing. Poll with the returned file_id.
        if resp.status_code == 503:
            file_id = (payload.get("data") or {}).get("file_id")
            if not file_id:
                raise RuntimeError(f"Sahara 503 with no file_id: {resp.text[:300]}")
            payload = self._poll(file_id)

        if resp.status_code not in (200, 503):
            raise RuntimeError(f"Sahara {resp.status_code}: {resp.text[:300]}")

        data = payload.get("data") or {}
        return data.get("audio_transcript", ""), data

    def _poll(self, file_id: str, attempts: int = 30, delay: float = 4.0):
        for _ in range(attempts):
            time.sleep(delay)
            r = requests.get(
                f"{STATUS_URL}/{file_id}", headers=self._headers(), timeout=60
            )
            p = _safe_json(r)
            status = (p.get("data") or {}).get("processing_status", "")
            if status == "FILE_TRANSCRIBED":
                return p
            if "FAIL" in status.upper() or "ERROR" in status.upper():
                raise RuntimeError(f"Sahara processing failed: {status}")
        raise TimeoutError(f"Sahara poll exhausted for {file_id}")


def _safe_json(resp):
    try:
        return resp.json()
    except Exception:
        return {}
