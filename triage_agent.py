"""The agentic layer: transcript -> structured record -> missing-field questions
-> triage decision. Model-agnostic on purpose so the SAME extraction runs over
every ASR output, keeping ASR the only variable in the benchmark.
"""

import json
import os
import re
from typing import Optional, List, Tuple
import requests

from triage_schema import IntakeRecord, Vitals, Tier, REQUIRED_FIELDS
from triage_rules import assess

# ---------------------------------------------------------------------------
# LLM provider. Set LLM_PROVIDER to: anthropic | openai | compatible
#
#   anthropic   ANTHROPIC_API_KEY
#   openai      OPENAI_API_KEY
#   compatible  OPENAI_API_KEY + OPENAI_BASE_URL
#               (works with Groq, Together, OpenRouter, Ollama, vLLM…)
#
# Groq is a good hackathon fallback: free tier, very fast.
#   LLM_PROVIDER=compatible
#   OPENAI_BASE_URL=https://api.groq.com/openai/v1
#   LLM_MODEL=llama-3.3-70b-versatile
# ---------------------------------------------------------------------------
PROVIDER = os.environ.get("LLM_PROVIDER", "").lower().strip()

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o-mini",
    "compatible": "llama-3.3-70b-versatile",
}


def _detect_provider() -> str:
    if PROVIDER:
        return PROVIDER
    if os.environ.get("OPENAI_BASE_URL"):
        return "compatible"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "anthropic"


def _model_name(provider: str) -> str:
    return os.environ.get("LLM_MODEL") or DEFAULT_MODELS.get(provider, "gpt-4o-mini")


def _call(system: str, user: str, max_tokens: int = 1200) -> str:
    provider = _detect_provider()
    model = _model_name(provider)

    if provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        headers = {"content-type": "application/json", "anthropic-version": "2023-06-01"}
        if key:
            headers["x-api-key"] = key
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json={
                "model": model,
                "max_tokens": max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
            timeout=90,
        )
        r.raise_for_status()
        data = r.json()
        return "".join(
            b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"
        )

    # OpenAI and any OpenAI-compatible endpoint share one shape.
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    key = os.environ.get("OPENAI_API_KEY", "")
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    # Ask for strict JSON where the endpoint supports it. Harmless if ignored,
    # and _parse_json still cleans up whatever comes back.
    if "extract" in system.lower() or "JSON" in system:
        body["response_format"] = {"type": "json_object"}

    r = requests.post(
        f"{base}/chat/completions",
        headers={"content-type": "application/json", "Authorization": f"Bearer {key}"},
        json=body,
        timeout=90,
    )
    if r.status_code == 400 and "response_format" in body:
        body.pop("response_format")
        r = requests.post(
            f"{base}/chat/completions",
            headers={"content-type": "application/json", "Authorization": f"Bearer {key}"},
            json=body,
            timeout=90,
        )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _parse_json(text: str) -> dict:
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", t, re.S)
    try:
        return json.loads(m.group(0) if m else t)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
EXTRACT_SYS = """You extract a structured clinical intake record from a triage \
transcript. The patient may speak Nigerian Pidgin, Yoruba, Hausa, English, or \
switch between them mid-sentence. ASR errors are likely; work with what is there.

Respond with ONLY a JSON object, no markdown fences, no commentary:

{
  "chief_complaint": "one short phrase in plain English",
  "symptoms": ["english_snake_case", ...],
  "negated_symptoms": ["symptoms the patient explicitly DENIES", ...],
  "duration_text": "patient's own words for how long, else \\"\\"",
  "duration_hours": number or null,
  "severity_0_10": integer or null,
  "age_years": number or null,
  "sex": "male" | "female" | null,
  "pregnant": true | false | null,
  "meds": [], "allergies": [], "history": [],
  "vitals": {"resp_rate": null, "heart_rate": null, "systolic_bp": null,
             "temperature_c": null, "avpu": null, "mobility": null},
  "languages_detected": ["en","pcm","yo","ha"],
  "low_confidence_fields": ["fields you inferred from garbled or ambiguous text"],
  "suggested_tier": "RED" | "ORANGE" | "YELLOW" | "GREEN"
}

Rules:
- Extract only what is stated or clearly implied. Never invent vitals, ages, or
  symptoms. Missing means null or empty, not a guess.
- Negation matters and may FOLLOW the noun in Pidgin ("belle no dey pain me"
  means NO abdominal pain -> negated_symptoms). "kò"/"ko" (Yoruba) and
  "ba...ba"/"babu" (Hausa) are negators.
- Translate symptom words to English snake_case: "belle dey pain" ->
  "abdominal_pain", "body dey hot" / "zazzabi" / "ibà" -> "fever",
  "e dey shake" / "giri" -> "convulsions", "no fit breathe" -> "breathlessness".
- suggested_tier follows SATS: RED = immediate life threat (airway, unresponsive,
  seizure, heavy bleeding, stroke signs, obstetric emergency), ORANGE = very
  urgent, YELLOW = urgent, GREEN = routine. When unsure between two tiers,
  choose the HIGHER one.
- Anything you had to guess at goes in low_confidence_fields."""

QUESTION_SYS = """You are a triage intake assistant asking ONE follow-up question \
to fill a missing field. Mirror the patient's own language mix -- if they spoke \
Pidgin, ask in simple Pidgin; if Yoruba or Hausa, use that; else plain English. \
One short spoken-style sentence, no preamble, no list, no quotation marks."""


def extract(transcript: str, asr_model: str = "") -> Tuple[IntakeRecord, Tier]:
    raw = _parse_json(_call(EXTRACT_SYS, f"Transcript:\n{transcript}"))
    v = raw.get("vitals") or {}
    rec = IntakeRecord(
        chief_complaint=raw.get("chief_complaint") or "",
        symptoms=raw.get("symptoms") or [],
        negated_symptoms=raw.get("negated_symptoms") or [],
        duration_text=raw.get("duration_text") or "",
        duration_hours=raw.get("duration_hours"),
        severity_0_10=raw.get("severity_0_10"),
        age_years=raw.get("age_years"),
        sex=raw.get("sex"),
        pregnant=raw.get("pregnant"),
        meds=raw.get("meds") or [],
        allergies=raw.get("allergies") or [],
        history=raw.get("history") or [],
        vitals=Vitals(
            resp_rate=v.get("resp_rate"), heart_rate=v.get("heart_rate"),
            systolic_bp=v.get("systolic_bp"), temperature_c=v.get("temperature_c"),
            avpu=v.get("avpu"), mobility=v.get("mobility"),
        ),
        languages_detected=raw.get("languages_detected") or [],
        low_confidence_fields=raw.get("low_confidence_fields") or [],
        source_transcript=transcript,
        asr_model=asr_model,
    )
    try:
        suggested = Tier(raw.get("suggested_tier", "GREEN"))
    except ValueError:
        suggested = Tier.GREEN
    return rec, suggested


def missing_fields(rec: IntakeRecord) -> List[str]:
    out = []
    for f in REQUIRED_FIELDS:
        val = getattr(rec, f, None)
        if val in (None, "", []):
            out.append(f)
    return out


def next_question(rec: IntakeRecord) -> Optional[str]:
    miss = missing_fields(rec)
    if not miss:
        return None
    ctx = (
        f"Known so far: {rec.chief_complaint or 'nothing yet'}. "
        f"Patient's words: {rec.source_transcript[:400]}. "
        f"Missing: {miss[0]}."
    )
    try:
        return _call(QUESTION_SYS, ctx, max_tokens=100).strip()
    except Exception:
        return f"Please tell me about: {miss[0].replace('_',' ')}"


def run(transcript: str, asr_model: str = ""):
    """Full loop -> (record, triage_result, follow_up_question_or_None)."""
    rec, suggested = extract(transcript, asr_model)
    result = assess(rec, llm_tier=suggested)
    # Never block on questions when the case is hot -- act first, ask later.
    q = None if result.tier in (Tier.RED, Tier.ORANGE) else next_question(rec)
    return rec, result, q
