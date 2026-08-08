"""Structured intake record + SATS-aligned triage tiers.

SATS (South African Triage Scale) is the de facto standard across many African
emergency units, so aligning to it means clinicians already know how to read
the output -- and judges recognise it.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from enum import Enum


class Tier(str, Enum):
    RED = "RED"        # immediate -- resuscitation
    ORANGE = "ORANGE"  # very urgent -- <10 min
    YELLOW = "YELLOW"  # urgent -- <60 min
    GREEN = "GREEN"    # routine -- <4 hr
    BLUE = "BLUE"      # non-clinical / already deceased on arrival

TIER_ORDER = [Tier.GREEN, Tier.YELLOW, Tier.ORANGE, Tier.RED]


def max_tier(a: Tier, b: Tier) -> Tier:
    """Escalate-only. Used everywhere so nothing can silently de-escalate."""
    if a == Tier.BLUE:
        return b
    if b == Tier.BLUE:
        return a
    return a if TIER_ORDER.index(a) >= TIER_ORDER.index(b) else b


@dataclass
class Vitals:
    resp_rate: Optional[int] = None
    heart_rate: Optional[int] = None
    systolic_bp: Optional[int] = None
    temperature_c: Optional[float] = None
    avpu: Optional[str] = None       # Alert / Voice / Pain / Unresponsive
    mobility: Optional[str] = None   # walking / with-help / stretcher


@dataclass
class IntakeRecord:
    # --- identity (kept minimal on purpose; see ETHICS.md) ---
    patient_ref: str = ""            # pseudonymous queue ref, never a name
    age_years: Optional[float] = None
    sex: Optional[str] = None
    pregnant: Optional[bool] = None

    # --- clinical ---
    chief_complaint: str = ""
    symptoms: List[str] = field(default_factory=list)
    negated_symptoms: List[str] = field(default_factory=list)
    duration_text: str = ""
    duration_hours: Optional[float] = None
    severity_0_10: Optional[int] = None
    vitals: Vitals = field(default_factory=Vitals)
    meds: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    history: List[str] = field(default_factory=list)

    # --- provenance / audit ---
    languages_detected: List[str] = field(default_factory=list)
    source_transcript: str = ""
    asr_model: str = ""
    low_confidence_fields: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["vitals"] = asdict(self.vitals) if not isinstance(self.vitals, dict) else self.vitals
        return d


# Fields the agent must chase before it will stop asking questions.
REQUIRED_FIELDS = ["chief_complaint", "duration_text", "age_years", "severity_0_10"]


@dataclass
class TriageResult:
    tier: Tier
    reasons: List[str] = field(default_factory=list)
    red_flags: List[str] = field(default_factory=list)
    tews_score: int = 0
    escalated_by_rule: bool = False
    disposition: str = ""
    requires_human_review: bool = True   # always true; this is decision support
