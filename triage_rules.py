"""Deterministic safety floor.

Design rule: the LLM may ESCALATE a tier but may never lower it below what
these rules produce. An LLM that hallucinates calm is the failure mode that
actually hurts someone, so calm is the thing we refuse to take on trust.

The lexicon covers Nigerian Pidgin, Yoruba and Hausa surface forms because
code-switched speech does not present red flags in clean clinical English.

!! VALIDATE THIS LEXICON WITH NATIVE SPEAKERS BEFORE ANY REAL USE. It is a
   starting point assembled for a prototype, not a clinically approved list.
"""

import re
from typing import List, Tuple
from triage_schema import Tier, TriageResult, IntakeRecord, max_tier

# ---------------------------------------------------------------------------
# Negation cues. Missing a negation flips meaning entirely -- "belle no dey
# pain me" must never become "abdominal pain".
# ---------------------------------------------------------------------------
NEGATION_CUES = [
    r"\bno\b", r"\bnot\b", r"\bnever\b", r"\bwithout\b", r"\bdeny\b", r"\bdenies\b",
    r"\bno dey\b", r"\bnothing\b", r"\bnone\b",
    r"\bkò\b", r"\bko\b", r"\bò\b",          # Yoruba
    r"\bba\b", r"\bbabu\b",                    # Hausa
]
NEG_WINDOW = 5  # tokens before the trigger


# ---------------------------------------------------------------------------
# (flag_name, tier, regex alternatives)
# ---------------------------------------------------------------------------
RED_FLAGS: List[Tuple[str, Tier, List[str]]] = [
    ("airway_breathing", Tier.RED, [
        r"can'?t breathe", r"cannot breathe", r"no fit breathe", r"breath(e|ing) hard",
        r"short(ness)? of breath", r"gasping", r"choking", r"struggling to breathe",
        r"ba na iya numfashi", r"numfashi", r"mi ò le mí", r"mi o le mi",
    ]),
    ("unresponsive", Tier.RED, [
        r"unconscious", r"unresponsive", r"not waking", r"won'?t wake",
        r"e no dey answer", r"faint(ed|ing)?", r"passed out", r"collapse[d]?",
        r"suma", r"dakú", r"daku",
    ]),
    ("seizure", Tier.RED, [
        r"seizure", r"convuls(ion|ing)", r"fitting", r"e dey shake", r"body dey jerk",
        r"giri", r"warwar",
    ]),
    ("severe_bleeding", Tier.RED, [
        r"heav(y|ily) bleed", r"bleeding a lot", r"blood dey comot", r"lot of blood",
        r"haemorrhag", r"hemorrhag", r"ẹ̀jẹ̀ púpọ̀", r"eje pupo", r"jini da yawa",
    ]),
    ("chest_pain", Tier.ORANGE, [
        r"chest pain", r"chest dey pain", r"pain in (my )?chest", r"crushing pain",
        r"ciwon kirji", r"àyà ń dùn", r"aya n dun",
    ]),
    ("stroke_signs", Tier.RED, [
        r"face (is )?droop", r"slurr(ed|ing) speech", r"one side.{0,15}(weak|numb)",
        r"can'?t move.{0,15}(arm|leg|side)", r"mouth don twist",
    ]),
    ("obstetric_emergency", Tier.RED, [
        r"pregnan.{0,20}bleed", r"bleed.{0,20}pregnan", r"belle.{0,20}blood",
        r"water (don )?break", r"labou?r pain", r"baby no dey move",
        r"reduced fetal movement",
    ]),
    ("poisoning", Tier.RED, [
        r"poison", r"overdose", r"drank.{0,20}(chemical|kerosene|acid)",
        r"swallow(ed)?.{0,20}(chemical|pesticide|rat poison)", r"sniper",
    ]),
    ("self_harm", Tier.RED, [
        r"kill (my)?self", r"suicid", r"end my life", r"harm (my)?self",
        r"want to die",
    ]),
    ("neonatal_infant_fever", Tier.ORANGE, [
        r"newborn.{0,20}(fever|hot)", r"baby.{0,20}(fever|hot body|convuls)",
        r"pikin.{0,20}body dey hot",
    ]),
    ("severe_dehydration", Tier.ORANGE, [
        r"not (passing|passed) urine", r"no urine", r"sunken eyes",
        r"e no fit chop", r"vomiting everything", r"cannot keep.{0,15}down",
    ]),
    ("major_trauma", Tier.RED, [
        r"\baccident\b", r"hit by.{0,15}(car|okada|bike|truck)", r"\bfell from\b",
        r"\bgunshot\b", r"\bstab(bed)?\b", r"deep cut", r"\bburn(ed|t|s)?\b",
    ]),
    ("severe_malaria_signs", Tier.ORANGE, [
        r"fever.{0,25}(confus|convuls|unconsc)", r"yellow eyes", r"dark urine",
        r"zazzab[iı].{0,25}(rikice|suma)",
    ]),
]


# A negation only governs its own clause. Without this, the "no" in
# "I no fit breathe, my chest dey pain me" suppresses the chest-pain flag --
# a false negative on a red flag, i.e. the direction that under-triages.
CLAUSE_BREAK = re.compile(r"[,;.!?]|\b(and|but|then|also|plus|though|however)\b", re.I)


def _is_negated(text: str, match_start: int) -> bool:
    prefix = text[:match_start]
    breaks = list(CLAUSE_BREAK.finditer(prefix))
    if breaks:
        prefix = prefix[breaks[-1].end():]
    tokens = re.findall(r"\S+", prefix)[-NEG_WINDOW:]
    window = " ".join(tokens).lower()
    return any(re.search(cue, window) for cue in NEGATION_CUES)


def scan_red_flags(text: str) -> List[Tuple[str, Tier]]:
    """Return non-negated red flags found in free text."""
    found, seen = [], set()
    low = (text or "").lower()
    for name, tier, patterns in RED_FLAGS:
        if name in seen:
            continue
        for pat in patterns:
            for m in re.finditer(pat, low, flags=re.IGNORECASE):
                if not _is_negated(low, m.start()):
                    found.append((name, tier))
                    seen.add(name)
                    break
            if name in seen:
                break
    return found


# ---------------------------------------------------------------------------
# TEWS -- Triage Early Warning Score (SATS component), adult simplification.
# ---------------------------------------------------------------------------
def tews(v) -> int:
    s = 0
    mob = (v.mobility or "").lower()
    if "help" in mob:
        s += 1
    elif "stretcher" in mob or "carried" in mob:
        s += 2

    if v.resp_rate is not None:
        rr = v.resp_rate
        s += 0 if 9 <= rr <= 14 else 1 if (15 <= rr <= 20 or rr < 9) else 2 if 21 <= rr <= 29 else 3

    if v.heart_rate is not None:
        hr = v.heart_rate
        s += 0 if 51 <= hr <= 100 else 1 if (101 <= hr <= 110 or 41 <= hr <= 50) else 2 if 111 <= hr <= 129 else 3

    if v.systolic_bp is not None:
        sbp = v.systolic_bp
        s += 0 if sbp >= 101 else 2 if 81 <= sbp <= 100 else 3

    if v.temperature_c is not None:
        s += 0 if 35.0 <= v.temperature_c <= 38.4 else 1

    avpu = (v.avpu or "").upper()
    if avpu.startswith("V"):
        s += 1
    elif avpu.startswith("P"):
        s += 2
    elif avpu.startswith("U"):
        s += 3
    return s


def tier_from_tews(score: int) -> Tier:
    if score >= 7:
        return Tier.RED
    if score >= 5:
        return Tier.ORANGE
    if score >= 3:
        return Tier.YELLOW
    return Tier.GREEN


DISPOSITION = {
    Tier.RED: "Resuscitation bay now. Call clinician immediately.",
    Tier.ORANGE: "Very urgent — clinician review within 10 minutes.",
    Tier.YELLOW: "Urgent — review within 60 minutes.",
    Tier.GREEN: "Routine — standard outpatient queue.",
    Tier.BLUE: "Non-clinical pathway.",
}


def assess(record: IntakeRecord, llm_tier: Tier = Tier.GREEN) -> TriageResult:
    """Combine rules + vitals + LLM opinion. Escalate-only."""
    haystack = " ".join(
        [record.source_transcript, record.chief_complaint, " ".join(record.symptoms)]
    )
    flags = scan_red_flags(haystack)

    rule_tier = Tier.GREEN
    for _, t in flags:
        rule_tier = max_tier(rule_tier, t)

    score = tews(record.vitals)
    vitals_tier = tier_from_tews(score)

    # Age-based escalation floors.
    age_tier = Tier.GREEN
    if record.age_years is not None:
        if record.age_years < 0.25:
            age_tier = Tier.ORANGE
        elif record.age_years < 5 or record.age_years >= 70:
            age_tier = Tier.YELLOW
    if record.pregnant:
        age_tier = max_tier(age_tier, Tier.YELLOW)

    sev_tier = Tier.GREEN
    if record.severity_0_10 is not None and record.severity_0_10 >= 8:
        sev_tier = Tier.YELLOW

    floor = max_tier(max_tier(rule_tier, vitals_tier), max_tier(age_tier, sev_tier))
    final = max_tier(floor, llm_tier)

    reasons = []
    if flags:
        reasons.append("Red flags: " + ", ".join(f for f, _ in flags))
    if score:
        reasons.append(f"TEWS {score} → {vitals_tier.value}")
    if age_tier != Tier.GREEN:
        reasons.append(f"Age/pregnancy floor → {age_tier.value}")
    if sev_tier != Tier.GREEN:
        reasons.append(f"Severity {record.severity_0_10}/10 → {sev_tier.value}")
    if final != llm_tier:
        reasons.append(f"Rule floor overrode model suggestion ({llm_tier.value} → {final.value})")

    return TriageResult(
        tier=final,
        reasons=reasons or ["No red flags; routine presentation."],
        red_flags=[f for f, _ in flags],
        tews_score=score,
        escalated_by_rule=(final != llm_tier),
        disposition=DISPOSITION[final],
        requires_human_review=True,
    )
