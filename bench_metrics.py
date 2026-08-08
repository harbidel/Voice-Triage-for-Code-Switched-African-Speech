"""Metrics that say something WER alone cannot.

Four layers, cheapest to most meaningful:
  1. WER               -- table stakes
  2. CS-boundary WER   -- where code-switching models actually break
  3. Clinical recall   -- do the words that change care survive?
  4. Triage agreement  -- does the downstream DECISION match ground truth?

(4) is the headline. A model can lose on WER and still triage correctly, and
that inversion is the most interesting thing you can put in front of a judge.
"""

import re
import unicodedata
from typing import List, Dict, Tuple

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
_PUNCT = re.compile(r"[^\w\s'\u00C0-\u024F\u1E00-\u1EFF]", re.UNICODE)


def normalise(text: str, keep_diacritics: bool = True) -> str:
    t = (text or "").lower().strip()
    t = _PUNCT.sub(" ", t)
    if not keep_diacritics:
        t = "".join(
            c for c in unicodedata.normalize("NFD", t)
            if unicodedata.category(c) != "Mn"
        )
    return re.sub(r"\s+", " ", t).strip()


def tokens(text: str, keep_diacritics: bool = True) -> List[str]:
    return normalise(text, keep_diacritics).split()


# ---------------------------------------------------------------------------
# Levenshtein alignment (no external dependency)
# ---------------------------------------------------------------------------
def _align(ref: List[str], hyp: List[str]):
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    bp = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0], bp[i][0] = i, "D"
    for j in range(1, m + 1):
        d[0][j], bp[0][j] = j, "I"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j], bp[i][j] = d[i - 1][j - 1], "C"
            else:
                sub, dele, ins = d[i - 1][j - 1] + 1, d[i - 1][j] + 1, d[i][j - 1] + 1
                best = min(sub, dele, ins)
                d[i][j] = best
                bp[i][j] = "S" if best == sub else ("D" if best == dele else "I")
    # backtrace -> list of (op, ref_index or None)
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        op = bp[i][j]
        if op == "C" or op == "S":
            ops.append((op, i - 1)); i -= 1; j -= 1
        elif op == "D":
            ops.append((op, i - 1)); i -= 1
        else:
            ops.append((op, None)); j -= 1
    ops.reverse()
    return d[n][m], ops


def wer(reference: str, hypothesis: str, keep_diacritics: bool = True) -> float:
    r, h = tokens(reference, keep_diacritics), tokens(hypothesis, keep_diacritics)
    if not r:
        return 0.0 if not h else 1.0
    dist, _ = _align(r, h)
    return dist / len(r)


# ---------------------------------------------------------------------------
# Code-switch boundary WER
# ---------------------------------------------------------------------------
# Lightweight language tagging: a token is EN if it appears in a common-English
# set, otherwise OTHER. Crude but reproducible and honest about its limits.
# Swap in a proper LID if you have time -- document whichever you used.
_EN_COMMON = set("""
a about after all also am an and any are as at back be because been before but by
call can come could day did do does doing done down each even feel first for from
get give go going good got had has have he her here him his how i if in into is it
its just know like little long look make man many me more most much my need never
new no not now of off on one only or other our out over people say see she should
so some take tell than that the their them then there these they thing think this
those time to today too two up us use very want was way we well went were what when
where which who why will with would year yes you your
pain fever cough blood chest head stomach belly body hot cold vomit vomiting
breathe breathing sick hospital doctor nurse medicine drug tablet water sleep eat
week month day night morning yesterday since ago severe mild bad worse better
pregnant baby child mother father years old
""".split())


def tag_languages(toks: List[str]) -> List[str]:
    return ["EN" if t in _EN_COMMON else "OTHER" for t in toks]


def boundary_indices(toks: List[str], window: int = 1) -> set:
    """Indices within `window` tokens of a language switch."""
    tags = tag_languages(toks)
    idx = set()
    for i in range(1, len(tags)):
        if tags[i] != tags[i - 1]:
            for k in range(max(0, i - window), min(len(toks), i + window + 1)):
                idx.add(k)
    return idx


def codeswitch_boundary_wer(reference: str, hypothesis: str, window: int = 1) -> Dict:
    """Error rate restricted to reference tokens near a language switch."""
    r, h = tokens(reference), tokens(hypothesis)
    if not r:
        return {"cs_wer": 0.0, "boundary_tokens": 0, "non_boundary_wer": 0.0}
    bidx = boundary_indices(r, window)
    _, ops = _align(r, h)

    b_err = b_tot = n_err = n_tot = 0
    for op, ri in ops:
        if ri is None:            # insertion -- attribute to neighbouring ref pos
            continue
        is_b = ri in bidx
        if is_b:
            b_tot += 1
            if op != "C":
                b_err += 1
        else:
            n_tot += 1
            if op != "C":
                n_err += 1
    return {
        "cs_wer": (b_err / b_tot) if b_tot else 0.0,
        "boundary_tokens": b_tot,
        "non_boundary_wer": (n_err / n_tot) if n_tot else 0.0,
        "cs_penalty": ((b_err / b_tot) - (n_err / n_tot)) if (b_tot and n_tot) else 0.0,
    }


# ---------------------------------------------------------------------------
# Clinical entity recall + negation fidelity
# ---------------------------------------------------------------------------
CLINICAL_TERMS = {
    "fever": [r"fever", r"body dey hot", r"hot body", r"zazzab", r"ib[àa]", r"temperature"],
    "cough": [r"cough", r"tari", r"ik[ọo]"],
    "chest_pain": [r"chest pain", r"chest dey pain", r"ciwon kirji", r"[àa]y[àa]"],
    "headache": [r"headache", r"head dey pain", r"ciwon kai", r"or[íi] .{0,6}f[ọo]"],
    "abdominal_pain": [r"stomach", r"belle", r"abdomen", r"ciwon ciki", r"in[úu]"],
    "vomiting": [r"vomit", r"throw up", r"amai", r"[èe]b[ìi]"],
    "diarrhoea": [r"diarrh", r"running stomach", r"run belle", r"gudawa"],
    "breathlessness": [r"breath", r"numfashi", r"m[íi]m[íi]"],
    "bleeding": [r"bleed", r"blood", r"jini", r"[ẹe]j[ẹe]"],
    "dizziness": [r"dizz", r"head dey turn", r"jiri"],
    "weakness": [r"weak", r"no strength", r"body no gree", r"[àa]il[ệe]ra"],
    "convulsion": [r"convuls", r"seizure", r"fitting", r"giri"],
}


def extract_terms(text: str) -> set:
    low = normalise(text)
    hits = set()
    for term, pats in CLINICAL_TERMS.items():
        for p in pats:
            if re.search(p, low):
                hits.add(term)
                break
    return hits


def clinical_recall(reference: str, hypothesis: str) -> Dict:
    ref_terms, hyp_terms = extract_terms(reference), extract_terms(hypothesis)
    if not ref_terms:
        return {"recall": None, "precision": None, "missed": [], "spurious": list(hyp_terms)}
    tp = ref_terms & hyp_terms
    return {
        "recall": len(tp) / len(ref_terms),
        "precision": (len(tp) / len(hyp_terms)) if hyp_terms else 0.0,
        "missed": sorted(ref_terms - hyp_terms),
        "spurious": sorted(hyp_terms - ref_terms),
    }


def negation_fidelity(reference: str, hypothesis: str) -> Dict:
    """Did a negated symptom survive as negated? Flipping 'no fever' into
    'fever' is a safety event, not a WER point."""
    from triage_rules import _is_negated, NEGATION_CUES, CLAUSE_BREAK

    def _negated_after(text: str, match_end: int, window: int = 3) -> bool:
        """Pidgin/Yoruba negation often follows the noun: 'belle no dey pain'.
        Look a few tokens FORWARD from the match, stopping at a clause break."""
        suffix = text[match_end:]
        brk = CLAUSE_BREAK.search(suffix)
        if brk:
            suffix = suffix[: brk.start()]
        toks = re.findall(r"\S+", suffix)[:window]
        win = " ".join(toks).lower()
        return any(re.search(cue, win) for cue in NEGATION_CUES)

    def negated_terms(text: str) -> set:
        low = normalise(text)
        out = set()
        for term, pats in CLINICAL_TERMS.items():
            if term in out:
                continue
            for p in pats:
                for m in re.finditer(p, low):
                    if _is_negated(low, m.start()) or _negated_after(low, m.end()):
                        out.add(term)
                        break
                if term in out:
                    break
        return out

    ref_neg, hyp_neg = negated_terms(reference), negated_terms(hypothesis)
    hyp_pos = extract_terms(hypothesis) - hyp_neg
    flipped = sorted(ref_neg & hyp_pos)   # was negated, came out asserted
    return {
        "negations_in_ref": sorted(ref_neg),
        "flipped_to_positive": flipped,
        "flip_count": len(flipped),
    }


# ---------------------------------------------------------------------------
# Downstream decision agreement -- the headline metric
# ---------------------------------------------------------------------------
TIER_RANK = {"GREEN": 0, "YELLOW": 1, "ORANGE": 2, "RED": 3}


def triage_agreement(gold_tier: str, pred_tier: str) -> Dict:
    g, p = TIER_RANK.get(gold_tier, 0), TIER_RANK.get(pred_tier, 0)
    return {
        "exact": g == p,
        "delta": p - g,
        "under_triaged": p < g,      # the dangerous direction
        "over_triaged": p > g,       # costly but safe
    }


def summarise(rows: List[Dict]) -> Dict:
    """Aggregate per-utterance rows into the table for your report."""
    def _avg(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    agent_rows = [r for r in rows if "exact" in r]
    na = len(agent_rows)
    return {
        "n": len(rows),
        "wer": _avg("wer"),
        "cs_boundary_wer": _avg("cs_wer"),
        "non_boundary_wer": _avg("non_boundary_wer"),
        "cs_penalty": _avg("cs_penalty"),
        "clinical_recall": _avg("clinical_recall"),
        "negation_flips": sum(r.get("flip_count", 0) for r in rows),
        "triage_n": na,
        "triage_exact_match": round(sum(1 for r in agent_rows if r["exact"]) / na, 4) if na else None,
        "under_triage_rate": round(sum(1 for r in agent_rows if r.get("under_triaged")) / na, 4) if na else None,
        "mean_latency_s": _avg("latency_s"),
    }
