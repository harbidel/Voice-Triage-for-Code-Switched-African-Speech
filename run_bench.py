"""Benchmark harness.

Manifest = JSONL, one line per utterance:
  {"audio":"data/clips/001.wav","reference":"my belle dey pain me since two days",
   "gold_tier":"YELLOW","lang":"en","source":"afriswitch"}

Usage:
  python -m bench.run_bench --manifest eval.jsonl --out report.md --limit 50
"""

import argparse, json, os, sys, time, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from asr_comparators import build_registry
import bench_metrics as M
from triage_agent import extract
from triage_rules import assess


def load(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def evaluate(models, items, use_agent=True, rate_delay=2.1):
    """rate_delay keeps Sahara under its 30 req/min ceiling."""
    results = {m.name: [] for m in models}
    for i, it in enumerate(items, 1):
        ref, gold = it["reference"], it.get("gold_tier", "GREEN")
        for m in models:
            row = {}
            try:
                tr = m.transcribe(it["audio"], language_hint=it.get("lang"))
                hyp, row["latency_s"] = tr.text, tr.latency_s
                row["wer"] = M.wer(ref, hyp)
                row.update(M.codeswitch_boundary_wer(ref, hyp))
                cr = M.clinical_recall(ref, hyp)
                row["clinical_recall"] = cr["recall"]
                row["missed_terms"] = cr["missed"]
                row.update(M.negation_fidelity(ref, hyp))
                row["hyp"] = hyp
                if use_agent:
                    # A dead LLM key must not vaporise the ASR metrics we
                    # already computed for this row.
                    try:
                        rec, sug = extract(hyp, m.name)
                        row.update(M.triage_agreement(gold, assess(rec, sug).tier.value))
                    except Exception as e:
                        row["agent_error"] = f"{type(e).__name__}: {e}"
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
                traceback.print_exc()
            results[m.name].append(row)
            if "sahara" in m.name.lower():
                time.sleep(rate_delay)
        print(f"  [{i}/{len(items)}] done", flush=True)
    return results


def render(results, items):
    out = ["# Benchmark Report — Code-Switched Clinical Speech", ""]
    out.append(f"Utterances: **{len(items)}**  |  Models: **{len(results)}**")
    out.append("")
    out.append("## Headline table")
    out.append("")
    out.append("| Model | WER ↓ | CS-boundary WER ↓ | CS penalty ↓ | Clinical recall ↑ | Negation flips ↓ | Triage exact ↑ | **Under-triage ↓** | Latency |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    summaries = {}
    for name, rows in results.items():
        s = M.summarise([r for r in rows if "error" not in r])
        summaries[name] = s
        out.append(
            f"| {name} | {s['wer']} | {s['cs_boundary_wer']} | {s['cs_penalty']} | "
            f"{s['clinical_recall']} | {s['negation_flips']} | {s['triage_exact_match']} | "
            f"**{s['under_triage_rate']}** | {s['mean_latency_s']}s |"
        )
    out += [
        "",
        "**Under-triage rate is the safety metric.** Over-triage costs a clinician's time; "
        "under-triage costs a patient. Rank on that column, not on WER.",
        "",
        "**CS penalty** = boundary WER minus non-boundary WER. It isolates how much of a "
        "model's error is caused specifically by code-switching rather than general difficulty.",
        "",
        "## WER vs. clinical usefulness",
        "",
    ]
    for name, s in summaries.items():
        if s["wer"] is not None and s["clinical_recall"] is not None:
            out.append(
                f"- **{name}**: WER {s['wer']} but clinical recall {s['clinical_recall']} "
                f"and {s['negation_flips']} negation flips."
            )
    out += ["", "## Failure examples", ""]
    for name, rows in results.items():
        bad = sorted(
            [r for r in rows if r.get("under_triaged") or r.get("flip_count", 0)],
            key=lambda r: -r.get("flip_count", 0),
        )[:3]
        if bad:
            out.append(f"### {name}")
            for r in bad:
                out.append(f"- Missed `{r.get('missed_terms')}`, flipped `{r.get('flipped_to_positive')}`")
                out.append(f"  - hyp: {str(r.get('hyp',''))[:160]}")
            out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="report.md")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--heavy", action="store_true", help="include the large HF model")
    ap.add_argument("--no-agent", action="store_true", help="skip triage stage (ASR metrics only)")
    a = ap.parse_args()

    models = build_registry(include_heavy=a.heavy)
    if not models:
        sys.exit("No ASR models available. Set INTRON_API_KEY / install faster-whisper.")
    print("Models:", ", ".join(m.name for m in models))

    items = load(a.manifest, a.limit)
    results = evaluate(models, items, use_agent=not a.no_agent)

    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(render(results, items))
    with open(a.out.replace(".md", ".json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
