"""Gradio app — voice triage & intake agent for code-switched African speech.

Tabs:
  1. Triage      — audio in, structured record + SATS tier out
  2. Compare     — same audio through all 3 ASR models, side by side
  3. Benchmark   — run the eval set, produce the report table
"""

import json
import os
import gradio as gr

from asr_comparators import build_registry
from triage_agent import run as run_agent
from triage_schema import Tier

TIER_COLOR = {
    "RED": "#c0392b", "ORANGE": "#e67e22",
    "YELLOW": "#f1c40f", "GREEN": "#27ae60", "BLUE": "#3498db",
}

MODELS = build_registry(include_heavy=False)
MODEL_NAMES = [m.name for m in MODELS] or ["<no model available>"]


def _card(res) -> str:
    c = TIER_COLOR.get(res.tier.value, "#777")
    flags = "".join(f"<li>{f.replace('_',' ')}</li>" for f in res.red_flags) or "<li>none detected</li>"
    reasons = "".join(f"<li>{r}</li>" for r in res.reasons)
    return f"""
<div style="border-left:10px solid {c};padding:14px 18px;background:#1c1c1c;border-radius:8px">
  <div style="font-size:30px;font-weight:700;color:{c}">{res.tier.value}</div>
  <div style="color:#ddd;margin:6px 0 12px">{res.disposition}</div>
  <b style="color:#aaa">Red flags</b><ul style="color:#ddd;margin:4px 0">{flags}</ul>
  <b style="color:#aaa">Why</b><ul style="color:#ddd;margin:4px 0">{reasons}</ul>
  <div style="margin-top:10px;padding:8px;background:#2a2a2a;border-radius:6px;color:#f39c12;font-size:13px">
    ⚠ Decision support only. A qualified clinician must confirm every tier.
  </div>
</div>"""


def triage(audio_path, model_name, lang):
    if not audio_path:
        return "Upload or record audio first.", "", ""
    model = next((m for m in MODELS if m.name == model_name), None)
    if model is None:
        return "No ASR model available. Set INTRON_API_KEY or install faster-whisper.", "", ""
    try:
        tr = model.transcribe(audio_path, language_hint=lang or None)
    except Exception as e:
        return f"ASR failed: {e}", "", ""
    try:
        rec, res, q = run_agent(tr.text, asr_model=tr.model_name)
    except Exception as e:
        return tr.text, f"Extraction failed: {e}", ""
    follow = f"**Next question:** {q}" if q else "**Intake complete** — no further questions."
    return (
        f"**[{tr.model_name}]** ({tr.latency_s}s)\n\n{tr.text}",
        _card(res),
        follow + "\n\n```json\n" + json.dumps(rec.to_dict(), indent=2, ensure_ascii=False) + "\n```",
    )


def compare(audio_path, lang):
    if not audio_path:
        return "Upload audio first."
    rows = ["| Model | Latency | Tier | Transcript |", "|---|---|---|---|"]
    for m in MODELS:
        try:
            tr = m.transcribe(audio_path, language_hint=lang or None)
            _, res, _ = run_agent(tr.text, asr_model=m.name)
            tier = res.tier.value
            txt = tr.text[:180].replace("|", "\\|").replace("\n", " ")
            rows.append(f"| {m.name} | {tr.latency_s}s | **{tier}** | {txt} |")
        except Exception as e:
            rows.append(f"| {m.name} | — | ERROR | {str(e)[:120]} |")
    rows.append("")
    rows.append("_Differing tiers across models on the same audio is the finding worth reporting._")
    return "\n".join(rows)


with gr.Blocks(title="Voice Triage — Code-Switched African Speech") as demo:
    gr.Markdown(
        "# 🩺 Voice Triage for Code-Switched African Speech\n"
        "Speak naturally in Pidgin, Yoruba, Hausa or English. The agent extracts a "
        "structured intake record, asks for what's missing, and assigns a **SATS** triage tier.\n\n"
        "*Prototype. Decision support only — never a substitute for a clinician.*"
    )

    with gr.Tab("Triage"):
        with gr.Row():
            with gr.Column():
                a1 = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Patient audio (≤120s)")
                m1 = gr.Dropdown(MODEL_NAMES, value=MODEL_NAMES[0], label="ASR model")
                l1 = gr.Textbox(value="en", label="Language hint (en, sw, yo, ha…)")
                b1 = gr.Button("Run triage", variant="primary")
            with gr.Column():
                o_card = gr.HTML(label="Triage")
        o_tr = gr.Markdown(label="Transcript")
        o_rec = gr.Markdown(label="Record")
        b1.click(triage, [a1, m1, l1], [o_tr, o_card, o_rec])

    with gr.Tab("Compare models"):
        a2 = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio")
        l2 = gr.Textbox(value="en", label="Language hint")
        b2 = gr.Button("Run all models", variant="primary")
        o2 = gr.Markdown()
        b2.click(compare, [a2, l2], o2)

    with gr.Tab("Benchmark"):
        gr.Markdown(
            "Run offline for the report:\n\n"
            "```bash\npython run_bench.py --manifest eval.jsonl --out report.md\n```\n\n"
            "Reports WER, code-switch-boundary WER, clinical entity recall, "
            "negation flips, and downstream triage agreement."
        )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
