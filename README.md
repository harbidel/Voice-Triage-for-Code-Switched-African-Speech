# Voice Triage for Code-Switched African Speech

A voice **agent** — not a transcriber. Patient speaks naturally in Nigerian Pidgin,
Yoruba, Hausa or English; the system extracts a structured intake record, asks
follow-up questions for missing fields, and assigns a **SATS** triage tier with a
deterministic safety floor.

## Why this exists
Nurses at clinic intake re-ask the same questions across a language gap, by hand,
all day. ASR alone doesn't fix that — a transcript still needs a human to turn it
into a decision. This closes that loop.

## Architecture

audio -> [asr_sahara / asr_comparators] -> transcript
     -> [triage_agent: LLM extraction -> IntakeRecord]
     -> [triage_rules: red flags + TEWS, escalate-only]
     -> SATS tier + disposition + follow-up question

All files sit in the repo root (flat layout) because the Hugging Face web
uploader does not reliably preserve subdirectories.

The extraction layer is **identical across all ASR models**, so benchmark
differences are attributable to ASR, not to pipeline differences.

## Safety design
- Rules **escalate only**. The LLM can raise a tier, never lower it below the
  rule floor. Hallucinated calm is the failure mode that hurts people.
- Negation scope stops at clause boundaries, so "I no fit breathe, my chest dey
  pain me" does not silently suppress the chest-pain flag.
- Every output is marked as requiring clinician confirmation.

## Setup
```bash
pip install -r requirements.txt
export INTRON_API_KEY=...      # voice.intron.io → Developer tab
# extraction layer -- pick ONE provider:
export ANTHROPIC_API_KEY=...                        # Claude (default)
# export OPENAI_API_KEY=...                         # OpenAI
# export LLM_PROVIDER=compatible \
#   OPENAI_BASE_URL=https://api.groq.com/openai/v1 \
#   OPENAI_API_KEY=... LLM_MODEL=llama-3.3-70b-versatile   # Groq (free tier)
python app.py
```
On HF Spaces set both as **Secrets**, never as plain variables.

## Benchmark
```bash
python run_bench.py --manifest eval.jsonl --out report.md
```
Reports WER, code-switch-boundary WER, CS penalty, clinical entity recall,
negation flips, and downstream triage agreement (incl. under-triage rate).

## Known limitations
- The multilingual red-flag lexicon is a prototype assembled without native-speaker
  clinical validation. It must be reviewed before any real deployment.
- Language tagging for CS-boundary WER uses a lexicon heuristic, not a trained LID.
- Sahara sync endpoint caps audio at 120s and 30 req/min.

## Demo link
https://huggingface.co/spaces/Harbidel/Voice-Triage
