"""
Quick test: qwen3.5-9b with enable_thinking=True, 20 samples, show raw outputs.
"""
import json, random, sys
from openai import OpenAI

API_KEY = "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"
API_BASE = "https://openrouter.ai/api/v1"
MODEL = "qwen/qwen3.5-9b"
DATA_FILE = "../neutralizing-biased-phrase/src/bias_data/WNC/biased.full.test"
N = 2
SEED = 42

client = OpenAI(api_key=API_KEY, base_url=API_BASE)

random.seed(SEED)
lines = open(DATA_FILE, encoding="utf-8").readlines()
samples = random.sample(lines, N)

ce_system = (
    "Task: Rewrite the biased input sentence to neutral point-of-view.\n"
    "You are processing Wikipedia sentences that are known to contain biased language. "
    "Your job is to neutralize the bias while preserving all factual content.\n"
    "Rules:\n"
    "1) Preserve factual meaning and named entities.\n"
    "2) Remove subjective, loaded, inflammatory, or opinionated wording.\n"
    "3) Keep edits minimal and targeted; do not add new claims.\n"
    "4) Output exactly one rewritten sentence.\n"
    "5) Return only sentence text. No prefaces, no explanations, no quotes, no tags."
)

ok = 0
noop = 0
for i, line in enumerate(samples):
    parts = line.strip().split("\t")
    if len(parts) < 4:
        continue
    src = parts[3].strip()  # original biased sentence (un-tokenized)
    messages = [
        {"role": "system", "content": ce_system},
        {"role": "user", "content": f"Sentence: {src}"}
    ]
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.6,
            max_tokens=3000,
            extra_body={
                "provider": {"allow_fallbacks": True},
            },
            stream=False,
        )
        msg = resp.choices[0].message
        content = msg.content or ""
        # Check for reasoning/thinking content in alternate fields
        reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None) or ""
        if i < 1:
            print(f"[{i+1:02d}] content={repr(content)}")
            print(f"  full_reasoning:\n{str(reasoning)}\n---END---")
        pred = content.strip()
        changed = (pred.lower() != src.lower())
        status = "[CHANGED]" if changed else "[NOOP]"
        if changed:
            ok += 1
        else:
            noop += 1
        print(f"[{i+1:02d}] {status}")
        print(f"  SRC : {src[:100]}")
        print(f"  PRED: {pred[:100]}")
        print()
    except Exception as e:
        print(f"[{i+1:02d}] ERROR: {e}")
        print()

print(f"CHANGED: {ok}/{N}  NOOP: {noop}/{N}")
