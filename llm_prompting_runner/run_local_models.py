import argparse
import csv
import json
import os
import random
import re
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sacrebleu import corpus_bleu
from sentence_transformers import SentenceTransformer, util
from tokenizers import BertWordPieceTokenizer
from tqdm import tqdm

try:
    from openai import OpenAI
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Failed to import `openai`. Install it with: uv pip install openai"
    ) from exc

try:
    from bert_score import score as bertscore_score

    HAS_BERTSCORE = True
except Exception:
    HAS_BERTSCORE = False

try:
    from nltk.translate.meteor_score import meteor_score

    HAS_METEOR = True
except Exception:
    HAS_METEOR = False


SCRIPT_DIR = Path(__file__).resolve().parent

# Global token/cost tracking (thread-safe)
_token_stats: Dict[str, int] = {"input": 0, "output": 0}
_stats_lock = threading.Lock()

# Reasoning budget: if > 0, passed as reasoning.max_tokens to OpenRouter
_reasoning_budget: int = 0

# Budget watchdog (real-time cost monitoring)
_budget_exceeded: bool = False
_budget_lock = threading.Lock()
_cost_baseline: float = 0.0          # key usage (USD) at experiment start
_samples_done_budget: int = 0        # how many samples completed so far
_est_cost_per_sample: float = 0.0    # user-provided estimate (0 = disabled)
_max_budget: float = 0.0             # absolute USD cap (0 = disabled)
_budget_api_key: str = ""
_budget_api_base: str = ""
_BUDGET_CHECK_EVERY: int = 10        # check after every N completions
_BUDGET_TOLERANCE: float = 0.30      # allow up to 30% over estimate
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "neutralizing-biased-phrase" / "src"
BERT_VOCAB_PATH = SRC_DIR / "bias_data" / "bert.vocab"

import sys

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tagging.model import BiasedPhraseTagger  # noqa: E402

warnings.filterwarnings(
    "ignore",
    message="The given NumPy array is not writable",
    module=r"bert_score\.score",
)
warnings.filterwarnings(
    "ignore",
    message=r"Some weights of .* were not initialized from the model checkpoint.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"Some weights of .* were not used when initializing.*",
)
try:
    from transformers.utils import logging as hf_logging  # type: ignore

    hf_logging.set_verbosity_error()
except Exception:
    pass


def safe_name(value: str) -> str:
    # Make provider/model slugs filesystem-safe for output filenames.
    return re.sub(r'[\\/:*?"<>|]+', "_", value)


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def detok(tokens: List[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r'\s+"', '"', text)
    text = re.sub(r'"\s+', '" ', text)
    return text.strip()


def detok_wordpiece(tokens: List[str]) -> str:
    if not tokens:
        return ""
    out = []
    for tok in tokens:
        if tok.startswith("##") and out:
            out[-1] = out[-1] + tok[2:]
        else:
            out.append(tok)
    return detok(out)


def wrap_bias_spans(tokens: List[str], tags: List[str]) -> str:
    out = []
    in_bias = False
    for tok, tag in zip(tokens, tags):
        if tag == "B":
            if in_bias:
                out.append("</bias>")
            out.append("<bias>")
            out.append(tok)
            in_bias = True
        elif tag == "I":
            out.append(tok)
        else:
            if in_bias:
                out.append("</bias>")
                in_bias = False
            out.append(tok)
    if in_bias:
        out.append("</bias>")
    return detok_wordpiece(out)


def extract_bias_spans(tokens: List[str], tags: List[str]) -> List[List[str]]:
    spans: List[List[str]] = []
    current: List[str] = []
    for tok, tag in zip(tokens, tags):
        if tag == "B":
            if current:
                spans.append(current)
            current = [tok]
        elif tag == "I":
            if current:
                current.append(tok)
            else:
                current = [tok]
        else:
            if current:
                spans.append(current)
                current = []
    if current:
        spans.append(current)
    return spans


def has_subsequence(haystack: List[str], needle: List[str]) -> bool:
    if not needle:
        return False
    n = len(needle)
    if n > len(haystack):
        return False
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return True
    return False


def bias_phrase_retention(
    src_tokens: List[str], src_tags: List[str], pred_tokens: List[str]
) -> Tuple[int, int, float]:
    spans = extract_bias_spans(src_tokens, src_tags)
    total = len(spans)
    if total == 0:
        return 0, 0, 0.0
    retained = sum(1 for span in spans if has_subsequence(pred_tokens, span))
    return total, retained, retained / total


@dataclass
class Example:
    idx: str
    source: str
    target: str
    source_wnc_tokens: List[str]


@dataclass
class PreparedExample:
    idx: str
    source: str
    target: str
    src_tokens: List[str]
    src_tags: List[str]
    src_bias_indices: set
    src_bias_count: int
    src_bias_density: float
    tagged_source_for_prompt: str
    src_plain_tokens: List[str]
    ref_plain_tokens: List[str]
    ref_edit_rate: float


class WordPieceAdapter:
    def __init__(self, vocab_path: Path):
        # Use packaged tokenizer implementation for compatibility and stability.
        self.tokenizer = BertWordPieceTokenizer(
            vocab=str(vocab_path),
            lowercase=True,
        )

    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.encode(text, add_special_tokens=False).tokens


class LocalBiasTagger:
    def __init__(self, ckpt_path: Path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=device)
        self.tok2id = ckpt["tok2id"]
        self.id2label = {v: k for k, v in ckpt["label2id"].items()}
        self.device = device
        self.model = BiasedPhraseTagger(
            vocab_size=len(self.tok2id),
            embedding_dim=128,
            hidden_dim=256,
            num_labels=len(self.id2label),
        ).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def predict_tags(self, tokens: List[str]) -> List[str]:
        if not tokens:
            return []
        ids = [self.tok2id.get(t, self.tok2id["<unk>"]) for t in tokens]
        input_ids = torch.tensor([ids], dtype=torch.long).to(self.device)
        lengths = torch.tensor([len(ids)], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(input_ids, lengths)
            pred = torch.argmax(logits, dim=-1).squeeze(0).cpu().tolist()
        return [self.id2label[p] for p in pred[: len(tokens)]]

    def bias_indices(self, tags: List[str]) -> set:
        return {i for i, t in enumerate(tags) if t in {"B", "I"}}


def load_wnc_rows(path: Path) -> List[Example]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            idx = parts[0]
            source_wnc_tokens = parts[1].strip().split() if len(parts) > 1 else []
            source = parts[3] if len(parts) > 4 else parts[1]
            target = parts[4] if len(parts) > 4 else parts[2]
            source = source.strip()
            target = target.strip()
            if source and target and source_wnc_tokens:
                rows.append(Example(idx=idx, source=source, target=target, source_wnc_tokens=source_wnc_tokens))
    return rows


def load_wnc(path: Path, max_examples: int, seed: int) -> List[Example]:
    rows = load_wnc_rows(path)

    rng = random.Random(seed)
    rng.shuffle(rows)

    eval_count = min(max_examples, len(rows))
    return rows[:eval_count]


def build_messages(
    strategy: str,
    prepared_ex: PreparedExample,
    few_shots: List[Example],
    few_shot_tagged_sources: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    rewrite_system = (
        "Task: Rewrite the biased input sentence to neutral point-of-view.\n"
        "Rules:\n"
        "1) Preserve factual meaning and named entities.\n"
        "2) Remove subjective, loaded, inflammatory, or opinionated wording.\n"
        "3) Keep edits minimal and targeted; do not add new claims.\n"
        "4) Keep dates, numbers, and named entities unchanged unless grammar requires it.\n"
        "5) Output exactly one rewritten sentence.\n"
        "6) Return only sentence text. No prefaces (e.g., 'Here is'), no explanations, no bullets, no quotes.\n"
        "7) Never output XML-like tags such as <bias> or </bias>."
    )

    if strategy == "npov":
        return [{"role": "system", "content": rewrite_system}, {"role": "user", "content": prepared_ex.source}]

    if strategy == "with_bias_tags":
        user = (
            "Rewrite to neutral language while preserving factual meaning. "
            "Tokens inside <bias>...</bias> are likely biased and must be neutralized first. "
            "The tags are hints only; do not output any tags. "
            "Do not add new facts. Return exactly one sentence.\n\n"
            f"Sentence: {prepared_ex.tagged_source_for_prompt}"
        )
        return [{"role": "system", "content": rewrite_system}, {"role": "user", "content": user}]

    if strategy in ("context_enriched", "context_enriched_soft", "context_enriched_constrained"):
        if strategy == "context_enriched_soft":
            rule3 = (
                "3) Tokens inside <bias>...</bias> are likely biased. Prefer replacing them with neutral alternatives. "
                "If surrounding words need grammatical adjustment, make only the smallest possible additional change.\n"
            )
        elif strategy == "context_enriched_constrained":
            rule3 = (
                "3) ONLY change the words inside <bias>...</bias> tags. Replace each biased word or phrase with a "
                "neutral equivalent. Do NOT change, add, or remove any other words in the sentence.\n"
            )
        else:
            rule3 = (
                "3) Tokens inside <bias>...</bias> are biased and must be removed or replaced with neutral alternatives. "
                "Do not keep any marked token unchanged.\n"
            )
        ce_system = (
            "Task: Rewrite the biased input sentence to neutral point-of-view.\n"
            "You are processing Wikipedia sentences that are known to contain biased language. "
            "Your job is to neutralize the bias while preserving all factual content.\n"
            "Rules:\n"
            "1) Preserve factual meaning and named entities.\n"
            "2) Remove subjective, loaded, inflammatory, or opinionated wording.\n"
            + rule3 +
            "4) Keep edits minimal and targeted; do not add new claims.\n"
            "5) Keep dates, numbers, and named entities unchanged unless grammar requires it.\n"
            "6) Output exactly one rewritten sentence.\n"
            "7) Return only sentence text. No prefaces, no explanations, no bullets, no quotes, no tags."
        )
        messages = [{"role": "system", "content": ce_system}]
        tagged_sources = few_shot_tagged_sources or []
        for i, ex in enumerate(few_shots):
            if i < len(tagged_sources):
                messages.append({"role": "user", "content": f"Sentence: {tagged_sources[i]}"})
            else:
                messages.append({"role": "user", "content": ex.source})
            messages.append({"role": "assistant", "content": ex.target})
        messages.append({"role": "user", "content": f"Sentence: {prepared_ex.tagged_source_for_prompt}"})
        return messages

    if strategy == "few_shot":
        messages = [
            {
                "role": "system",
                "content": rewrite_system,
            }
        ]
        for ex in few_shots:
            messages.append({"role": "user", "content": ex.source})
            messages.append({"role": "assistant", "content": ex.target})
        messages.append({"role": "user", "content": prepared_ex.source})
        return messages

    if strategy == "few_shot_flat":
        parts = [rewrite_system, ""]
        for ex in few_shots:
            parts.append(f"Original: {ex.source}\nNeutral: {ex.target}\n")
        parts.append(f"Now rewrite:\n{prepared_ex.source}")
        flat_prompt = "\n".join(parts)
        return [{"role": "user", "content": flat_prompt}]

    return [
        {
            "role": "system",
            "content": rewrite_system,
        },
        {"role": "user", "content": prepared_ex.source},
    ]


def get_key_usage(api_key: str, api_base: str) -> float:
    """Query OpenRouter for cumulative USD usage of this API key."""
    import requests as _requests
    resp = _requests.get(
        f"{api_base}/key",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=15,
    )
    resp.raise_for_status()
    return float(resp.json()["data"]["usage"])


def check_budget_watchdog() -> None:
    """Called every _BUDGET_CHECK_EVERY completions. Sets _budget_exceeded if cost overruns."""
    global _budget_exceeded
    if _est_cost_per_sample <= 0 or not _budget_api_key:
        return
    # Read n outside the lock to avoid holding lock during HTTP call
    with _budget_lock:
        n = _samples_done_budget
    if n == 0:
        return
    try:
        current_usage = get_key_usage(_budget_api_key, _budget_api_base)
    except Exception as e:
        print(f"\n[BUDGET] WARNING: Could not query key usage: {e}")
        return
    actual_per_sample = (current_usage - _cost_baseline) / n
    limit = _est_cost_per_sample * (1 + _BUDGET_TOLERANCE)
    status = (
        f"[BUDGET] n={n}  baseline=${_cost_baseline:.4f}  current=${current_usage:.4f}"
        f"  actual/sample=${actual_per_sample:.5f}  estimated/sample=${_est_cost_per_sample:.5f}"
        f"  limit=${limit:.5f}"
    )
    print(f"\n{status}")
    total_spent = current_usage - _cost_baseline
    if _max_budget > 0 and total_spent >= _max_budget:
        _budget_exceeded = True
        print(
            f"[BUDGET CAP HIT] spent=${total_spent:.4f} >= cap=${_max_budget:.2f}. STOPPING."
        )
    elif actual_per_sample > limit:
        _budget_exceeded = True
        print(
            f"[BUDGET EXCEEDED] actual/sample=${actual_per_sample:.5f} > limit=${limit:.5f}"
            f" (>{_BUDGET_TOLERANCE*100:.0f}% over estimate). STOPPING."
        )


def call_openai_chat(
    client: "OpenAI",
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    request_timeout: float,
    allow_fallbacks: bool,
) -> str:
    if _budget_exceeded:
        raise RuntimeError("[BUDGET EXCEEDED] Refusing new API call.")
    # max_tokens <= 0 means no limit (do not pass the parameter)
    # stream=False: for thinking models (e.g. qwen3.5-flash), thinking tokens go to
    # message.reasoning separately; max_tokens only limits the visible content.
    extra_body: Dict = {"provider": {"allow_fallbacks": allow_fallbacks}}
    if _reasoning_budget > 0:
        extra_body["reasoning"] = {"max_tokens": _reasoning_budget}
    elif _reasoning_budget == -1:
        extra_body["reasoning"] = {"effort": "none"}

    create_kwargs: Dict = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        timeout=request_timeout,
        extra_body=extra_body,
    )
    if max_tokens > 0:
        create_kwargs["max_tokens"] = max_tokens

    resp = client.chat.completions.create(**create_kwargs)
    if not resp.choices:
        raise RuntimeError(f"API returned empty choices (model={model}): {resp}")
    content = resp.choices[0].message.content or ""
    if resp.usage is not None:
        with _stats_lock:
            _token_stats["input"] += resp.usage.prompt_tokens or 0
            _token_stats["output"] += resp.usage.completion_tokens or 0
    return content.strip()


def call_openai_with_retry(
    client: "OpenAI",
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    request_timeout: float,
    allow_fallbacks: bool,
    retries: int = 3,
) -> str:
    delay = 1.0
    last_exc = None
    for _ in range(retries):
        try:
            return call_openai_chat(
                client,
                model,
                messages,
                temperature,
                max_tokens,
                request_timeout=request_timeout,
                allow_fallbacks=allow_fallbacks,
            )
        except Exception as exc:  # pragma: no cover
            last_exc = exc
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"OpenAI-compatible API call failed after {retries} attempts: {last_exc}")


def refine_once(
    client: "OpenAI",
    model: str,
    source: str,
    draft: str,
    temperature: float,
    max_tokens: int,
    request_timeout: float,
    allow_fallbacks: bool,
    retries: int,
) -> str:
    critique_prompt = (
        "Critique this rewrite for neutrality and meaning preservation. "
        "Keep critique concise in 1-2 sentences.\n\n"
        f"Source: {source}\nRewrite: {draft}"
    )
    critique = call_openai_with_retry(
        client,
        model,
        [{"role": "user", "content": critique_prompt}],
        temperature,
        max_tokens,
        request_timeout=request_timeout,
        allow_fallbacks=allow_fallbacks,
        retries=retries,
    )
    improve_prompt = (
        "Improve the rewrite using the critique. Return only the improved final sentence.\n\n"
        f"Source: {source}\nCurrent rewrite: {draft}\nCritique: {critique}"
    )
    return call_openai_with_retry(
        client,
        model,
        [{"role": "user", "content": improve_prompt}],
        temperature,
        max_tokens,
        request_timeout=request_timeout,
        allow_fallbacks=allow_fallbacks,
        retries=retries,
    )


def load_api_key(args: argparse.Namespace) -> str:
    if args.api_key:
        return args.api_key.strip()
    for name in ("DASHSCOPE_API_KEY", "OPENAI_API_KEY"):
        value = os.getenv(name, "").strip()
        if value:
            return value
    if args.api_key_file:
        p = Path(args.api_key_file).resolve()
        if p.exists():
            raw = p.read_text(encoding="utf-8").strip()
            if "=" in raw and "\n" not in raw:
                _, val = raw.split("=", 1)
                return val.strip().strip('"').strip("'")
            if raw:
                return raw
    raise RuntimeError(
        "No API key found. Use --api_key, set DASHSCOPE_API_KEY/OPENAI_API_KEY, "
        "or place key in --api_key_file."
    )


def normalize_prediction(text: str, source_fallback: str) -> str:
    s = (text or "").strip()
    if not s:
        return source_fallback

    # Remove markdown fences/labels if the model emits formatted output.
    s = s.replace("```", " ").replace("Rewritten:", " ").replace("Rewrite:", " ").strip()
    s = re.sub(
        r"^\s*(here is|the sentence|certainly|i apologize|it seems like|this sentence)\b[^:]*:\s*",
        "",
        s,
        flags=re.IGNORECASE,
    )
    s = " ".join(s.split())

    s = s.strip(" \"'`")
    return s if s else source_fallback


def is_garbled_output(text: str) -> bool:
    if not text:
        return True
    s = text.strip()
    if not s:
        return True
    # Frequent replacement characters/question-mark runs are usually corrupted output.
    if "�" in s:
        return True
    if re.search(r"\?{6,}", s):
        return True
    return False


def is_low_quality_rewrite(prediction: str, source: str) -> bool:
    s = (prediction or "").strip()
    if is_garbled_output(s):
        return True
    if re.match(
        r"^\s*(it seems like|here is|the sentence|certainly|i cannot|i apologize|without additional context|if you're|the phrase you provided)\b",
        s,
        flags=re.IGNORECASE,
    ):
        return True
    if "<bias>" in s or "</bias>" in s:
        return True
    src_len = len(source.strip())
    pred_len = len(s)
    if src_len >= 50 and pred_len < int(src_len * 0.55):
        return True
    if src_len >= 40 and pred_len > int(src_len * 2.2):
        return True
    return False


def edit_rate(a_tokens: List[str], b_tokens: List[str]) -> float:
    sm = SequenceMatcher(a=a_tokens, b=b_tokens)
    edits = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":
            edits += max(i2 - i1, j2 - j1)
        elif tag == "delete":
            edits += i2 - i1
        elif tag == "insert":
            edits += j2 - j1
    return edits / max(1, len(a_tokens))


def compute_metrics(
    references: List[str],
    predictions: List[str],
    semantic_model: SentenceTransformer,
) -> Dict[str, float]:
    bleu = corpus_bleu(predictions, [references]).score / 100.0
    token_acc = (
        sum(p.strip() == r.strip() for p, r in zip(predictions, references))
        / max(1, len(predictions))
    )

    ref_emb = semantic_model.encode(references, convert_to_tensor=True, show_progress_bar=False)
    pred_emb = semantic_model.encode(predictions, convert_to_tensor=True, show_progress_bar=False)
    sem = util.cos_sim(pred_emb, ref_emb).diagonal().mean().item()

    result = {
        "BLEU": round(bleu, 4),
        "Token-Level Accuracy": round(float(token_acc), 4),
        "SemanticSimilarity": round(float(sem), 4),
    }

    if HAS_BERTSCORE:
        _, _, f1 = bertscore_score(
            predictions,
            references,
            lang="en",
            verbose=False,
            rescale_with_baseline=True,
        )
        result["BERTScoreF1"] = round(float(f1.mean().item()), 4)
    else:
        result["BERTScoreF1"] = float("nan")

    if HAS_METEOR:
        meteor_vals = [
            meteor_score([simple_tokenize(ref)], simple_tokenize(pred))
            for ref, pred in zip(references, predictions)
        ]
        result["METEOR"] = round(float(np.mean(meteor_vals)), 4)
    else:
        result["METEOR"] = float("nan")

    return result


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data_path).resolve()
    tagger_ckpt = Path(args.tagger_ckpt).resolve()

    eval_rows = load_wnc(data_path, args.samples, args.seed)
    few_rows = load_wnc_rows(Path(args.few_shot_path).resolve())
    eval_ids = {ex.idx for ex in eval_rows}
    few_pool = [ex for ex in few_rows if ex.idx not in eval_ids]
    if len(few_pool) < args.few_shot_k:
        raise RuntimeError(
            f"Few-shot pool too small after excluding eval overlap: "
            f"need {args.few_shot_k}, got {len(few_pool)}."
        )
    rng = random.Random(args.seed + 1)
    rng.shuffle(few_pool)
    few_shots = few_pool[: args.few_shot_k]

    tagger = LocalBiasTagger(tagger_ckpt)

    # Pre-compute tagged sources for few-shot examples (used by context_enriched strategy)
    few_shot_tagged_sources: List[str] = []
    for ex in few_shots:
        fs_tags = tagger.predict_tags(ex.source_wnc_tokens)
        few_shot_tagged_sources.append(wrap_bias_spans(ex.source_wnc_tokens, fs_tags))
    sem_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    # Semantic similarity few-shot: pre-compute embeddings for pool + eval rows
    # (populated after prepared_eval_rows is built below)
    few_pool_embeddings = None
    few_pool_tagged_sources_all: List[str] = []
    eval_embeddings = None
    eval_emb_map: Dict[str, int] = {}
    wp_tokenizer = WordPieceAdapter(Path(args.bert_vocab).resolve())
    api_key = load_api_key(args)

    prepared_eval_rows: List[PreparedExample] = []
    for ex in eval_rows:
        # Use original WNC tokenization for source-side tagging compatibility.
        src_tokens = ex.source_wnc_tokens
        src_tags = tagger.predict_tags(src_tokens)
        src_bias_indices = tagger.bias_indices(src_tags)
        src_bias_count = len(src_bias_indices)
        src_bias_density = src_bias_count / max(1, len(src_tokens))
        tagged_source_for_prompt = wrap_bias_spans(src_tokens, src_tags)
        src_plain_tokens = simple_tokenize(ex.source.lower())
        ref_plain_tokens = simple_tokenize(ex.target.lower())
        ref_edit_rate = edit_rate(src_plain_tokens, ref_plain_tokens)
        prepared_eval_rows.append(
            PreparedExample(
                idx=ex.idx,
                source=ex.source,
                target=ex.target,
                src_tokens=src_tokens,
                src_tags=src_tags,
                src_bias_indices=src_bias_indices,
                src_bias_count=src_bias_count,
                src_bias_density=src_bias_density,
                tagged_source_for_prompt=tagged_source_for_prompt,
                src_plain_tokens=src_plain_tokens,
                ref_plain_tokens=ref_plain_tokens,
                ref_edit_rate=ref_edit_rate,
            )
        )

    if all(ex.src_bias_count == 0 for ex in prepared_eval_rows):
        raise RuntimeError(
            "Sanity check failed: all source_bias_count values are 0. "
            "Tagger/tokenization input is likely incompatible."
        )

    if args.sim_few_shot:
        # Limit pool size to avoid encoding all 160K+ training examples.
        sim_pool = few_pool[: args.sim_pool_size] if args.sim_pool_size > 0 else few_pool
        print(f"[sim_few_shot] Pre-computing embeddings. Pool={len(sim_pool)}, Eval={len(prepared_eval_rows)}")
        few_pool_sources = [ex.source for ex in sim_pool]
        few_pool_embeddings = sem_model.encode(
            few_pool_sources, convert_to_tensor=True, show_progress_bar=False
        )
        # Pre-compute tagged sources for sim_pool (safe: single-threaded, fast for 2000 examples)
        for ex in sim_pool:
            fs_tags = tagger.predict_tags(ex.source_wnc_tokens)
            few_pool_tagged_sources_all.append(wrap_bias_spans(ex.source_wnc_tokens, fs_tags))
        # Replace few_pool with the limited pool for generate_one to reference
        few_pool = sim_pool
        eval_sources = [ex.source for ex in prepared_eval_rows]
        eval_embeddings = sem_model.encode(
            eval_sources, convert_to_tensor=True, show_progress_bar=False
        )
        eval_emb_map = {ex.idx: i for i, ex in enumerate(prepared_eval_rows)}
        print(f"[sim_few_shot] Done.")

    summary_rows = []
    for model_name in args.models:
        for strategy in args.strategies:
            run_name = f"{safe_name(model_name)}__{strategy}"
            pred_path = out_dir / f"{run_name}.jsonl"

            refs = []
            preds = []
            bias_retention_vals = []
            bias_phrase_retention_vals = []
            over_edit_vals = []
            no_bias_reduction_count = 0
            thread_state = threading.local()

            with pred_path.open("w", encoding="utf-8") as fout:
                def generate_one(ex: PreparedExample) -> Tuple[PreparedExample, str]:
                    if not hasattr(thread_state, "client"):
                        thread_state.client = OpenAI(
                            api_key=api_key,
                            base_url=args.api_base,
                            default_headers={
                                "HTTP-Referer": "https://github.com/ntu-ai6130",
                                "X-Title": "AI6130-bias-neutralization",
                            },
                        )
                    local_client = thread_state.client
                    quality_retries = max(0, args.quality_retry_attempts)
                    if args.quality_retry_once:
                        quality_retries = max(quality_retries, 1)
                    pred_local = ex.source
                    for retry_idx in range(quality_retries + 1):
                        if args.sim_few_shot:
                            ex_i = eval_emb_map[ex.idx]
                            scores = util.cos_sim(eval_embeddings[ex_i : ex_i + 1], few_pool_embeddings)[0]
                            top_k = scores.argsort(descending=True)[: args.few_shot_k].tolist()
                            sim_shots = [few_pool[i] for i in top_k]
                            sim_tagged = [few_pool_tagged_sources_all[i] for i in top_k]
                            msgs = build_messages(strategy, ex, sim_shots, sim_tagged)
                        else:
                            msgs = build_messages(strategy, ex, few_shots, few_shot_tagged_sources)
                        if retry_idx > 0:
                            msgs = msgs + [
                                {
                                    "role": "user",
                                    "content": (
                                        "Output format correction: return exactly one rewritten sentence text only. "
                                        "No explanation, no assistant commentary, no list."
                                    ),
                                }
                            ]
                        try:
                            pred_local = call_openai_with_retry(
                                client=local_client,
                                model=model_name,
                                messages=msgs,
                                temperature=args.temperature,
                                max_tokens=args.max_tokens,
                                request_timeout=args.request_timeout,
                                allow_fallbacks=args.allow_fallbacks,
                                retries=args.retries,
                            )
                        except RuntimeError:
                            break  # fallback to ex.source (set above)
                        if strategy == "self_refine":
                            pred_local = refine_once(
                                client=local_client,
                                model=model_name,
                                source=ex.source,
                                draft=pred_local,
                                temperature=args.temperature,
                                max_tokens=args.max_tokens,
                                request_timeout=args.request_timeout,
                                allow_fallbacks=args.allow_fallbacks,
                                retries=args.retries,
                            )
                        pred_local = normalize_prediction(pred_local, ex.source)
                        if not is_low_quality_rewrite(pred_local, ex.source):
                            break
                    if is_low_quality_rewrite(pred_local, ex.source):
                        pred_local = ex.source
                    return ex, pred_local

                def process_result(ex: PreparedExample, pred: str) -> None:
                    nonlocal no_bias_reduction_count
                    pred_tokens = wp_tokenizer.tokenize(pred)
                    pred_tags = tagger.predict_tags(pred_tokens)
                    pred_bias = tagger.bias_indices(pred_tags)

                    src_bias_count = ex.src_bias_count
                    pred_bias_count = len(pred_bias)
                    src_bias_density = ex.src_bias_density
                    pred_bias_density = pred_bias_count / max(1, len(pred_tokens))
                    if src_bias_density > 0:
                        bias_retention = min(1.0, pred_bias_density / src_bias_density)
                        bias_retention_vals.append(bias_retention)
                    else:
                        bias_retention = 0.0

                    src_bias_phrase_count, retained_bias_phrase_count, bias_phrase_retention_value = bias_phrase_retention(
                        ex.src_tokens, ex.src_tags, pred_tokens
                    )
                    if src_bias_phrase_count > 0:
                        bias_phrase_retention_vals.append(bias_phrase_retention_value)

                    pred_plain_tokens = simple_tokenize(pred.lower())
                    pred_edit_rate = edit_rate(ex.src_plain_tokens, pred_plain_tokens)
                    over_edit_rate = max(0.0, pred_edit_rate - ex.ref_edit_rate)
                    over_edit_vals.append(over_edit_rate)

                    refs.append(ex.target)
                    preds.append(pred)

                    if src_bias_density > 0 and pred_bias_density >= src_bias_density:
                        no_bias_reduction_count += 1

                    fout.write(
                        json.dumps(
                            {
                                "id": ex.idx,
                                "source": ex.source,
                                "reference": ex.target,
                                "prediction": pred,
                                "source_bias_count": src_bias_count,
                                "prediction_bias_count": pred_bias_count,
                                "bias_retention_sample": round(bias_retention, 4),
                                "source_bias_phrase_count": src_bias_phrase_count,
                                "retained_bias_phrase_count": retained_bias_phrase_count,
                                "bias_phrase_retention_sample": round(bias_phrase_retention_value, 4),
                                "over_edit_rate_sample": round(over_edit_rate, 4),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    fout.flush()

                def process_result_with_budget(ex: "PreparedExample", pred: str) -> None:
                    global _samples_done_budget
                    process_result(ex, pred)
                    with _budget_lock:
                        _samples_done_budget += 1
                        n = _samples_done_budget
                    if n % _BUDGET_CHECK_EVERY == 0:
                        check_budget_watchdog()

                if args.parallel_requests > 1:
                    with ThreadPoolExecutor(max_workers=args.parallel_requests) as pool:
                        futures = {pool.submit(generate_one, row): row for row in prepared_eval_rows}
                        for fut in tqdm(
                            as_completed(futures),
                            total=len(prepared_eval_rows),
                            desc=run_name,
                            leave=True,
                        ):
                            if _budget_exceeded:
                                for f in futures:
                                    f.cancel()
                                print(f"\n[BUDGET] Stopped early. {len(preds)} samples saved.")
                                break
                            process_result_with_budget(*fut.result())
                else:
                    for ex, pred in tqdm(
                        map(generate_one, prepared_eval_rows),
                        total=len(prepared_eval_rows),
                        desc=run_name,
                        leave=True,
                    ):
                        process_result_with_budget(ex, pred)

            metrics = compute_metrics(refs, preds, sem_model)
            metrics["BiasRetentionRate"] = round(
                float(np.mean(bias_retention_vals)) if bias_retention_vals else 0.0, 4
            )
            metrics["BiasPhraseRetentionRate"] = round(
                float(np.mean(bias_phrase_retention_vals)) if bias_phrase_retention_vals else 0.0, 4
            )
            metrics["OverEditRate"] = round(float(np.mean(over_edit_vals)) if over_edit_vals else 0.0, 4)
            metrics["n"] = len(preds)
            metrics["model"] = model_name
            metrics["strategy"] = strategy
            metrics["NoBiasReductionCount"] = no_bias_reduction_count
            summary_rows.append(metrics)

            print(f"[done] {run_name} -> {pred_path}")

    summary_json = out_dir / "summary.json"
    summary_csv = out_dir / "summary.csv"
    summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSaved summary: {summary_json}")
    print(f"Saved summary: {summary_csv}")

    # Token usage report
    inp = _token_stats["input"]
    out = _token_stats["output"]
    total = inp + out
    print(f"\n[TOKEN USAGE] input={inp:,}  output={out:,}  total={total:,}")
    print(f"[NOTE] Cost depends on model pricing; check OpenRouter dashboard for actual charges.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run prompting experiments via OpenAI SDK (Alibaba Cloud OpenAI-compatible endpoint)."
    )
    parser.add_argument(
        "--data_path",
        default=str(REPO_ROOT / "neutralizing-biased-phrase" / "src" / "bias_data" / "WNC" / "biased.full.test"),
    )
    parser.add_argument(
        "--few_shot_path",
        default=str(REPO_ROOT / "neutralizing-biased-phrase" / "src" / "bias_data" / "WNC" / "biased.full.train"),
    )
    parser.add_argument(
        "--tagger_ckpt",
        default=str(REPO_ROOT / "neutralizing-biased-phrase" / "src" / "train_tagging" / "biased_phrase_tagger.ckpt"),
    )
    parser.add_argument("--output_dir", default=str(SCRIPT_DIR / "outputs"))
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["meta-llama/llama-3.1-8b-instruct"],
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["zero_shot", "few_shot", "with_bias_tags", "npov", "self_refine"],
        choices=["zero_shot", "few_shot", "few_shot_flat", "with_bias_tags", "npov", "self_refine", "context_enriched", "context_enriched_soft", "context_enriched_constrained"],
    )
    parser.add_argument(
        "--sim_few_shot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use semantic similarity to select per-example few-shot demonstrations "
            "(top-k most similar from the training pool). "
            "Default: fixed random k shots (seed-based)."
        ),
    )
    parser.add_argument(
        "--sim_pool_size",
        type=int,
        default=2000,
        help=(
            "Max number of training pool examples to use for similarity search "
            "(0 = use all). Default: 2000. Larger values are slower but may improve quality."
        ),
    )
    parser.add_argument("--few_shot_k", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=220)
    parser.add_argument("--api_base", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api_key", default="")
    parser.add_argument("--api_key_file", default=str(SCRIPT_DIR / ".local.env"))
    parser.add_argument("--bert_vocab", default=str(BERT_VOCAB_PATH))
    parser.add_argument("--parallel_requests", type=int, default=1)
    parser.add_argument("--request_timeout", type=float, default=60.0)
    parser.add_argument(
        "--allow_fallbacks",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--quality_retry_once",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--quality_retry_attempts", type=int, default=1)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--reasoning_budget",
        type=int,
        default=0,
        help="Max reasoning tokens for thinking models (0 = no limit). Passed as reasoning.max_tokens to OpenRouter.",
    )
    parser.add_argument(
        "--estimated_cost_per_sample",
        type=float,
        default=0.0,
        help=(
            "Expected USD cost per sample (from estimate_budget.py). "
            "If > 0, actual cost is checked every 10 samples via OpenRouter /api/v1/key. "
            "Experiment stops if actual/sample exceeds this by >30%%."
        ),
    )
    parser.add_argument(
        "--max_budget",
        type=float,
        default=0.0,
        help="Absolute USD cap. Experiment stops immediately when total spend reaches this.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _reasoning_budget = args.reasoning_budget
    _max_budget = args.max_budget
    if args.estimated_cost_per_sample > 0 or args.max_budget > 0:
        # Resolve API key (same logic as run())
        _api_key_resolved = args.api_key
        if not _api_key_resolved and Path(args.api_key_file).exists():
            for line in Path(args.api_key_file).read_text().splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    _api_key_resolved = line.split("=", 1)[1].strip()
                    break
        _est_cost_per_sample = args.estimated_cost_per_sample
        _budget_api_key = _api_key_resolved
        _budget_api_base = args.api_base
        try:
            _cost_baseline = get_key_usage(_budget_api_key, _budget_api_base)
            cap_msg = f"  Cap=${_max_budget:.2f}" if _max_budget > 0 else ""
            print(f"[BUDGET] Watchdog active. Baseline usage=${_cost_baseline:.4f} USD  Estimate=${_est_cost_per_sample:.5f}/sample  Limit=+{_BUDGET_TOLERANCE*100:.0f}%{cap_msg}")
        except Exception as e:
            print(f"[BUDGET] WARNING: Could not fetch baseline usage: {e}. Watchdog disabled.")
            _est_cost_per_sample = 0.0
    run(args)
