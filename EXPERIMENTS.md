# Prompt Engineering for Wikipedia NPOV Bias Neutralization — Experiments

This document describes the experiments, scripts, and output directories contributed in this branch. It complements [REPORT_WRITEUP.md](REPORT_WRITEUP.md) with implementation-level details.

## 1. Overview

Task: neutralize biased Wikipedia sentences (WNC corpus) via LLM prompt engineering, evaluate with lexical + semantic + BiLSTM-tagger-based bias metrics.

Contributions in this branch:

1. **New strategy — `context_enriched` (CE)**: combines BiLSTM bias-tag hints with multi-turn few-shot demonstrations. Implemented in [run_local_models.py](llm_prompting_runner/run_local_models.py) and [run_claude.py](llm_prompting_runner/run_claude.py).
2. **Budget watchdog**: prevents runaway API spend via real-time OpenRouter key-usage polling + `--reasoning_budget` flag to disable hidden thinking tokens on Qwen-flash/thinking models.
3. **Ablations** on constraint strength (CE / CE_constrained / CE_soft), CoT on/off (qwen3.5-flash, 9b, 27b), model scale, and haiku 4.5 few-shot reproducibility.

All scripts and output directories live under [llm_prompting_runner/](llm_prompting_runner/).

## 2. Strategies

| Strategy | Message structure | Rationale |
|---|---|---|
| `zero_shot` | `[system, user]` | Baseline, only system-prompt rules. |
| `with_bias_tags` | `[system, user(<bias>…</bias>)]` | Tagger marks biased spans → model knows *where* to edit. |
| `few_shot` | `[system, user, user_ex1, asst_ex1, …, user_now]` | Model learns edit *magnitude* from demos. |
| `context_enriched` (CE) | `[system, user_ex1(tagged), asst_ex1, …, user_now(tagged)]` | Combines tags + few-shot → solves both "where" and "how much". |
| `CE_constrained` | CE + hard constraint in system prompt | Caps edit distance, improves Pareto balance. |
| `CE_soft` | CE with softened "hints only" phrasing | Baseline for constraint ablation. |
| `npov` / `self_refine` | inherited from teammate | Included for cross-strategy comparison. |

## 3. Scripts

All in [llm_prompting_runner/](llm_prompting_runner/).

| File | Purpose |
|---|---|
| [run_local_models.py](llm_prompting_runner/run_local_models.py) | OpenRouter / POE runner for all non-Claude models. Adds CE variants, `--reasoning_budget`, and a budget watchdog that polls `/api/v1/key` every 10 samples and aborts on >30% overspend. |
| [run_claude.py](llm_prompting_runner/run_claude.py) | Claude Code CLI runner (`claude -p`). Adds CE support, uses PS7 + env-var prompt passing to avoid Windows `cmd.exe` newline truncation. |
| [estimate_budget.py](llm_prompting_runner/estimate_budget.py) | Pre-flight cost estimator. Queries OpenRouter pricing, supports sample-count or token-count modes. **Required before any run.** |
| [bootstrap_ci.py](llm_prompting_runner/bootstrap_ci.py) | Bootstrap 95% CI for NoBias%, BERTScore, OverEditRate over jsonl output files. |
| [rescore_outputs_inplace.py](llm_prompting_runner/rescore_outputs_inplace.py) | Re-runs the evaluation suite on existing jsonl outputs without re-calling the LLM. |
| [test_qwen35_cot.py](llm_prompting_runner/test_qwen35_cot.py) | Smoke test for qwen3.5-flash CoT behavior / thinking-token accounting. |

### Standard invocation

```bash
cd llm_prompting_runner

# 1. Estimate cost first (mandatory)
python estimate_budget.py --model deepseek/deepseek-v3.2-exp \
    --n 500 --input_per_sample 870 --output_per_sample 40 \
    --api_key $OPENROUTER_API_KEY

# 2. Run experiment — must pass both flags for thinking models
python run_local_models.py \
    --models deepseek/deepseek-v3.2-exp \
    --strategies context_enriched \
    --samples 500 \
    --output_dir outputs_v32_ce \
    --reasoning_budget -1 \
    --estimated_cost_per_sample 0.0012
```

## 4. Output directories

Naming: `outputs_<model>[_<strategy|tag>]/`. Each contains `<model>__<strategy>.jsonl` per-sample records plus `summary.csv` / `summary.json`. All paths below are relative to the repo root.

### Main result tables (DeepSeek V3.2, context_enriched family, n=500)

| Dir | Strategy | NoBias%↓ | BERT↑ | OER↓ |
|---|---|---|---|---|
| [llm_prompting_runner/outputs_v32_ce_constrained/](llm_prompting_runner/outputs_v32_ce_constrained/) | CE_constrained | **25.8** | 0.811 | 0.089 |
| [llm_prompting_runner/outputs_v32_ce/](llm_prompting_runner/outputs_v32_ce/) | CE | 27.8 | 0.815 | 0.096 |
| [llm_prompting_runner/outputs_v32_ce_soft/](llm_prompting_runner/outputs_v32_ce_soft/) | CE_soft | 28.8 | 0.802 | 0.122 |
| [llm_prompting_runner/outputs_v32_fewshot_500/](llm_prompting_runner/outputs_v32_fewshot_500/) | few_shot | — | — | — |

### Earlier DeepSeek V3.2 runs (POE endpoint)

- [llm_prompting_runner/outputs_deepseek_v3/](llm_prompting_runner/outputs_deepseek_v3/) — Exp 1.1 baseline across 4 strategies, n=500.
- [llm_prompting_runner/outputs_ce_constrained_500/](llm_prompting_runner/outputs_ce_constrained_500/), [llm_prompting_runner/outputs_ce_ablation/](llm_prompting_runner/outputs_ce_ablation/), [llm_prompting_runner/outputs_ce_k5/](llm_prompting_runner/outputs_ce_k5/) — Exp 1.7 first-pass constraint ablation (model version was V3-0324, re-done in 1.8).

### CoT ablation (Exp 1.3 / 1.4 — qwen3.5 family)

- [llm_prompting_runner/outputs_qwen35flash_cot/](llm_prompting_runner/outputs_qwen35flash_cot/), [llm_prompting_runner/outputs_qwen35flash_nocot/](llm_prompting_runner/outputs_qwen35flash_nocot/) — CoT on/off on qwen3.5-flash, n=2000. Core finding: CoT drops NoBias 53% → 42.75%.
- [llm_prompting_runner/outputs_qwen35flash_4strats/](llm_prompting_runner/outputs_qwen35flash_4strats/) — 4-strategy comparison on qwen3.5-flash.
- [llm_prompting_runner/outputs_qwen35_9b_cot/](llm_prompting_runner/outputs_qwen35_9b_cot/), [llm_prompting_runner/outputs_qwen35_9b_nocot/](llm_prompting_runner/outputs_qwen35_9b_nocot/), [llm_prompting_runner/outputs_qwen35_9b/](llm_prompting_runner/outputs_qwen35_9b/), [llm_prompting_runner/outputs_qwen35_9b_v2/](llm_prompting_runner/outputs_qwen35_9b_v2/) — 9B CoT fails completely (output == source).
- [llm_prompting_runner/outputs_qwen35_27b_cot/](llm_prompting_runner/outputs_qwen35_27b_cot/), [llm_prompting_runner/outputs_qwen35_27b_nocot/](llm_prompting_runner/outputs_qwen35_27b_nocot/) — 27B runs; CoT run aborted at 112/2000.
- [llm_prompting_runner/outputs_qwen35_thinking/](llm_prompting_runner/outputs_qwen35_thinking/) — the accident: qwen3.5-flash with thinking unintentionally ON, ~$10 burned on ~1718 samples.
- [llm_prompting_runner/outputs_qwen_ce/](llm_prompting_runner/outputs_qwen_ce/) — qwen CE variant.
- [llm_prompting_runner/outputs_llama3_8b/](llm_prompting_runner/outputs_llama3_8b/) — Llama 3.1 8B, 4-strategy comparison.

### Claude experiments (Exp 1.5 / 1.6 / 1.9)

- [llm_prompting_runner/outputs_claude_ce/](llm_prompting_runner/outputs_claude_ce/) — Claude sonnet CE v1 ("MUST change"), n=500. Negative result: NoBias 33.2%, BERT 0.753 (aggressive deletion).
- [llm_prompting_runner/outputs_claude_ce_v2/](llm_prompting_runner/outputs_claude_ce_v2/) — CE v2 ("hints only"), n=100. BERT recovers to 0.790 but NoBias degrades to 53% (model does nothing).
- [llm_prompting_runner/outputs_haiku45_50/](llm_prompting_runner/outputs_haiku45_50/), [llm_prompting_runner/outputs_haiku45_ce50/](llm_prompting_runner/outputs_haiku45_ce50/), [llm_prompting_runner/outputs_haiku45_ce_multiturn/](llm_prompting_runner/outputs_haiku45_ce_multiturn/), [llm_prompting_runner/outputs_haiku45_cot_50/](llm_prompting_runner/outputs_haiku45_cot_50/) — haiku 4.5 strategy-comparison pilots.
- [llm_prompting_runner/outputs_haiku45_fewshot_flat_250/](llm_prompting_runner/outputs_haiku45_fewshot_flat_250/), [llm_prompting_runner/outputs_haiku45_fewshot_openrouter/](llm_prompting_runner/outputs_haiku45_fewshot_openrouter/), [llm_prompting_runner/outputs_haiku45_fewshot_t07/](llm_prompting_runner/outputs_haiku45_fewshot_t07/), [llm_prompting_runner/outputs_haiku45_ce_openrouter/](llm_prompting_runner/outputs_haiku45_ce_openrouter/) — Exp 1.9 teammate-fewshot reproducibility study. Five runs, all ~42% NoBias vs teammate's claimed 28.8%.
- [llm_prompting_runner/outputs_teammate_repro/](llm_prompting_runner/outputs_teammate_repro/) — running teammate's original code verbatim.

### Pilots / early validation (small, kept for reference)

- [llm_prompting_runner/outputs_haiku45_ce_pilot/](llm_prompting_runner/outputs_haiku45_ce_pilot/), [llm_prompting_runner/outputs_haiku45_cot_pilot/](llm_prompting_runner/outputs_haiku45_cot_pilot/), [llm_prompting_runner/outputs_haiku45_fewshot_10/](llm_prompting_runner/outputs_haiku45_fewshot_10/), [llm_prompting_runner/outputs_haiku45_ce_cli/](llm_prompting_runner/outputs_haiku45_ce_cli/), [llm_prompting_runner/outputs_haiku45_ce_cli_10/](llm_prompting_runner/outputs_haiku45_ce_cli_10/) — n=10–14 pilots before scaling.
- [llm_prompting_runner/outputs_sim_pilot/](llm_prompting_runner/outputs_sim_pilot/), [llm_prompting_runner/outputs_sim_pilot2/](llm_prompting_runner/outputs_sim_pilot2/) — similarity-prompt pilots.

## 5. Key findings (summary)

1. **CE_constrained is Pareto-optimal on DeepSeek V3.2**: NoBias 25.8%, BERT 0.811, beating teammate's claimed best (haiku few_shot 28.8 / 0.831) on both axes — subject to the reproducibility caveat below.
2. **CoT > parameter count for small models**: 9B CoT fully regresses, 27B CoT incomplete, but qwen3.5-flash+CoT (42.75%) crosses the 50% bias-retention ceiling that all no-CoT 7B–27B models hit.
3. **CE fails on single-turn CLI**: flat-text few-shot in a single `claude -p` call collapses the task-vector formation; CE needs multi-turn API to realize its advantage. Explains the Claude-CE negative result.
4. **Teammate's haiku few_shot (28.8%, 0.831) does not reproduce**: 5 runs across format / temperature / thinking variations land at ~42% NoBias. Confirmed not a bug in our scoring (formulas match byte-for-byte). Likely causes: model version drift, t=1.0 CLI non-determinism.
5. **Budget watchdog works**: hidden thinking tokens caused a ~$10 accident; now blocked by mandatory `--reasoning_budget -1` + real-time overspend abort.

## 6. Constraints and gotchas

- Python environment: `D:\STJU\cot_env\python.exe` (Windows) — uses GBK console encoding; **do not emit emoji in scripts**, use `[OK]`, `->`, etc.
- Any new LLM run: pre-flight [estimate_budget.py](llm_prompting_runner/estimate_budget.py) first, then always pass `--reasoning_budget -1` and `--estimated_cost_per_sample <value>` to [run_local_models.py](llm_prompting_runner/run_local_models.py).
- `.local.env` is git-ignored — create your own with `OPENROUTER_API_KEY=…`, `POE_API_KEY=…`, `ANTHROPIC_API_KEY=…`.
