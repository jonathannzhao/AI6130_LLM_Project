#!/usr/bin/env python3
"""
通用预算估算工具。跑实验前必须先调用。

用法示例：
  # 已知总token数
  python estimate_budget.py --model qwen/qwen3.5-27b-a3b-instruct --input_tokens 1742263 --output_tokens 77135

  # 按样本数估算
  python estimate_budget.py --model qwen/qwen3.5-27b-a3b-instruct --n 2000 --input_per_sample 870 --output_per_sample 40

  # 查询模型定价列表（不计算费用）
  python estimate_budget.py --list_models

  # 手动指定价格（跳过网络请求）
  python estimate_budget.py --model xxx --input_tokens 1000000 --output_tokens 100000 --input_price 0.4 --output_price 1.2
"""

import argparse
import os
import sys

import requests


def fetch_all_models(api_key: str):
    """从 OpenRouter 拉取全部模型定价。返回 {model_id: {input, output}} USD/M tokens。"""
    url = "https://openrouter.ai/api/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[ERROR] Failed to fetch model list: {e}")
        sys.exit(1)

    pricing = {}
    for m in data.get("data", []):
        mid = m.get("id", "")
        p = m.get("pricing", {})
        try:
            inp = float(p.get("prompt", 0)) * 1_000_000   # per M tokens
            out = float(p.get("completion", 0)) * 1_000_000
        except (TypeError, ValueError):
            continue
        pricing[mid] = {"input": inp, "output": out}
    return pricing


def get_model_price(model_id: str, api_key: str):
    """返回指定模型的 (input_price, output_price) USD/M tokens。"""
    pricing = fetch_all_models(api_key)
    if model_id not in pricing:
        # 模糊匹配
        candidates = [k for k in pricing if model_id.lower() in k.lower()]
        if len(candidates) == 1:
            print(f"[INFO] Exact model not found, using: {candidates[0]}")
            return pricing[candidates[0]]["input"], pricing[candidates[0]]["output"]
        elif len(candidates) > 1:
            print(f"[ERROR] Model '{model_id}' not found. Similar models:")
            for c in candidates[:10]:
                p = pricing[c]
                print(f"  {c}  in=${p['input']:.4f}/M  out={p['output']:.4f}/M")
            sys.exit(1)
        else:
            print(f"[ERROR] Model '{model_id}' not found on OpenRouter.")
            print("        Use --list_models to see available models.")
            sys.exit(1)
    p = pricing[model_id]
    return p["input"], p["output"]


def main():
    parser = argparse.ArgumentParser(
        description="通用预算估算工具 - 跑实验前调用",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # 模型
    parser.add_argument("--model", help="OpenRouter 模型 ID，如 qwen/qwen3.5-27b-a3b-instruct")

    # Token 数（两种输入方式二选一）
    grp = parser.add_argument_group("方式1: 直接指定总 token 数")
    grp.add_argument("--input_tokens", type=int, help="总 input tokens")
    grp.add_argument("--output_tokens", type=int, help="总 output tokens")

    grp2 = parser.add_argument_group("方式2: 按样本数估算")
    grp2.add_argument("--n", type=int, help="样本数")
    grp2.add_argument("--input_per_sample", type=int, help="每样本 input tokens")
    grp2.add_argument("--output_per_sample", type=int, help="每样本 output tokens")

    # 价格覆盖
    parser.add_argument("--input_price", type=float, help="input 价格 USD/M tokens（跳过 API 查询）")
    parser.add_argument("--output_price", type=float, help="output 价格 USD/M tokens（跳过 API 查询）")

    # API key
    parser.add_argument("--api_key", default="", help="OpenRouter API key（也可设 DASHSCOPE_API_KEY / OPENAI_API_KEY 环境变量）")

    # 其他
    parser.add_argument("--list_models", action="store_true", help="列出所有模型及定价后退出")
    parser.add_argument("--filter", default="", help="配合 --list_models 过滤关键词")

    args = parser.parse_args()

    api_key = (
        getattr(args, "api_key", None)
        or os.environ.get("OPENROUTER_API_KEY", "")
        or os.environ.get("DASHSCOPE_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
    )

    # --- 列出模型 ---
    if args.list_models:
        if not api_key:
            print("[ERROR] OPENROUTER_API_KEY not set")
            sys.exit(1)
        pricing = fetch_all_models(api_key)
        kw = args.filter.lower()
        rows = [(mid, p["input"], p["output"]) for mid, p in sorted(pricing.items())
                if kw in mid.lower()]
        if not rows:
            print(f"[INFO] No models match filter '{args.filter}'")
            sys.exit(0)
        print(f"\n{'Model':<55} {'Input$/M':>10} {'Output$/M':>10}")
        print("-" * 77)
        for mid, inp, out in rows:
            print(f"{mid:<55} {inp:>10.4f} {out:>10.4f}")
        print(f"\nTotal: {len(rows)} models")
        sys.exit(0)

    # --- 计算费用 ---
    if not args.model:
        parser.error("--model is required (or use --list_models)")

    # 确定 token 数
    if args.input_tokens is not None and args.output_tokens is not None:
        total_input = args.input_tokens
        total_output = args.output_tokens
        mode = "direct"
    elif args.n and args.input_per_sample and args.output_per_sample:
        total_input = args.n * args.input_per_sample
        total_output = args.n * args.output_per_sample
        mode = "per_sample"
    else:
        parser.error(
            "请指定 token 数：\n"
            "  方式1: --input_tokens N --output_tokens N\n"
            "  方式2: --n N --input_per_sample N --output_per_sample N"
        )

    # 确定价格
    if args.input_price is not None and args.output_price is not None:
        input_price_per_m = args.input_price
        output_price_per_m = args.output_price
        price_source = "manual"
    else:
        if not api_key:
            print("[ERROR] OPENROUTER_API_KEY not set.")
            print("        Either set the env var, or use --input_price and --output_price.")
            sys.exit(1)
        input_price_per_m, output_price_per_m = get_model_price(args.model, api_key)
        price_source = "openrouter"

    # 计算
    input_cost = total_input / 1_000_000 * input_price_per_m
    output_cost = total_output / 1_000_000 * output_price_per_m
    total_cost = input_cost + output_cost

    # 输出
    print()
    print("=" * 50)
    print("  BUDGET ESTIMATE")
    print("=" * 50)
    print(f"  Model       : {args.model}")
    print(f"  Price source: {price_source}")
    if mode == "per_sample":
        print(f"  Samples     : {args.n:,}")
        print(f"  Input/sample: {args.input_per_sample:,} tokens")
        print(f"  Output/sample: {args.output_per_sample:,} tokens")
    print()
    print(f"  Total input : {total_input:>12,} tokens  @ ${input_price_per_m:.4f}/M  = ${input_cost:.4f}")
    print(f"  Total output: {total_output:>12,} tokens  @ ${output_price_per_m:.4f}/M  = ${output_cost:.4f}")
    print(f"  {'':40}  --------")
    print(f"  {'TOTAL':40}  ${total_cost:.4f}")
    print("=" * 50)

    if total_cost >= 10:
        print(f"\n  [!!!] WARNING: Estimated cost >= $10. Confirm before running!\n")
    elif total_cost >= 5:
        print(f"\n  [!!]  CAUTION: Estimated cost >= $5.\n")
    elif total_cost >= 1:
        print(f"\n  [!]   Cost >= $1. Check before running.\n")
    else:
        print(f"\n  [OK]  Cost looks reasonable.\n")


if __name__ == "__main__":
    main()
