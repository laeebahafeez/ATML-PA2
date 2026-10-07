from __future__ import annotations

import argparse
import pandas as pd

from common.data import load_yaml, repo_path, write_jsonl
from common.generation import batch_generate
from common.models import clear_gpu, load_policy, load_tokenizer


def policy_specs(cfg):
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg):
    return pd.read_csv(repo_path(cfg["paths"]["xstest"]))


def generate_for_policy(cfg, policy_name: str, batch_size: int = 4):
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(policy_name)
    adapter = specs[policy_name]
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    df = load_xstest(cfg)
    records = []
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        prompts = [[{"role": "user", "content": str(x)}] for x in chunk["prompt"].tolist()]
        gen = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=256,
            max_new_tokens=int(cfg["safety_max_new_tokens"]),
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        for (_, row), response, n_tok in zip(chunk.iterrows(), gen["responses"], gen["response_lengths"]):
            records.append({
                "xstest_id": int(row["xstest_id"]),
                "policy": policy_name,
                "prompt": str(row["prompt"]),
                "benchmark_class": str(row["benchmark_class"]),
                "type": str(row["type"]),
                "response": response,
                "response_tokens": int(n_tok),
            })
    clear_gpu(model)
    del model
    clear_gpu()
    return records


def output_path(cfg, policy_name: str):
    return repo_path(cfg["results_dir"]) / "task4_safety" / f"generated_{policy_name}.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policies", nargs="+", help="subset of sft dpo ppo grpo (default: all four)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    specs = policy_specs(cfg)
    print("Policies:", list(specs))
    print("XSTest rows:", len(load_xstest(cfg)))

    # Frozen standard checkpoints only (Task 1 Step 1, Task 2 Step 1, Task 3 Step 1).
    for name in args.policies or list(specs):
        adapter = specs[name]
        if adapter is not None and not (repo_path(adapter) / "adapter_config.json").exists():
            raise FileNotFoundError(f"{name}: standard adapter not found at {adapter}; finish Tasks 1-3 first.")
        out = output_path(cfg, name)
        if out.exists() and not args.force:
            print(f"[{name}] exists, skipping: {out}")
            continue
        # Same greedy decoding, cap and prompt order for every policy.
        records = generate_for_policy(cfg, name, batch_size=args.batch_size)
        write_jsonl(out, records)
        mean_len = sum(r["response_tokens"] for r in records) / len(records)
        print(f"[{name}] wrote {len(records)} responses (mean {mean_len:.1f} tokens) -> {out}")


if __name__ == "__main__":
    main()
