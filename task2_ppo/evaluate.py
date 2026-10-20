from __future__ import annotations

import argparse

from common.data import load_yaml, read_jsonl
from common.models import load_policy, load_reward_model, load_tokenizer
from common.policy_eval import run_heldout_eval


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def evaluate(config_path: str, adapter: str | None, name: str):
    cfg = load_yaml(config_path)
    return run_heldout_eval(cfg, adapter, name, f"{cfg['results_dir']}/eval",
                            max_new_tokens=int(cfg["eval_max_response_length"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True, help="adapter dir, 'midpoint' for the supplied checkpoint, or 'none' for SFT")
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    adapter = {"none": None, "sft": None, "midpoint": cfg["paths"]["ppo_midpoint_policy"]}.get(args.adapter.lower(), args.adapter)
    evaluate(args.config, adapter, args.name)


if __name__ == "__main__":
    main()
