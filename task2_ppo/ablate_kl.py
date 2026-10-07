"""Task 2 Step 3: reward over-optimization / KL-pressure study.

Matched short forks (cfg['fork_updates'] updates, same prompts/seed/eps) from the identical
midpoint for beta_KL in cfg['kl_values'], then the common held-out evaluation. The midpoint
itself is evaluated as the shared starting point so changes are read relative to it.
"""
from __future__ import annotations

import argparse

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from task2_ppo.continue_train import run_or_load_fork
from task2_ppo.evaluate import evaluate


def eval_once(config_path, adapter, name, force=False):
    cfg = load_yaml(config_path)
    p = repo_path(cfg["results_dir"]) / "eval" / f"{name}.json"
    if p.exists() and not force:
        return load_json(p)
    return evaluate(config_path, adapter, name)


def trajectory(cfg, run_name):
    return read_jsonl(repo_path(cfg["results_dir"]) / run_name / "train_log.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("KL beta conditions:", cfg["kl_values"])
    print("Fork update budget:", cfg["fork_updates"])

    eps = float(cfg["clip_epsilon"])
    out = {"midpoint": eval_once(args.config, cfg["paths"]["ppo_midpoint_policy"], "midpoint", args.force), "forks": []}
    for kl in cfg["kl_values"]:
        name = run_or_load_fork(args.config, eps, float(kl), force=args.force)
        ev = eval_once(args.config, f"outputs/task2_ppo/{name}", name, args.force)
        traj = trajectory(cfg, name)
        out["forks"].append({
            "kl_beta": float(kl), "clip_epsilon": eps, "run": name, "heldout": ev,
            "train_final": traj[-1] if traj else None,
            "train_trajectory": {k: [r[k] for r in traj] for k in ["reward", "kl", "entropy", "response_length", "kl_seq_sum"]},
        })

    save_json(repo_path(cfg["results_dir"]) / "kl_ablation.json", out)
    m = out["midpoint"]
    print(f"\nmidpoint      | R {m['reward']['mean']:.3f} | KL {m['kl']:.4f} | H {m['entropy']:.3f} | len {m['response_tokens']['mean']:.0f}")
    for f in out["forks"]:
        h = f["heldout"]
        print(f"beta_KL={f['kl_beta']:<5g}| R {h['reward']['mean']:.3f} | KL {h['kl']:.4f} | H {h['entropy']:.3f} | "
              f"len {h['response_tokens']['mean']:.0f}")


if __name__ == "__main__":
    main()
