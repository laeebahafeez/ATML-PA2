"""Task 1 evaluation for one DPO condition (or the untouched SFT policy with --adapter none).

Writes, under results/task1_dpo/:
  eval_<name>.json              summary metrics
  eval_<name>_pairs.jsonl       per-pair DPO margins on the held-out preference set
  eval_<name>_generations.jsonl per-prompt generations with reward / KL / entropy / length
  eval_<name>_wordlimit.jsonl   greedy responses to the 10 explicit word-limit prompts
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.data import load_yaml, prompt_messages, prompt_messages_from_preference, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate, response_sequence_logprobs
from common.logging_utils import load_json, save_json, set_seed
from common.metrics import word_count, word_limit_compliance
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode
from common.policy_eval import generate_and_score, summarize_generations
from common.rl_utils import describe
from task1_dpo.dpo import dpo_loss
from task1_dpo.train import filter_fitting_rows, make_collate


def load_evaluation_bundle(config_path: str, adapter: str | None):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["dpo_standard_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def _pair_id(row):
    return str(row.get("prompt_id", row.get("source_index")))


@torch.no_grad()
def preference_metrics(model, tokenizer, rows, beta, max_length, batch_size=4):
    """Held-out DPO margin m = [log pi(y+) - log ref(y+)] - [log pi(y-) - log ref(y-)] per pair."""
    rows, dropped = filter_fitting_rows(rows, tokenizer, max_length)
    loader = DataLoader(rows, batch_size=batch_size, shuffle=False, collate_fn=make_collate(tokenizer, max_length))
    device = next(model.parameters()).device
    out, i = [], 0
    for chosen, rejected in loader:
        chosen = {k: v.to(device) for k, v in chosen.items()}
        rejected = {k: v.to(device) for k, v in rejected.items()}
        pol_c, _, mask_c = response_sequence_logprobs(model, chosen)
        pol_r, _, mask_r = response_sequence_logprobs(model, rejected)
        with reference_mode(model):
            ref_c, _, _ = response_sequence_logprobs(model, chosen)
            ref_r, _, _ = response_sequence_logprobs(model, rejected)
        for b in range(pol_c.shape[0]):
            row = rows[i]
            i += 1
            m = float((pol_c[b] - ref_c[b] - (pol_r[b] - ref_r[b])).item())
            loss, _ = dpo_loss(pol_c[b : b + 1], pol_r[b : b + 1], ref_c[b : b + 1], ref_r[b : b + 1], beta)
            out.append({
                "pair_id": _pair_id(row),
                "length_stratum": row.get("length_stratum"),
                "chosen_tokens": int(mask_c[b].sum().item()),
                "rejected_tokens": int(mask_r[b].sum().item()),
                "chosen_logratio": float((pol_c[b] - ref_c[b]).item()),
                "rejected_logratio": float((pol_r[b] - ref_r[b]).item()),
                "policy_margin": float((pol_c[b] - pol_r[b]).item()),
                "dpo_margin": m,
                "dpo_loss": float(loss.item()),
                "correct": bool(m > 0),
            })
    return out, dropped


def summarize_pairs(pairs):
    m = np.array([p["dpo_margin"] for p in pairs], dtype=float)
    return {
        "n_pairs": len(pairs),
        "dpo_loss": float(np.mean([p["dpo_loss"] for p in pairs])) if pairs else float("nan"),
        "preference_accuracy": float(np.mean(m > 0)) if pairs else float("nan"),
        "mean_dpo_margin": float(m.mean()) if pairs else float("nan"),
        "mean_chosen_logratio": float(np.mean([p["chosen_logratio"] for p in pairs])) if pairs else float("nan"),
        "mean_rejected_logratio": float(np.mean([p["rejected_logratio"] for p in pairs])) if pairs else float("nan"),
    }


@torch.no_grad()
def word_limit_eval(cfg, model, tokenizer, max_new_tokens):
    rows = read_jsonl(cfg["paths"]["word_limit_prompts"])
    prompts = [prompt_messages(r) for r in rows]
    gen = batch_generate(model, tokenizer, prompts, max_prompt_length=256, max_new_tokens=max_new_tokens,
                         temperature=0.0, top_p=1.0, do_sample=False)
    recs = []
    for r, p, text, n in zip(rows, prompts, gen["responses"], gen["response_lengths"]):
        user = p[-1]["content"]
        comp = word_limit_compliance(user, text)
        recs.append({"prompt_id": r["prompt_id"], "prompt": user, "response": text, "response_tokens": int(n),
                     "words": word_count(text), "compliant": comp})
    valid = [x["compliant"] for x in recs if x["compliant"] is not None]
    return recs, {
        "n": len(recs),
        "compliance_rate": float(np.mean(valid)) if valid else float("nan"),
        "words": describe([x["words"] for x in recs]),
        "response_tokens": describe([x["response_tokens"] for x in recs]),
    }


def run_evaluation(config_path: str, adapter: str | None, name: str, beta: float | None = None,
                   pair_rows=None, skip_generation: bool = False):
    bundle = load_evaluation_bundle(config_path, adapter)
    cfg, tok, policy = bundle["cfg"], bundle["tokenizer"], bundle["policy"]
    rm, rm_tok = bundle["reward"]
    set_seed(int(cfg["seed"]))
    res_dir = repo_path(cfg["results_dir"])

    if beta is None:
        train_summary = res_dir / f"train_{name}_summary.json"
        beta = float(load_json(train_summary)["beta"]) if train_summary.exists() else float(cfg["beta"])

    summary = {"name": name, "adapter": adapter, "beta_for_loss": beta}
    max_len = int(cfg["max_sequence_length"])
    bs = int(cfg.get("eval_batch_size", 4))

    # 1) Held-out preference fitting (meaningless for SFT, where policy == reference and m == 0).
    if adapter:
        pairs, dropped = preference_metrics(policy, tok, bundle["rows"] if pair_rows is None else pair_rows, beta, max_len, bs)
        write_jsonl(res_dir / f"eval_{name}_pairs.jsonl", pairs)
        summary["heldout_preference"] = summarize_pairs(pairs)
        summary["heldout_preference"]["n_dropped_overlong_prompts"] = len(dropped)

    # 2) Common generation protocol on the first N held-out prompts.
    if not skip_generation:
        n = int(cfg.get("eval_generation_prompts", 100))
        gen_rows = [{"prompt_id": _pair_id(r), "messages": prompt_messages_from_preference(r)} for r in bundle["rows"][:n]]
        recs, pooled = generate_and_score(
            cfg, policy, tok, rm, rm_tok, gen_rows,
            max_new_tokens=int(cfg["max_generation_tokens"]),
            batch_size=int(cfg.get("eval_generation_batch_size", 8)),
            reward_max_length=int(cfg.get("reward_max_length", 1280)),
        )
        write_jsonl(res_dir / f"eval_{name}_generations.jsonl", recs)
        summary["generation"] = summarize_generations(recs, pooled)

        wl_recs, wl_sum = word_limit_eval(cfg, policy, tok, int(cfg["max_generation_tokens"]))
        write_jsonl(res_dir / f"eval_{name}_wordlimit.jsonl", wl_recs)
        summary["word_limit"] = wl_sum

    save_json(res_dir / f"eval_{name}.json", summary)
    print(f"[dpo-eval:{name}]", {k: v for k, v in summary.items() if k in ("heldout_preference",)})
    if "generation" in summary:
        g = summary["generation"]
        print(f"[dpo-eval:{name}] reward {g['reward']['mean']:.3f} KL {g['kl']:.4f} len {g['response_tokens']['mean']:.1f} "
              f"word-limit {summary['word_limit']['compliance_rate']:.2f}")
    clear_gpu(policy, rm)
    del bundle, policy, rm
    clear_gpu()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True, help="adapter directory, or 'none' for the untouched SFT policy")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--beta", type=float, help="beta used to report the held-out DPO loss (default: the run's training beta)")
    args = ap.parse_args()
    adapter = None if args.adapter.lower() in {"none", "sft"} else args.adapter
    run_evaluation(args.config, adapter, args.name, args.beta)


if __name__ == "__main__":
    main()
