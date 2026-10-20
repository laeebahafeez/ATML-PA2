"""Task 3: GRPO continuation from the supplied midpoint adapter.

One update = sample K completions for each of `prompts_per_update` prompts, score them with the
frozen reward model, compute within-prompt group-relative advantages, mask completions that hit
the generation cap (when configured), and take `policy_epochs` clipped-objective steps with a
k3 KL penalty to the frozen reference. `loss_type` selects canonical per-sequence 1/T_k
normalization ("grpo") or the constant 1/L_max Dr.-GRPO-style normalization ("dr_grpo").
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.metrics import masked_mean, sampled_kl
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, trainable_parameters
from common.rl_utils import (
    detach_rollout,
    disable_dropout,
    ensure_fp32_trainable,
    grad_norm,
    peak_vram_gib,
    prompt_schedule,
    reference_logprobs,
    reset_peak_vram,
    response_logprobs,
    response_logprobs_chunked,
)
from task3_grpo.grpo import ZERO_STD_TOL, group_relative_advantages, grpo_policy_loss, mask_truncated_sequences


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    ensure_fp32_trainable(policy)
    disable_dropout(policy)
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


def group_stats(rewards: torch.Tensor, group_ids: torch.Tensor):
    stds = []
    for g in torch.unique(group_ids):
        stds.append(float(rewards[group_ids == g].std(unbiased=False).item()))
    stds = np.asarray(stds)
    return float(stds.mean()), float((stds < ZERO_STD_TOL).mean())


def grpo_step_loss(policy, seq, attn, pw, rid, old_lp, ref_lp, adv, token_mask, cfg, loss_type, micro):
    """Exact micro-batched version of task3_grpo.grpo.grpo_policy_loss.

    Full-batch loss = -(1/B) sum_k per_seq_k + beta * sum_tokens kl / N_valid. Each micro-batch S
    contributes policy_term_S * |S|/B + beta * kl_sum_S / N_valid, so gradients are identical
    to a single full-batch pass whatever the micro-batch size.
    """
    B = seq.shape[0]
    n_valid = float(token_mask.sum().item())
    eps, beta = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    agg = {"policy_term": 0.0, "kl_sum": 0.0, "clip_sum": 0.0, "ent_sum": 0.0}
    for s in range(0, B, micro):
        sl = slice(s, s + micro)
        new_lp, _ = response_logprobs(policy, seq[sl], attn[sl], pw, rid[sl])
        _, diag = grpo_policy_loss(  # diagnostics only (released helper)
            new_lp.detach(), old_lp[sl], adv[sl], token_mask[sl], ref_lp[sl], eps, 0.0,
            loss_type=loss_type, max_completion_length=int(cfg["max_completion_length"]),
        )
        # Policy term with gradient (same formula as grpo_policy_loss, beta=0).
        ratio = torch.exp(new_lp - old_lp[sl])
        a = adv[sl][:, None]
        obj = torch.minimum(ratio * a, ratio.clamp(1 - eps, 1 + eps) * a)
        tok_sum = (obj * token_mask[sl]).sum(-1)
        if loss_type == "grpo":
            per_seq = tok_sum / token_mask[sl].sum(-1).clamp_min(1.0)
        else:
            per_seq = tok_sum / float(cfg["max_completion_length"])
        lr = ref_lp[sl] - new_lp
        kl_tok = (torch.exp(lr) - lr - 1.0) * token_mask[sl]
        loss = -(per_seq.sum() / B) + (beta * kl_tok.sum() / n_valid if n_valid > 0 else 0.0)
        if n_valid > 0:
            loss.backward()
        with torch.no_grad():
            agg["policy_term"] += float(-(per_seq.sum() / B).item())
            agg["kl_sum"] += float(kl_tok.sum().item())
            m = token_mask[sl]
            agg["clip_sum"] += float(diag["clip_fraction"].item()) * float(m.sum().item())
            agg["ent_sum"] += float(diag["sample_entropy"].item()) * float(m.sum().item())
    nv = max(n_valid, 1.0)
    return {
        "policy_loss": agg["policy_term"],
        "kl_k3": agg["kl_sum"] / nv,
        "loss": agg["policy_term"] + beta * agg["kl_sum"] / nv,
        "clip_fraction": agg["clip_sum"] / nv,
        "n_valid_tokens": n_valid,
    }


def run_grpo(config_path: str, output: str | None = None, updates: int | None = None, loss_type: str = "grpo", run_name: str = "standard"):
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    tok, policy, opt = bundle["tokenizer"], bundle["policy"], bundle["optimizer"]
    rm, rm_tok = bundle["reward_model"], bundle["reward_tokenizer"]
    params = trainable_parameters(policy)
    K, n_updates = int(cfg["num_generations"]), int(cfg["updates"])
    L = int(cfg["max_completion_length"])
    micro = int(cfg.get("grad_micro_batch", K))
    gen_cfg = cfg.get("generation", {})
    res_dir = repo_path(cfg["results_dir"]) / run_name
    log_path, comp_path = res_dir / "train_log.jsonl", res_dir / "completions.jsonl"
    for p in (log_path, comp_path):
        if p.exists():
            p.unlink()

    schedule = prompt_schedule(bundle["prompt_rows"], n_updates, int(cfg["prompts_per_update"]), int(cfg["seed"]))
    print(f"[grpo:{run_name}] updates={n_updates} K={K} loss_type={loss_type} micro={micro}")

    reset_peak_vram()
    elapsed = wall_timer()
    set_seed(int(cfg["seed"]))
    for u, rows in enumerate(schedule):
        prompts, gids = [], []
        for g, r in enumerate(rows):
            prompts += [prompt_messages(r)] * K
            gids += [g] * K
        gids_t = torch.tensor(gids)

        gen = batch_generate(
            policy, tok, prompts,
            max_prompt_length=int(cfg["max_prompt_length"]),
            max_new_tokens=L,
            temperature=float(gen_cfg.get("temperature", 0.7)),
            top_p=float(gen_cfg.get("top_p", 0.9)),
            do_sample=True,
        )
        gen = detach_rollout(gen)
        seq, attn, pw = gen["sequences"], gen["attention_mask"], gen["prompt_width"]
        rid, rmask = gen["response_ids"], gen["response_mask"]

        with torch.no_grad():
            rewards = score_reward_pairs(rm, rm_tok, prompts, gen["responses"],
                                         max_length=int(cfg.get("reward_max_length", 1280))).float().cpu()
            adv = group_relative_advantages(rewards, gids_t).to(seq.device)
            token_mask = mask_truncated_sequences(rmask, gen["truncated"]) if cfg.get("mask_truncated_completions", True) else rmask
            old_lp, ent = response_logprobs_chunked(policy, seq, attn, pw, rid, with_entropy=True)
            ref_lp = reference_logprobs(policy, seq, attn, pw, rid)
        mean_std, frac_zero = group_stats(rewards, gids_t)

        step_stats = []
        for _ in range(int(cfg.get("policy_epochs", 1))):
            opt.zero_grad(set_to_none=True)
            st = grpo_step_loss(policy, seq, attn, pw, rid, old_lp, ref_lp, adv, token_mask, cfg, loss_type, micro)
            st["grad_norm"] = grad_norm(params)
            if st["n_valid_tokens"] > 0:
                torch.nn.utils.clip_grad_norm_(params, float(cfg["max_grad_norm"]))
                opt.step()
            step_stats.append(st)

        lengths = rmask.sum(-1).cpu().numpy()
        n_tok = token_mask.sum(-1).cpu().numpy()
        denom = np.maximum(n_tok, 1.0) if loss_type == "grpo" else np.full_like(n_tok, float(L))
        rec = {
            "update": u + 1,
            "prompt_ids": [str(r.get("prompt_id")) for r in rows],
            "reward": float(rewards.mean()),
            "group_reward_std": mean_std,
            "frac_zero_std_groups": frac_zero,
            "kl": float(sampled_kl(old_lp, ref_lp, rmask).item()),
            "kl_k3": float(np.mean([s["kl_k3"] for s in step_stats])),
            "policy_loss": float(np.mean([s["policy_loss"] for s in step_stats])),
            "loss": float(np.mean([s["loss"] for s in step_stats])),
            "grad_norm": float(np.mean([s["grad_norm"] for s in step_stats])),
            "clip_fraction": float(np.mean([s["clip_fraction"] for s in step_stats])),
            "entropy": float(masked_mean(ent, rmask).item()),
            "response_length": float(lengths.mean()),
            "response_length_std": float(lengths.std()),
            "truncated_frac": float(np.mean(gen["truncated"])),
            "masked_completions": int(sum(1 for t in n_tok if t == 0)),
            "elapsed_s": elapsed(),
            "peak_vram_gib": peak_vram_gib(),
        }
        append_jsonl(log_path, rec)
        for i in range(len(prompts)):
            a = float(adv[i].item())
            append_jsonl(comp_path, {
                "update": u + 1, "group": int(gids[i]), "prompt_id": str(rows[gids[i]].get("prompt_id")),
                "response": gen["responses"][i], "reward": float(rewards[i]), "advantage": a,
                "response_tokens": int(lengths[i]), "loss_tokens": int(n_tok[i]),
                "truncated": bool(gen["truncated"][i]),
                # Total gradient weight this completion's tokens receive in the policy term,
                # sum_t |A_k| / denom_k / B  (1/T_k per token for grpo, 1/L_max for dr_grpo).
                "gradient_mass": abs(a) * float(n_tok[i]) / float(denom[i]) / len(prompts),
            })
        print(f"[grpo:{run_name}] {u + 1}/{n_updates} R {rec['reward']:.3f} std {mean_std:.3f} zero {frac_zero:.2f} "
              f"KL {rec['kl']:.4f} H {rec['entropy']:.3f} len {rec['response_length']:.0f} gn {rec['grad_norm']:.3f} "
              f"t {rec['elapsed_s']:.0f}s")

    policy.save_pretrained(str(out))
    summary = {
        "run_name": run_name, "loss_type": loss_type, "updates": n_updates, "num_generations": K,
        "prompts_per_update": int(cfg["prompts_per_update"]), "clip_epsilon": float(cfg["clip_epsilon"]),
        "kl_beta": float(cfg["kl_beta"]), "max_completion_length": L, "seed": int(cfg["seed"]),
        "wall_clock_s": elapsed(), "peak_vram_gib": peak_vram_gib(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "adapter": out.relative_to(repo_path(".")).as_posix(),
    }
    save_json(res_dir / "summary.json", summary)
    print(f"[grpo:{run_name}] done in {summary['wall_clock_s']:.0f}s, peak VRAM {summary['peak_vram_gib']:.2f} GiB -> {out}")
    clear_gpu(policy, rm)
    del bundle, policy, rm
    clear_gpu()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name)


if __name__ == "__main__":
    main()
