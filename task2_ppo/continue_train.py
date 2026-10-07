"""Task 2: PPO continuation from the supplied midpoint policy/critic.

One update = sample `prompts_per_update` on-policy responses, score them with the frozen
reward model, build KL-shaped token rewards against the frozen reference (base model with the
adapter disabled), run GAE with the released critic, then `ppo_epochs` clipped-policy and
value-regression steps on that batch. Every fork starts from the identical supplied
policy/value state and sees the identical seeded prompt sequence.
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
from common.models import (
    clear_gpu,
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    token_values,
    trainable_parameters,
    value_parameter_groups,
)
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
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards, value_mse_loss


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    # Keep optimizer state in fp32 and make the importance ratio exactly 1 before a step.
    ensure_fp32_trainable(policy)
    ensure_fp32_trainable(value_model)
    disable_dropout(policy)
    disable_dropout(value_model)

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


def response_values(value_model, sequences, attention_mask, prompt_width, n_resp):
    """V(s_t) for each response token t: the critic's output at the position that predicts token t."""
    v = token_values(value_model, sequences, attention_mask)
    return v[:, prompt_width - 1 : -1][:, :n_resp].float()


def explained_variance(pred, target, mask):
    m = mask.bool()
    y, p = target[m], pred[m]
    var_y = y.var(unbiased=False)
    if var_y.item() < 1e-8:
        return float("nan")
    return float(1.0 - (y - p).var(unbiased=False) / var_y)


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard"):
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    tok, policy, value_model = bundle["tokenizer"], bundle["policy"], bundle["value_model"]
    rm, rm_tok = bundle["reward_model"], bundle["reward_tokenizer"]
    p_opt, v_opt = bundle["policy_optimizer"], bundle["value_optimizer"]
    p_params, v_params = trainable_parameters(policy), trainable_parameters(value_model)

    eps, beta_kl = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    n_updates, epochs = int(cfg["updates"]), int(cfg["ppo_epochs"])
    max_norm = float(cfg["max_grad_norm"])
    gen_cfg = cfg.get("generation", {})
    res_dir = repo_path(cfg["results_dir"]) / run_name
    log_path, roll_path = res_dir / "train_log.jsonl", res_dir / "rollouts.jsonl"
    for p in (log_path, roll_path):
        if p.exists():
            p.unlink()

    schedule = prompt_schedule(bundle["prompt_rows"], n_updates, int(cfg["prompts_per_update"]), int(cfg["seed"]))
    print(f"[ppo:{run_name}] updates={n_updates} eps={eps} kl_beta={beta_kl} epochs={epochs}")

    reset_peak_vram()
    elapsed = wall_timer()
    set_seed(int(cfg["seed"]))  # identical sampling stream for every fork
    for u, rows in enumerate(schedule):
        prompts = [prompt_messages(r) for r in rows]

        # ---- 1. On-policy rollout -------------------------------------------------------
        gen = batch_generate(
            policy, tok, prompts,
            max_prompt_length=int(cfg["max_prompt_length"]),
            max_new_tokens=int(cfg["max_response_length"]),
            temperature=float(gen_cfg.get("temperature", 0.7)),
            top_p=float(gen_cfg.get("top_p", 0.9)),
            do_sample=True,
        )
        gen = detach_rollout(gen)
        seq, attn, pw = gen["sequences"], gen["attention_mask"], gen["prompt_width"]
        rid, mask = gen["response_ids"], gen["response_mask"]
        T = rid.shape[1]

        # ---- 2. Old / reference log-probs, critic values, learned reward ----------------
        with torch.no_grad():
            old_lp, ent = response_logprobs_chunked(policy, seq, attn, pw, rid, with_entropy=True)
            ref_lp = reference_logprobs(policy, seq, attn, pw, rid)
            values = response_values(value_model, seq, attn, pw, T)
            raw_reward = score_reward_pairs(rm, rm_tok, prompts, gen["responses"],
                                            max_length=int(cfg.get("reward_max_length", 1280))).to(seq.device)
            no_eos = torch.tensor([not t for t in gen["terminated_with_eos"]], device=seq.device, dtype=torch.float32)
            task_reward = raw_reward - float(cfg.get("missing_eos_penalty", 0.0)) * no_eos

            # ---- 3. KL-shaped rewards, GAE, returns --------------------------------------
            rewards = shaped_rewards(task_reward, old_lp, ref_lp, mask, beta_kl)
            adv, returns = compute_gae(rewards, values * mask, mask, float(cfg["gamma"]), float(cfg["gae_lambda"]))
            adv_n = normalize_advantages(adv, mask)
            ev = explained_variance(values, returns, mask)

        # ---- 4. PPO epochs on this batch -------------------------------------------------
        stats = {k: [] for k in ["policy_loss", "value_loss", "clip_fraction", "grad_norm", "value_grad_norm",
                                 "approx_kl_old_new", "ratio_max", "ratio_min", "active_clip_fraction"]}
        for _ in range(epochs):
            new_lp, _ = response_logprobs(policy, seq, attn, pw, rid)
            pl, ratio, clipfrac = ppo_policy_loss(new_lp, old_lp, adv_n, mask, eps)
            p_opt.zero_grad(set_to_none=True)
            pl.backward()
            gn = grad_norm(p_params)
            torch.nn.utils.clip_grad_norm_(p_params, max_norm)
            p_opt.step()

            v_pred = response_values(value_model, seq, attn, pw, T)
            vl = float(cfg["value_coef"]) * value_mse_loss(v_pred, returns, mask)
            v_opt.zero_grad(set_to_none=True)
            vl.backward()
            vgn = grad_norm(v_params)
            torch.nn.utils.clip_grad_norm_(v_params, max_norm)
            v_opt.step()

            with torch.no_grad():
                logr = (new_lp.detach() - old_lp)
                # tokens whose gradient is actually zeroed by the clip (the min picks the clipped branch)
                active = (((ratio > 1 + eps) & (adv_n > 0)) | ((ratio < 1 - eps) & (adv_n < 0))).float()
                m = mask.bool()
                stats["policy_loss"].append(float(pl.item()))
                stats["value_loss"].append(float(vl.item()))
                stats["clip_fraction"].append(float(clipfrac.item()))
                stats["active_clip_fraction"].append(float(masked_mean(active, mask).item()))
                stats["grad_norm"].append(gn)
                stats["value_grad_norm"].append(vgn)
                stats["approx_kl_old_new"].append(float(masked_mean(0.5 * logr**2, mask).item()))
                stats["ratio_max"].append(float(ratio[m].max().item()) if m.any() else 1.0)
                stats["ratio_min"].append(float(ratio[m].min().item()) if m.any() else 1.0)

        rec = {
            "update": u + 1,
            "prompt_ids": [str(r.get("prompt_id")) for r in rows],
            "reward": float(raw_reward.mean().item()),
            "effective_reward": float(task_reward.mean().item()),
            "kl": float(sampled_kl(old_lp, ref_lp, mask).item()),
            "kl_seq_sum": float(((old_lp - ref_lp) * mask).sum(-1).mean().item()),
            "entropy": float(masked_mean(ent, mask).item()),
            "response_length": float(mask.sum(-1).mean().item()),
            "truncated_frac": float(np.mean(gen["truncated"])),
            "eos_frac": float(np.mean(gen["terminated_with_eos"])),
            "value_mean": float(masked_mean(values, mask).item()),
            "return_mean": float(masked_mean(returns, mask).item()),
            "value_explained_variance": ev,
            **{k: float(np.mean(v)) for k, v in stats.items()},
            "ratio_max": float(np.max(stats["ratio_max"])),
            "ratio_min": float(np.min(stats["ratio_min"])),
            "approx_kl_last_epoch": stats["approx_kl_old_new"][-1],
            "elapsed_s": elapsed(),
            "peak_vram_gib": peak_vram_gib(),
        }
        append_jsonl(log_path, rec)
        for i, r in enumerate(rows):
            append_jsonl(roll_path, {"update": u + 1, "prompt_id": str(r.get("prompt_id")),
                                     "prompt": prompts[i][-1]["content"], "response": gen["responses"][i],
                                     "reward": float(raw_reward[i].item()), "response_tokens": int(gen["response_lengths"][i])})
        print(f"[ppo:{run_name}] {u + 1}/{n_updates} R {rec['reward']:.3f} KL {rec['kl']:.4f} H {rec['entropy']:.3f} "
              f"len {rec['response_length']:.0f} clip {rec['clip_fraction']:.3f} pl {rec['policy_loss']:.4f} "
              f"vl {rec['value_loss']:.4f} gn {rec['grad_norm']:.3f} t {rec['elapsed_s']:.0f}s")

    policy.save_pretrained(str(out))
    value_model.save_pretrained(str(out) + "_value")
    summary = {
        "run_name": run_name, "updates": n_updates, "clip_epsilon": eps, "kl_beta": beta_kl,
        "ppo_epochs": epochs, "prompts_per_update": int(cfg["prompts_per_update"]),
        "max_response_length": int(cfg["max_response_length"]), "seed": int(cfg["seed"]),
        "wall_clock_s": elapsed(), "peak_vram_gib": peak_vram_gib(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "adapter": out.relative_to(repo_path(".")).as_posix(),
    }
    save_json(res_dir / "summary.json", summary)
    print(f"[ppo:{run_name}] done in {summary['wall_clock_s']:.0f}s, peak VRAM {summary['peak_vram_gib']:.2f} GiB -> {out}")
    clear_gpu(policy, value_model, rm)
    del bundle, policy, value_model, rm
    clear_gpu()
    return summary


def fork_name(eps: float, kl: float) -> str:
    return f"fork_eps{eps:g}_kl{kl:g}".replace(".", "p")


def run_or_load_fork(config_path: str, eps: float, kl: float, force: bool = False) -> str:
    """Short fixed-budget fork from the midpoint. The (eps=0.2, kl=0.1) fork is shared by the
    clipping and KL studies, so it is trained once."""
    cfg = load_yaml(config_path)
    name = fork_name(eps, kl)
    out = f"outputs/task2_ppo/{name}"
    if force or not (repo_path(out) / "adapter_config.json").exists():
        run_ppo(config_path, output=out, updates=int(cfg["fork_updates"]), clip_epsilon=eps, kl_beta=kl, run_name=name)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name)


if __name__ == "__main__":
    main()
