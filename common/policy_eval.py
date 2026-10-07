"""Common held-out generation protocol for Tasks 1-3.

For a fixed prompt list, a fixed seed and fixed decoding settings, this generates one sampled
response per prompt and records, per response: learned-reward score, length, truncation,
sampled KL to the frozen reference (course estimator: response-token log-prob difference),
and exact token-level policy entropy. Every condition in a comparison is evaluated with this
same function and the same arguments.
"""
from __future__ import annotations

import torch

from common.data import prompt_messages
from common.generation import batch_generate, score_reward_pairs
from common.models import reference_mode
from common.rl_utils import describe, response_logprobs_chunked, seeded


@torch.no_grad()
def generate_and_score(
    cfg: dict,
    policy,
    tokenizer,
    reward_model,
    reward_tokenizer,
    rows: list[dict],
    max_new_tokens: int,
    max_prompt_length: int = 256,
    batch_size: int = 8,
    seed: int | None = None,
    do_sample: bool = True,
    reward_max_length: int = 1280,
    messages_fn=prompt_messages,
):
    gen_cfg = cfg.get("generation", {})
    seed = int(cfg["seed"] if seed is None else seed)
    records = []
    kl_num = kl_den = ent_num = 0.0

    with seeded(seed):
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            prompts = [messages_fn(r) for r in chunk]
            gen = batch_generate(
                policy,
                tokenizer,
                prompts,
                max_prompt_length=max_prompt_length,
                max_new_tokens=max_new_tokens,
                temperature=float(gen_cfg.get("temperature", 0.7)),
                top_p=float(gen_cfg.get("top_p", 0.9)),
                do_sample=do_sample,
            )
            seq, attn, pw = gen["sequences"], gen["attention_mask"], gen["prompt_width"]
            rid, rmask = gen["response_ids"], gen["response_mask"]

            pol_lp, ent = response_logprobs_chunked(policy, seq, attn, pw, rid, with_entropy=True)
            with reference_mode(policy):
                ref_lp, _ = response_logprobs_chunked(policy, seq, attn, pw, rid, with_entropy=False)
            rewards = score_reward_pairs(
                reward_model, reward_tokenizer, prompts, gen["responses"], max_length=reward_max_length
            ).cpu()

            diff = (pol_lp - ref_lp) * rmask
            kl_num += float(diff.sum().item())
            kl_den += float(rmask.sum().item())
            ent_num += float((ent * rmask).sum().item())

            for i, row in enumerate(chunk):
                n = max(float(rmask[i].sum().item()), 1.0)
                records.append(
                    {
                        "prompt_id": str(row.get("prompt_id", row.get("source_index", start + i))),
                        "prompt": prompts[i][-1]["content"] if prompts[i] else "",
                        "response": gen["responses"][i],
                        "reward": float(rewards[i].item()),
                        "response_tokens": int(gen["response_lengths"][i]),
                        "terminated_with_eos": bool(gen["terminated_with_eos"][i]),
                        "truncated": bool(gen["truncated"][i]),
                        "kl_seq_sum": float(diff[i].sum().item()),
                        "kl_token_mean": float(diff[i].sum().item() / n),
                        "entropy_token_mean": float((ent[i] * rmask[i]).sum().item() / n),
                    }
                )
    return records, {"kl_token_weighted": kl_num / max(kl_den, 1.0), "entropy_token_weighted": ent_num / max(kl_den, 1.0)}


def summarize_generations(records: list[dict], pooled: dict | None = None) -> dict:
    out = {
        "n": len(records),
        "reward": describe([r["reward"] for r in records]),
        "response_tokens": describe([r["response_tokens"] for r in records]),
        "truncation_rate": float(sum(r["truncated"] for r in records) / max(len(records), 1)),
        "kl_seq_sum": describe([r["kl_seq_sum"] for r in records]),
        "kl_token_mean_per_seq": describe([r["kl_token_mean"] for r in records]),
        "entropy_token_mean_per_seq": describe([r["entropy_token_mean"] for r in records]),
    }
    if pooled:
        # Course convention: KL/entropy aggregated over all valid response tokens (token-weighted).
        out["kl"] = pooled["kl_token_weighted"]
        out["entropy"] = pooled["entropy_token_weighted"]
    return out



def run_heldout_eval(cfg: dict, adapter: str | None, name: str, out_dir, max_new_tokens: int):
    """Load one frozen policy and evaluate it on the first `eval_prompts` held-out RL prompts.

    Shared by Task 2 and Task 3 so every PPO/GRPO condition (and the SFT / midpoint reference
    points) is scored with identical prompts, seed, decoding and generation cap.
    """
    from common.data import read_jsonl, repo_path, write_jsonl
    from common.logging_utils import save_json
    from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer

    rows = read_jsonl(cfg["paths"]["rl_prompt_eval"])[: int(cfg.get("eval_prompts", 64))]
    tok = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=adapter, trainable=False)
    rm, rm_tok = load_reward_model(cfg)
    recs, pooled = generate_and_score(
        cfg, policy, tok, rm, rm_tok, rows,
        max_new_tokens=int(max_new_tokens),
        max_prompt_length=int(cfg.get("max_prompt_length", 256)),
        batch_size=int(cfg.get("eval_batch_size", 8)),
        reward_max_length=int(cfg.get("reward_max_length", 1280)),
    )
    out_dir = repo_path(out_dir)
    write_jsonl(out_dir / f"{name}_generations.jsonl", recs)
    summary = {"name": name, "adapter": adapter, "max_new_tokens": int(max_new_tokens), **summarize_generations(recs, pooled)}
    save_json(out_dir / f"{name}.json", summary)
    print(f"[eval:{name}] reward {summary['reward']['mean']:.3f}±{summary['reward']['std']:.3f} KL {summary['kl']:.4f} "
          f"H {summary['entropy']:.3f} len {summary['response_tokens']['mean']:.0f} trunc {summary['truncation_rate']:.2f}")
    clear_gpu(policy, rm)
    del policy, rm
    clear_gpu()
    return summary
