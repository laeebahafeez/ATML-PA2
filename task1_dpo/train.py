from __future__ import annotations

import argparse

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import response_sequence_logprobs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.models import clear_gpu, count_parameters, load_policy, load_tokenizer, reference_mode, trainable_parameters
from common.rl_utils import ensure_fp32_trainable, grad_norm, peak_vram_gib, reset_peak_vram
from task1_dpo.dpo import dpo_loss

# A pair is only usable if the prompt leaves room for some response tokens inside
# max_sequence_length (encode_prompt_response raises otherwise). The same deterministic
# filter is applied to every condition and the number of dropped rows is logged.
MIN_RESPONSE_TOKENS = 32


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def filter_fitting_rows(rows, tokenizer, max_length):
    keep, dropped = [], []
    for row in rows:
        n = len(tokenizer.apply_chat_template(prompt_messages_from_preference(row), tokenize=True, add_generation_prompt=True))
        (keep if n + MIN_RESPONSE_TOKENS <= max_length else dropped).append(row)
    return keep, [str(r.get("prompt_id", r.get("source_index"))) for r in dropped]


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)

    tokenizer = load_tokenizer(cfg["base_model"])
    rows, dropped = filter_fitting_rows(rows, tokenizer, int(cfg["max_sequence_length"]))
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    model = load_policy(cfg, trainable=True, fresh_lora=True)
    ensure_fp32_trainable(model)
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "dropped_prompt_ids": dropped,
        "dataset_path": str(path),
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def _to(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg = bundle["cfg"]
    output = repo_path(output_path or cfg["standard_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    results_dir = repo_path(cfg["results_dir"])
    log_path = results_dir / f"train_{run_name}_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    model, optimizer, loader, beta = bundle["model"], bundle["optimizer"], bundle["loader"], bundle["beta"]
    params = trainable_parameters(model)
    device = next(model.parameters()).device
    accum = int(cfg["grad_accum_steps"])
    max_norm = float(cfg["max_grad_norm"])
    epochs = int(cfg.get("epochs", 1))
    total_params, n_trainable = count_parameters(model)
    print(f"[dpo:{run_name}] rows={len(bundle['rows'])} dropped={len(bundle['dropped_prompt_ids'])} beta={beta} "
          f"trainable={n_trainable}/{total_params}")

    reset_peak_vram()
    elapsed = wall_timer()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    step, micro, seen = 0, 0, 0
    window = {"loss": [], "acc": [], "margin": [], "chosen_reward": [], "rejected_reward": []}
    n_micro_total = epochs * len(loader)

    for epoch in range(epochs):
        for chosen, rejected in loader:
            chosen, rejected = _to(chosen, device), _to(rejected, device)
            with torch.no_grad(), reference_mode(model):
                ref_c, _, _ = response_sequence_logprobs(model, chosen)
                ref_r, _, _ = response_sequence_logprobs(model, rejected)
            pol_c, _, _ = response_sequence_logprobs(model, chosen)
            pol_r, _, _ = response_sequence_logprobs(model, rejected)

            loss, diag = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta)
            (loss / accum).backward()
            micro += 1
            seen += chosen["input_ids"].shape[0]

            window["loss"].append(float(loss.item()))
            window["acc"].append(float(diag["preference_accuracy"].item()))
            window["margin"].append(float(diag["logit_mean"].item()) / beta)
            window["chosen_reward"].append(float((beta * (pol_c - ref_c)).mean().item()))
            window["rejected_reward"].append(float((beta * (pol_r - ref_r)).mean().item()))

            if micro % accum == 0 or micro == n_micro_total:
                gnorm = grad_norm(params)
                torch.nn.utils.clip_grad_norm_(params, max_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                rec = {
                    "step": step,
                    "epoch": epoch,
                    "examples_seen": seen,
                    "loss": sum(window["loss"]) / len(window["loss"]),
                    "train_preference_accuracy": sum(window["acc"]) / len(window["acc"]),
                    "train_dpo_margin": sum(window["margin"]) / len(window["margin"]),
                    "chosen_implicit_reward": sum(window["chosen_reward"]) / len(window["chosen_reward"]),
                    "rejected_implicit_reward": sum(window["rejected_reward"]) / len(window["rejected_reward"]),
                    "grad_norm": gnorm,
                    "lr": optimizer.param_groups[0]["lr"],
                    "elapsed_s": elapsed(),
                }
                append_jsonl(log_path, rec)
                window = {k: [] for k in window}
                if step % 5 == 0 or micro == n_micro_total:
                    print(f"[dpo:{run_name}] step {step} seen {seen} loss {rec['loss']:.4f} "
                          f"acc {rec['train_preference_accuracy']:.3f} gnorm {gnorm:.3f} t {rec['elapsed_s']:.0f}s")

    model.save_pretrained(str(output))
    summary = {
        "run_name": run_name,
        "dataset": bundle["dataset_path"],
        "n_train_pairs": len(bundle["rows"]),
        "n_dropped_overlong_prompts": len(bundle["dropped_prompt_ids"]),
        "dropped_prompt_ids": bundle["dropped_prompt_ids"],
        "beta": beta,
        "epochs": epochs,
        "optimizer_steps": step,
        "batch_size": int(cfg["batch_size"]),
        "grad_accum_steps": accum,
        "learning_rate": float(cfg["learning_rate"]),
        "seed": int(cfg["seed"]),
        "wall_clock_s": elapsed(),
        "peak_vram_gib": peak_vram_gib(),
        "trainable_params": n_trainable,
        "adapter": output.relative_to(repo_path(".")).as_posix(),
    }
    save_json(results_dir / f"train_{run_name}_summary.json", summary)
    print(f"[dpo:{run_name}] saved adapter -> {output}")
    clear_gpu(model, optimizer)
    del bundle, model, optimizer
    clear_gpu()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()
