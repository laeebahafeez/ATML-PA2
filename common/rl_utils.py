"""Helpers shared by the Task 1-3 training loops and the common held-out evaluation.

Everything here is task-agnostic: memory-light response log-prob/entropy computation,
peak-VRAM / wall-clock bookkeeping, gradient norms, and deterministic prompt schedules.
"""
from __future__ import annotations

import math
import random
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

from common.models import reference_mode


# --------------------------------------------------------------------------------------
# Log-probabilities and entropy of generated responses
# --------------------------------------------------------------------------------------

def _forward_response_logits(model, sequences, attention_mask, prompt_width, n_resp):
    """Logits that predict response tokens 0..n_resp-1 (shape [B, n_resp, V]).

    Uses `logits_to_keep` so the LM head is only applied to the response positions;
    on a 151k vocabulary this is the dominant memory cost of a forward pass.
    """
    keep = n_resp + 1
    try:
        out = model(
            input_ids=sequences,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            logits_to_keep=keep,
        )
        logits = out.logits[:, :-1, :]
    except TypeError:  # very old transformers: fall back to full logits
        out = model(input_ids=sequences, attention_mask=attention_mask, use_cache=False, return_dict=True)
        logits = out.logits[:, prompt_width - 1 : -1, :]
    return logits[:, :n_resp, :]


def response_logprobs(model, sequences, attention_mask, prompt_width, response_ids, with_entropy=False):
    """Per-token log pi(a_t|s_t) for the sampled response tokens.

    Equivalent to common.generation.response_token_logprobs but memory-light. Returns
    (logp [B,T], entropy [B,T] or None). Entropy is the exact full-vocabulary entropy,
    computed without gradient (it is a diagnostic only).
    """
    n_resp = response_ids.shape[1]
    logits = _forward_response_logits(model, sequences, attention_mask, prompt_width, n_resp)
    logp_all = F.log_softmax(logits.float(), dim=-1)
    logp = torch.gather(logp_all, -1, response_ids.unsqueeze(-1)).squeeze(-1)
    ent = None
    if with_entropy:
        with torch.no_grad():
            ent = -(logp_all.exp() * logp_all).sum(-1)
    return logp, ent


@torch.no_grad()
def response_logprobs_chunked(model, sequences, attention_mask, prompt_width, response_ids, with_entropy=True, chunk=4):
    """No-grad version of `response_logprobs`, processed in row chunks to bound memory."""
    lps, ents = [], []
    for s in range(0, sequences.shape[0], chunk):
        lp, ent = response_logprobs(
            model,
            sequences[s : s + chunk],
            attention_mask[s : s + chunk],
            prompt_width,
            response_ids[s : s + chunk],
            with_entropy=with_entropy,
        )
        lps.append(lp)
        if with_entropy:
            ents.append(ent)
    return torch.cat(lps), (torch.cat(ents) if with_entropy else None)


@torch.no_grad()
def reference_logprobs(model, sequences, attention_mask, prompt_width, response_ids, chunk=4):
    """Log-probs under the frozen reference policy (= base model with the LoRA adapter disabled)."""
    with reference_mode(model):
        lp, _ = response_logprobs_chunked(
            model, sequences, attention_mask, prompt_width, response_ids, with_entropy=False, chunk=chunk
        )
    return lp


# --------------------------------------------------------------------------------------
# Bookkeeping
# --------------------------------------------------------------------------------------

def reset_peak_vram():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def peak_vram_gib():
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**30


def grad_norm(parameters) -> float:
    """Global L2 norm of the current gradients (before clipping)."""
    norms = [p.grad.detach().float().norm(2) for p in parameters if p.grad is not None]
    if not norms:
        return 0.0
    return float(torch.norm(torch.stack(norms), 2).item())


def ensure_fp32_trainable(model):
    """AdamW on fp16 parameters is unstable (eps underflows, tiny updates are lost).

    PEFT normally autocasts adapter weights to fp32; this upcasts anything that slipped
    through (e.g. a modules_to_save copy of an fp16 head).
    """
    n = 0
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
            n += 1
    return n


def detach_rollout(gen: dict) -> dict:
    """batch_generate runs under torch.inference_mode(); its tensors cannot be used in an
    autograd graph. Clone them into normal tensors before the training forward passes."""
    out = dict(gen)
    for k in ("sequences", "attention_mask", "response_ids", "response_mask"):
        out[k] = gen[k].clone()
    return out


def disable_dropout(model):
    """On-policy RL needs pi_old == pi_theta before the first step; LoRA dropout would make the
    importance ratio noisy (spurious clipping). Gradient checkpointing still applies because the
    model stays in train mode."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0


def prompt_schedule(rows: list[dict], n_updates: int, per_update: int, seed: int) -> list[list[dict]]:
    """Deterministic prompt order shared by every condition (matched forks see identical prompts)."""
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    need = n_updates * per_update
    if need > len(order):
        order = (order * math.ceil(need / len(order)))[:need]
    return [[rows[i] for i in order[u * per_update : (u + 1) * per_update]] for u in range(n_updates)]


def describe(values) -> dict:
    """Mean / std / median / IQR summary used for every length/reward distribution we report."""
    a = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float)
    if a.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "median": float("nan"), "iqr": float("nan")}
    q1, q3 = np.percentile(a, [25, 75])
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "median": float(np.median(a)),
        "iqr": float(q3 - q1),
    }


@contextmanager
def seeded(seed: int):
    """Temporarily fix all RNGs (used so every condition's evaluation samples are reproducible)."""
    py_state, np_state, t_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(t_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
