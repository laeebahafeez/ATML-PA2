"""The micro-batched GRPO step must give the same loss and gradients as the released full-batch loss."""
import pytest
import torch

TINY = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


@pytest.fixture(scope="module")
def tiny_policy():
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32)
    except Exception as exc:  # offline
        pytest.skip(f"tiny model unavailable: {exc}")
    cfg = peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM", init_lora_weights=False)
    return peft.get_peft_model(model, cfg)


@pytest.mark.parametrize("loss_type", ["grpo", "dr_grpo"])
def test_microbatch_equals_full_batch(tiny_policy, loss_type):
    from common.rl_utils import response_logprobs
    from task3_grpo.continue_train import grpo_step_loss
    from task3_grpo.grpo import grpo_policy_loss

    torch.manual_seed(0)
    B, P, T = 4, 5, 6
    seq = torch.randint(0, 1000, (B, P + T))
    attn = torch.ones_like(seq)
    rid = seq[:, P:]
    mask = torch.ones(B, T)
    mask[1, 3:] = 0
    mask[3] = 0  # a masked (truncated) completion
    adv = torch.tensor([0.5, -1.0, 1.5, 0.3])
    with torch.no_grad():
        old, _ = response_logprobs(tiny_policy, seq, attn, P, rid)
        old = old + 0.05 * torch.randn_like(old)  # ratios != 1 so clipping is exercised
        ref = old - 0.1 * torch.rand_like(old)
    cfg = {"clip_epsilon": 0.2, "kl_beta": 0.1, "max_completion_length": T}
    params = [p for p in tiny_policy.parameters() if p.requires_grad]

    tiny_policy.zero_grad()
    new, _ = response_logprobs(tiny_policy, seq, attn, P, rid)
    full, _ = grpo_policy_loss(new, old, adv, mask, ref, 0.2, 0.1, loss_type=loss_type, max_completion_length=T)
    full.backward()
    g_full = [p.grad.clone() for p in params]

    for micro in (1, 3):
        tiny_policy.zero_grad()
        st = grpo_step_loss(tiny_policy, seq, attn, P, rid, old, ref, adv, mask, cfg, loss_type, micro)
        assert abs(st["loss"] - float(full)) < 1e-5
        for a, b in zip(g_full, [p.grad for p in params]):
            assert torch.allclose(a, b, atol=1e-6)
