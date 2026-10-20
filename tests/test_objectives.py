"""Validation of the three corrected objective implementations against the manual's equations.

Run:  python -m pytest tests -q
"""
import math

import torch

from task1_dpo.dpo import dpo_loss
from task2_ppo.ppo import compute_gae, ppo_policy_loss, shaped_rewards
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss


# ----------------------------------------------------------------------------- DPO
def test_dpo_matches_manual_equation():
    torch.manual_seed(0)
    pc, pr, rc, rr = (torch.randn(16) * 5 for _ in range(4))
    beta = 0.1
    loss, diag = dpo_loss(pc, pr, rc, rr, beta)
    m = (pc - rc) - (pr - rr)
    expected = -torch.nn.functional.logsigmoid(beta * m).mean()
    assert torch.allclose(loss, expected)
    assert math.isclose(float(diag["preference_accuracy"]), float((m > 0).float().mean()))


def test_dpo_at_initialization_is_log2_regardless_of_reference_margin():
    # policy == reference  =>  m = 0  =>  loss = log 2, even if the reference strongly prefers one side.
    rc, rr = torch.tensor([-10.0, -50.0]), torch.tensor([-80.0, -5.0])
    loss, diag = dpo_loss(rc.clone(), rr.clone(), rc, rr, beta=0.3)
    assert math.isclose(float(loss), math.log(2), rel_tol=1e-6)
    assert float(diag["preference_accuracy"]) == 0.0


def test_dpo_gradient_raises_chosen_lowers_rejected():
    pc = torch.tensor([-20.0], requires_grad=True)
    pr = torch.tensor([-20.0], requires_grad=True)
    loss, _ = dpo_loss(pc, pr, torch.tensor([-20.0]), torch.tensor([-20.0]), 0.1)
    loss.backward()
    assert pc.grad.item() < 0 < pr.grad.item()  # descent increases log pi(y+) and decreases log pi(y-)


# ----------------------------------------------------------------------------- PPO
def test_ppo_takes_pessimistic_bound():
    eps = 0.2
    old = torch.zeros(1, 4)
    ratio = torch.tensor([[1.5, 0.5, 1.5, 0.5]])
    adv = torch.tensor([[1.0, -1.0, -1.0, 1.0]])
    new = torch.log(ratio)
    mask = torch.ones(1, 4)
    loss, _, clipfrac = ppo_policy_loss(new, old, adv, mask, eps)
    # per-token min(rho A, clip(rho) A): [1.2, -0.8, -1.5, 0.5]
    expected = -torch.tensor([1.2, -0.8, -1.5, 0.5]).mean()
    assert torch.allclose(loss, expected, atol=1e-6)
    assert float(clipfrac) == 1.0  # every ratio lies outside [0.8, 1.2]


def test_ppo_clipped_tokens_get_no_gradient():
    eps = 0.2
    new = torch.log(torch.tensor([[1.5, 1.5]])).requires_grad_(True)
    adv = torch.tensor([[1.0, -1.0]])
    loss, _, _ = ppo_policy_loss(new, torch.zeros(1, 2), adv, torch.ones(1, 2), eps)
    loss.backward()
    assert new.grad[0, 0].item() == 0.0       # A>0, rho>1+eps: clipped, no incentive to grow further
    assert new.grad[0, 1].item() != 0.0       # A<0, rho>1+eps: unclipped branch, pushed back down


def test_gae_terminal_reward_lambda1_gamma1():
    rewards = shaped_rewards(torch.tensor([2.0]), torch.zeros(1, 5), torch.zeros(1, 5), torch.tensor([[1, 1, 1, 0, 0.0]]), 0.1)
    adv, ret = compute_gae(rewards, torch.zeros(1, 5), torch.tensor([[1, 1, 1, 0, 0.0]]), gamma=1.0, lam=1.0)
    assert torch.allclose(adv, torch.tensor([[2.0, 2.0, 2.0, 0.0, 0.0]]))
    assert torch.allclose(ret, adv)


# ----------------------------------------------------------------------------- GRPO
def test_grpo_advantages_are_per_group():
    rewards = torch.tensor([1.0, 2.0, 3.0, 10.0, 10.0, 10.0, 0.0, 4.0])
    gids = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2])
    adv = group_relative_advantages(rewards, gids)
    for g in range(3):
        r = rewards[gids == g]
        expected = (r - r.mean()) / (r.std(unbiased=False) + 1e-6)
        assert torch.allclose(adv[gids == g], expected, atol=1e-5)
    assert torch.all(adv[gids == 1] == 0)     # zero-std (uninformative) group -> zero advantage
    assert abs(float(adv[gids == 0].mean())) < 1e-6


def test_grpo_easy_prompt_does_not_dominate():
    # The buggy batch-wide normalization gives every completion of the easy prompt a positive advantage.
    rewards = torch.tensor([5.0, 6.0, -1.0, 0.0])
    adv = group_relative_advantages(rewards, torch.tensor([0, 0, 1, 1]))
    assert adv[0] < 0 < adv[1] and adv[2] < 0 < adv[3]


def test_grpo_length_normalizations():
    new = torch.zeros(2, 4, requires_grad=True)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1.0]])
    adv = torch.tensor([1.0, 1.0])
    l_grpo, _ = grpo_policy_loss(new, torch.zeros(2, 4), adv, mask, torch.zeros(2, 4), 0.2, 0.0, "grpo")
    l_dr, _ = grpo_policy_loss(new, torch.zeros(2, 4), adv, mask, torch.zeros(2, 4), 0.2, 0.0, "dr_grpo", max_completion_length=4)
    assert torch.allclose(l_grpo, torch.tensor(-1.0))            # mean_k (1/T_k) sum_t A
    assert torch.allclose(l_dr, torch.tensor(-(2 + 4) / 4 / 2))  # mean_k (1/L) sum_t A
