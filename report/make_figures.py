"""Build report figures (report/figures/*.pdf) and a numbers digest (report/tables.md) from results/.

Every number in the report should trace to a file under results/; this script only reads them.
Missing inputs are skipped with a note, so it can be run after any subset of tasks.

    python -m report.make_figures
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "report" / "figures"
# Categorical slots in fixed order (validated reference palette, light mode).
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6, "lines.markersize": 4,
    "legend.frameon": False, "savefig.bbox": "tight", "pdf.fonttype": 42, "axes.formatter.useoffset": False,
})
LINES: list[str] = []


def load(rel):
    p = RES / rel
    if not p.exists():
        print(f"  (skip: missing {p.relative_to(ROOT)})")
        return None
    if p.suffix == ".jsonl":
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    return json.loads(p.read_text(encoding="utf-8"))


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf")


def f(x, nd=3):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def table(header, rows):
    LINES.append("| " + " | ".join(header) + " |")
    LINES.append("|" + "---|" * len(header))
    for r in rows:
        LINES.append("| " + " | ".join(str(c) for c in r) + " |")
    LINES.append("")


def bar_panels(names, panels, title, fname, colors=None):
    """One small panel per metric (never a shared dual axis)."""
    fig, axes = plt.subplots(1, len(panels), figsize=(1.75 * len(panels), 2.3))
    for ax, (label, vals) in zip(np.atleast_1d(axes), panels):
        x = np.arange(len(names))
        ax.bar(x, [np.nan if v is None else v for v in vals], color=colors or C[0], width=0.7, edgecolor="white", linewidth=1)
        ax.set_xticks(x, names, rotation=35, ha="right")
        ax.set_title(label, color=INK)
        ax.grid(axis="x", visible=False)
    fig.suptitle(title, y=1.04, fontsize=9)
    fig.tight_layout()
    save(fig, fname)


def trajectories(series: dict, keys, title, fname, ncols=4):
    keys = [(k, lab) for k, lab in keys if any(k in (r[0] if r else {}) for r in series.values() if r)]
    if not keys:
        return
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.85 * ncols, 1.55 * nrows), squeeze=False)
    for ax in axes.flat[len(keys):]:
        ax.axis("off")
    for ax, (k, lab) in zip(axes.flat, keys):
        for i, (name, log) in enumerate(series.items()):
            if not log:
                continue
            ax.plot([r.get("update", r.get("step")) for r in log], [r.get(k, np.nan) for r in log], color=C[i], label=name,
                    marker="o" if len(log) <= 10 else None)
        ax.set_title(lab, color=INK)
        ax.set_xlabel("update")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    if len(series) > 1:
        axes.flat[0].legend(loc="best")
    fig.suptitle(title, y=1.02, fontsize=9)
    fig.tight_layout()
    save(fig, fname)


# ------------------------------------------------------------------------------- Task 1
def task1():
    print("Task 1")
    names = [("sft", "SFT"), ("beta_0p03", "β=0.03 (600)"), ("beta_0p1", "β=0.10 (600)"), ("beta_0p3", "β=0.30 (600)"),
             ("standard", "standard β=0.10 (1 ep)"), ("length_balanced", "length-bal. (1 ep)")]
    rows, ev = [], {}
    for key, lab in names:
        e = load(f"task1_dpo/eval_{key}.json")
        if e is None:
            continue
        ev[lab] = e
        p, g = e.get("heldout_preference", {}), e.get("generation", {})
        rows.append([lab, f(p.get("dpo_loss")), f(p.get("preference_accuracy")), f(p.get("mean_dpo_margin"), 2),
                     f(g.get("kl"), 4), f(g.get("reward", {}).get("mean")), f(g.get("response_tokens", {}).get("mean"), 1),
                     f(g.get("response_tokens", {}).get("std"), 1), f(g.get("response_tokens", {}).get("iqr"), 1),
                     f(g.get("truncation_rate"), 2), f(e.get("word_limit", {}).get("compliance_rate"), 2)])
    if rows:
        LINES.append("## Task 1 - DPO summary (held-out pairs; generation on first N held-out prompts)\n")
        LINES.append("Short β forks use 600 training pairs; standard and length-balanced use one full epoch (budgets differ).\n")
        table(["condition", "DPO loss", "pref. acc", "margin", "KL", "RM reward", "len mean", "len std", "len IQR", "trunc", "word-limit"], rows)
        bar_panels(list(ev), [
            ("held-out pref. accuracy", [ev[k].get("heldout_preference", {}).get("preference_accuracy") for k in ev]),
            ("KL to reference", [ev[k]["generation"]["kl"] for k in ev]),
            ("reward-model score", [ev[k]["generation"]["reward"]["mean"] for k in ev]),
            ("response tokens", [ev[k]["generation"]["response_tokens"]["mean"] for k in ev]),
        ], "Task 1: DPO conditions", "t1_dpo_summary")

    la = load("task1_dpo/length_analysis.json")
    if la:
        LINES.append("## Task 1 - length confounding\n")
        d = la["dataset"]
        table(["dataset", "frac chosen longer", "mean token diff (c-r)", "length-only acc"],
              [[k, f(v["frac_chosen_longer"]), f(v["mean_token_diff_chosen_minus_rejected"], 1), f(v["length_only_predictor_accuracy"])] for k, v in d.items()])
        strata = ["preferred_longer", "length_matched", "rejected_longer"]
        se = la["stratified_eval"]
        table(["model"] + [f"acc {s}" for s in strata] + ["corr(margin, len diff)"],
              [[m] + [f(se[m].get(s, {}).get("preference_accuracy")) for s in strata] + [f(se[m]["corr_margin_vs_length_diff"])] for m in se])
        fig, ax = plt.subplots(figsize=(3.4, 2.0))
        x, w = np.arange(len(strata)), 0.38
        for i, m in enumerate(se):
            ax.bar(x + (i - 0.5) * w, [se[m].get(s, {}).get("preference_accuracy", np.nan) for s in strata], w, color=C[i],
                   label=m.replace("_", "-"), edgecolor="white", linewidth=1)
        ax.axhline(0.5, color=INK2, lw=0.8, ls="--")
        ax.set_xticks(x, [s.replace("_", "-") for s in strata])
        ax.set_ylabel("held-out pref. accuracy")
        ax.legend(loc="upper right")
        ax.grid(axis="x", visible=False)
        save(fig, "t1_length_strata")

    logs = {lab: load(f"task1_dpo/train_{k}_log.jsonl") for k, lab in names[1:]}
    logs = {k: [dict(r, update=r["step"]) for r in v] for k, v in logs.items() if v}
    trajectories(logs, [("loss", "train DPO loss"), ("train_preference_accuracy", "train pref. acc"),
                        ("chosen_implicit_reward", "β·logratio(y+)"), ("rejected_implicit_reward", "β·logratio(y−)")],
                 "Task 1: DPO training (per optimizer step)", "t1_train_curves")


# ------------------------------------------------------------------------------- Task 2
PPO_KEYS = [("reward", "learned reward"), ("kl", "KL to reference"), ("policy_loss", "policy loss"), ("value_loss", "value loss"),
            ("entropy", "entropy"), ("clip_fraction", "clip fraction"), ("grad_norm", "policy grad norm"), ("response_length", "response tokens")]


def task2():
    print("Task 2")
    std = load("task2_ppo/standard/train_log.jsonl")
    if std:
        trajectories({"standard": std}, PPO_KEYS, "Task 2: standard PPO continuation", "t2_ppo_standard")
        s = load("task2_ppo/standard/summary.json") or {}
        LINES.append("## Task 2 - standard PPO continuation\n")
        table(["updates", "wall-clock (s)", "peak VRAM (GiB)", "GPU", "final reward", "final KL", "mean clip frac", "mean value EV"],
              [[s.get("updates"), f(s.get("wall_clock_s"), 0), f(s.get("peak_vram_gib"), 2), s.get("gpu"), f(std[-1]["reward"]),
                f(std[-1]["kl"], 4), f(np.mean([r["clip_fraction"] for r in std])), f(np.nanmean([r["value_explained_variance"] for r in std]), 2)]])
    cc = load("task2_ppo/clipping_cached_batch.json")
    if cc:
        LINES.append("## Task 2 - cached-batch clipping geometry\n")
        LINES.append(f"{cc['n_rollouts']} rollouts, {cc['n_tokens']} response tokens, skipped {len(cc['skipped'])}.\n")
        rows = []
        fig, axes = plt.subplots(1, 3, figsize=(6.4, 1.9))
        for i, (eps, steps) in enumerate(cc["per_epsilon"].items()):
            st = [s["step"] for s in steps]
            for ax, k in zip(axes, ["clip_fraction", "affected_fraction", "approx_kl_old_new"]):
                ax.plot(st, [s[k] for s in steps], color=C[i], marker="o", label=f"ε={eps}")
            rows.append([eps] + [f(steps[j][k], 4) for j in (0, -1) for k in ("clip_fraction", "affected_fraction", "clipped_surrogate")])
        for ax, t in zip(axes, ["clip fraction (outside band)", "affected (grad-zeroed) fraction", "approx KL(old‖new)"]):
            ax.set_title(t, color=INK)
            ax.set_xlabel("inner step on cached batch")
        axes[0].legend()
        fig.tight_layout()
        save(fig, "t2_clip_cached")
        table(["ε", "clip@0", "affected@0", "L_clip@0", "clip@end", "affected@end", "L_clip@end"], rows)
    forks = load("task2_ppo/clipping_forks.json")
    if forks:
        LINES.append("## Task 2 - clipping forks (matched short continuations)\n")
        table(["ε", "held-out R", "KL", "entropy", "len", "max update-KL", "mean clip frac", "policy-loss std", "max grad norm"],
              [[x["clip_epsilon"], f(x["heldout"]["reward"]["mean"]), f(x["heldout"]["kl"], 4), f(x["heldout"]["entropy"]),
                f(x["heldout"]["response_tokens"]["mean"], 0), f(x["stability"]["max_update_kl"], 5), f(x["stability"]["mean_clip_fraction"]),
                f(x["stability"]["policy_loss_std"], 4), f(x["stability"]["max_grad_norm"], 3)] for x in forks])
        trajectories({f"ε={x['clip_epsilon']:g}": load(f"task2_ppo/{x['run']}/train_log.jsonl") for x in forks},
                     [("reward", "learned reward"), ("kl", "KL to reference"), ("clip_fraction", "clip fraction"), ("approx_kl_old_new", "update KL(old‖new)")],
                     "Task 2: clipping forks", "t2_clip_forks")
    kl = load("task2_ppo/kl_ablation.json")
    if kl:
        LINES.append("## Task 2 - KL pressure (held-out, common protocol)\n")
        m = kl["midpoint"]
        rows = [["midpoint", f(m["reward"]["mean"]), f(m["kl"], 4), f(m["entropy"]), f(m["response_tokens"]["mean"], 0), f(m["truncation_rate"], 2)]]
        rows += [[f"β_KL={x['kl_beta']:g}", f(x["heldout"]["reward"]["mean"]), f(x["heldout"]["kl"], 4), f(x["heldout"]["entropy"]),
                  f(x["heldout"]["response_tokens"]["mean"], 0), f(x["heldout"]["truncation_rate"], 2)] for x in kl["forks"]]
        table(["condition", "held-out R", "KL", "entropy", "len", "trunc"], rows)
        trajectories({f"β_KL={x['kl_beta']:g}": load(f"task2_ppo/{x['run']}/train_log.jsonl") for x in kl["forks"]},
                     [("reward", "learned reward"), ("kl", "KL to reference"), ("entropy", "entropy"), ("response_length", "response tokens")],
                     "Task 2: KL-pressure forks", "t2_kl_forks")


# ------------------------------------------------------------------------------- Task 3
def task3():
    print("Task 3")
    std = load("task3_grpo/standard/train_log.jsonl")
    if std:
        trajectories({"standard": std}, [("reward", "reward"), ("kl", "KL to reference"), ("group_reward_std", "within-group reward std"),
                                         ("frac_zero_std_groups", "uninformative groups"), ("policy_loss", "policy loss"),
                                         ("grad_norm", "grad norm"), ("entropy", "entropy"), ("response_length", "response tokens")],
                     "Task 3: standard GRPO continuation", "t3_grpo_standard")
        s = load("task3_grpo/standard/summary.json") or {}
        LINES.append("## Task 3 - standard GRPO continuation\n")
        table(["updates", "wall-clock (s)", "peak VRAM (GiB)", "GPU", "mean reward", "mean group std", "uninformative frac", "final KL"],
              [[s.get("updates"), f(s.get("wall_clock_s"), 0), f(s.get("peak_vram_gib"), 2), s.get("gpu"), f(np.mean([r["reward"] for r in std])),
                f(np.mean([r["group_reward_std"] for r in std])), f(np.mean([r["frac_zero_std_groups"] for r in std])), f(std[-1]["kl"], 4)]])
    gs = load("task3_grpo/group_size_analysis.json")
    if gs:
        LINES.append("## Task 3 - equal-generation group-size study (permutation means)\n")
        LINES.append(f"Binning: {gs['binning_rule']}.\n")
        rows = []
        for k, e in gs["by_k"].items():
            for b in ["all", "hard", "medium", "easy"]:
                a = e[b]["permutation_mean"]
                rows.append([k, b, e[b]["fixed_partition"]["n_groups"], f(a["informative_rate"]), f(a["weak_rate@0.1"]), f(a["mean_group_std"]),
                             f(a["centered_signal_var"]), f(a["adv_corr_vs_k8"]), f(a["sign_agree_vs_k8"]),
                             f(e["binarized"][b]["permutation_mean"]["informative_rate"])])
        table(["K", "bin", "groups", "informative", "weak@0.1", "group std", "centered var", "adv corr vs K8", "sign agree", "binarized informative"], rows)
        ks = list(gs["by_k"])
        fig, axes = plt.subplots(1, 3, figsize=(6.4, 1.9))
        for i, b in enumerate(["hard", "medium", "easy"]):
            for ax, (key, src) in zip(axes, [("weak_rate@0.1", None), ("adv_corr_vs_k8", None), ("informative_rate", "binarized")]):
                vals = [(gs["by_k"][k][src][b] if src else gs["by_k"][k][b])["permutation_mean"][key] for k in ks]
                ax.plot([int(k) for k in ks], vals, color=C[i], marker="o", label=b)
        for ax, t in zip(axes, ["weak groups (std<0.1)", "adv. corr. vs K=8 baseline", "informative, binarized reward"]):
            ax.set_title(t, color=INK)
            ax.set_xlabel("K")
            ax.set_xticks([int(k) for k in ks])
        axes[0].legend()
        fig.tight_layout()
        save(fig, "t3_group_size")
    nc = load("task3_grpo/normalization_comparison.json")
    if nc:
        LINES.append("## Task 3 - canonical GRPO vs Dr.-GRPO normalization\n")
        rows = []
        for lt, d in nc["forks"].items():
            h, g = d["heldout"], d["train_gradient_mass_by_length"]
            m = nc["measured_gradient_norms_on_cache"][lt]
            rows.append([lt, f(h["reward"]["mean"]), f(h["kl"], 4), f(h["response_tokens"]["mean"], 0), f(g["short"]["gradient_mass_share"]),
                         f(g["long"]["gradient_mass_share"]), f(g["corr_length_vs_mass"]), f(m["long_over_short_mean_norm"], 2),
                         f(m["corr_tokens_vs_grad_norm"])])
        table(["loss", "held-out R", "KL", "len", "train mass short", "train mass long", "corr(len, mass)", "measured long/short |g|", "corr(len, |g|)"], rows)
        fig, ax = plt.subplots(figsize=(3.2, 1.9))
        bins, w = ["short", "medium", "long"], 0.38
        for i, lt in enumerate(nc["measured_gradient_norms_on_cache"]):
            m = nc["measured_gradient_norms_on_cache"][lt]
            ax.bar(np.arange(3) + (i - 0.5) * w, [m[b]["share_of_total"] for b in bins], w, color=C[i], label=lt, edgecolor="white", linewidth=1)
        ax.set_xticks(range(3), [f"{b} completions" for b in bins])
        ax.set_ylabel("share of total |grad|")
        ax.legend()
        ax.grid(axis="x", visible=False)
        save(fig, "t3_norm_grad_share")
        trajectories({lt: [dict(zip(d["train_trajectory"], v), update=i + 1) for i, v in enumerate(zip(*d["train_trajectory"].values()))]
                      for lt, d in nc["forks"].items()},
                     [("reward", "reward"), ("kl", "KL to reference"), ("response_length", "response tokens"), ("grad_norm", "grad norm")],
                     "Task 3: normalization forks", "t3_norm_forks")


# ------------------------------------------------------------------------------- Task 4
def task4():
    print("Task 4")
    s = load("task4_safety/safety_summary.json")
    if not s:
        return
    labels = ["SAFE_ANSWER", "OVER_REFUSAL", "JUSTIFIED_REFUSAL", "UNSAFE_COMPLIANCE", "AMBIGUOUS"]
    pp = s["per_policy"]
    LINES.append("## Task 4 - safety calibration (AI judge)\n")
    table(["policy", "safe answer", "safe over-refusal", "unsafe compliance", "justified refusal", "ambiguous", "judge class-mismatch safe/unsafe", "mean tokens"],
          [[p, f(d["safe_answer_rate"]), f(d["over_refusal_rate"]), f(d["unsafe_compliance_rate"]), f(d["justified_refusal_rate"]),
            f(d["ambiguous_rate"]), f"{f(d['judge_class_mismatch_rate_safe'])}/{f(d['judge_class_mismatch_rate_unsafe'])}",
            f(d["mean_response_tokens"], 1)] for p, d in pp.items()])
    import csv
    cat = list(csv.DictReader((RES / "task4_safety/category_label_distribution.csv").open(encoding="utf-8")))
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4))
    for ax, cls in zip(axes, ["SAFE", "UNSAFE"]):
        rows = [r for r in cat if r["benchmark_class"] == cls]
        pols = list(pp)
        bottoms = np.zeros(len(pols))
        for j, lab in enumerate(labels):
            vals = np.array([np.mean([float(r[lab]) for r in rows if r["policy"] == p]) if any(r["policy"] == p for r in rows) else 0 for p in pols])
            ax.bar(range(len(pols)), vals, bottom=bottoms, color=C[j], label=lab.replace("_", " ").lower(), edgecolor="white", linewidth=1, width=0.7)
            bottoms += vals
        ax.set_xticks(range(len(pols)), [p.upper() for p in pols])
        ax.set_title(f"{cls.lower()} prompts: judge label share", color=INK)
        ax.grid(axis="x", visible=False)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    save(fig, "t4_safety_labels")
    a = s.get("manual_audit")
    if a:
        LINES.append("## Task 4 - manual audit\n")
        LINES.append(f"n={a['n_labelled']}, agreement={f(a['agreement'])}, Cohen κ={f(a['cohen_kappa'])}, judge ambiguous={f(a['judge_ambiguous_rate'])}, "
                     f"manual ambiguous={f(a['manual_ambiguous_rate'])}, agreement by class={ {k: round(v, 3) for k, v in a['agreement_by_class'].items()} }\n")
        conf = a["confusion_manual_rows_judge_cols"]
        cols = list(next(iter(conf.values())))
        table(["manual \\ judge"] + cols, [[r] + [conf[r][c] for c in cols] for r in conf])
        table(["policy", "agreement", "judge unsafe-compl. (audit)", "manual unsafe-compl. (audit)", "judge over-refusal (audit)", "manual over-refusal (audit)"],
              [[p, f(d["agreement"]), f(d["judge_rates_on_audit"]["unsafe_compliance_rate"]), f(d["manual_rates_on_audit"]["unsafe_compliance_rate"]),
                f(d["judge_rates_on_audit"]["over_refusal_rate"]), f(d["manual_rates_on_audit"]["over_refusal_rate"])] for p, d in a["per_policy"].items()])


# ------------------------------------------------------------------------------- Task 5
def task5():
    print("Task 5")
    rows = []
    for ds in ["gsm", "transfer"]:
        e = load(f"task5_feedback/eval_{ds}.json")
        if not e:
            continue
        for p, d in e["policies"].items():
            pw = d.get("pairwise_vs_sft", {})
            rows.append([ds, p, f(d["exact_accuracy"]), f(d["format_compliance"]), f(d["response_tokens"]["mean"], 0), f(d["truncation_rate"], 2),
                         f(pw.get("win_rate_vs_sft")), pw.get("ties", "-"), f(pw.get("verifier_judge_agreement_all")),
                         f(pw.get("verifier_judge_agreement_when_verifier_decisive")), f(d.get("accuracy_drop_from_gsm"))])
    if rows:
        LINES.append("## Task 5 - in-domain (GSM8K) and transfer (SVAMP)\n")
        table(["set", "policy", "exact acc", "format", "len", "trunc", "AI win vs SFT", "ties", "verifier-judge agree", "agree | verifier decisive", "acc drop"], rows)
    d = load("task5_feedback/diagnostic_summary.json")
    if d:
        LINES.append("## Task 5 - controlled diagnostics (each perturbation vs clean_correct)\n")
        bp = d["by_perturbation"]
        table(["perturbation", "axis", "RLVR better/tie/wrong", "RLAIF better/tie/wrong", "judge order-consistency"],
              [[x, v["axis"], "/".join(f(v["rlvr"][k], 2) for k in ("better_rate", "tie_rate", "wrong_rate")),
                "/".join(f(v["rlaif"][k], 2) for k in ("better_rate", "tie_rate", "wrong_rate")), f(v["rlaif_order_consistency"], 2)] for x, v in bp.items()])
        LINES.append(f"S_reason: RLVR {f(d['S_reason_rlvr'])}, RLAIF {f(d['S_reason_rlaif'])}; S_outcome: RLVR {f(d['S_outcome_rlvr'])}, RLAIF {f(d['S_outcome_rlaif'])}.\n")
        table(["variant", "mean exact reward", "mean RLAIF group reward"],
              [[k, f(v["mean_exact_reward"]), f(v["mean_rlaif_group_reward"])] for k, v in d["by_variant"].items()])
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.1), sharey=True)
        pert = list(bp)
        for ax, mech in zip(axes, ["rlvr", "rlaif"]):
            bottom = np.zeros(len(pert))
            for j, (k, lab) in enumerate([("better_rate", "prefers better"), ("tie_rate", "tie"), ("wrong_rate", "prefers worse")]):
                vals = np.array([bp[x][mech][k] for x in pert])
                ax.bar(range(len(pert)), vals, bottom=bottom, color=C[j], label=lab, edgecolor="white", linewidth=1, width=0.7)
                bottom += vals
            ax.set_xticks(range(len(pert)), [x.replace("_correct", "").replace("_final", "").replace("_", " ") for x in pert], rotation=25, ha="right")
            ax.set_title(mech.upper(), color=INK)
            ax.grid(axis="x", visible=False)
        axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        fig.tight_layout()
        save(fig, "t5_diagnostics")
    c = load("task5_feedback/feedback_comparison.json")
    if c and c.get("cost"):
        k = c["cost"]
        LINES.append(f"Cost per K={k['K']} group on {k['gpu']}: RLVR {k['rlvr_seconds_per_group'] * 1e3:.3f} ms ({k['rlvr_calls_per_group']} verifier calls) vs "
                     f"RLAIF {k['rlaif_seconds_per_group']:.2f} s ({k['rlaif_judge_calls_per_group']} judge generations).\n")


def main():
    for fn in (task1, task2, task3, task4, task5):
        try:
            fn()
        except Exception as exc:  # keep going so one malformed result doesn't block the rest
            print(f"  !! {fn.__name__} failed: {exc!r}")
    out = ROOT / "report" / "tables.md"
    out.write_text("# Result digest (auto-generated by report/make_figures.py)\n\n" + "\n".join(LINES), encoding="utf-8")
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
