"""Read-only tables, paired user bootstrap and scientific figures; no model choices."""

import sys, json, pickle
from pathlib import Path
import numpy as np, pandas as pd, polars as pl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B.parents[1] / "recsys"))
from data2 import dump, lookup, weights
from reward3 import ci, predict, error_frame, matched
from reward2 import metrics, metadata, prediction_bins

root = B / "runs/study_03"
out = B.parents[1] / "validation/recomputed/study03"
fig = out / "figures"
fig.mkdir(parents=True, exist_ok=True)
tab = out / "tables"
tab.mkdir(parents=True, exist_ok=True)

PAIRS = [
    ("base", "fresh_count_ablation"),
    ("pure", "fresh_marginal"),
    ("pure", "old_gpt"),
    ("base", "old_gpt"),
    ("base", "pure"),
    ("base", "context_count"),
    ("base", "ablate_pop"),
    ("base", "ablate_pop_tuned"),
    ("rl_h1_raw", "base"),
    ("rl_h1_raw", "sft"),
    ("rl_h1_raw", "exact"),
    ("rl_h1_raw", "mixed100"),
    ("rl_h5_raw", "rl_h1_raw"),
    ("rl_h1_duration", "rl_h1_raw"),
    ("exact_duration", "rl_h1_duration"),
]


def main():
    global root, out, fig, tab
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=root)
    p.add_argument("--reports", type=Path, default=out)
    args = p.parse_args()
    root = args.out
    out = args.reports
    fig = out / "figures"
    tab = out / "tables"
    fig.mkdir(parents=True, exist_ok=True)
    tab.mkdir(parents=True, exist_ok=True)
    final = root / "final"
    eligibility = []
    for seed in [11, 29, 47]:
        val = pd.read_csv(root / f"seed_{seed}/validation_metrics.csv")
        baseline = float(val[val.method == "base"].ndcg20.iloc[0])
        for r in val.to_dict("records"):
            eligibility.append(
                dict(
                    seed=seed,
                    method=r["method"],
                    ndcg20=r["ndcg20"],
                    baseline_ndcg20=baseline,
                    ratio=r["ndcg20"] / baseline,
                    admissible=r["ndcg20"] >= 0.95 * baseline,
                )
            )
    pd.DataFrame(eligibility).to_csv(tab / "validation_admissibility.csv", index=False)
    m = pd.read_csv(final / "metrics.csv")
    mean = m.groupby("method").mean(numeric_only=True)
    sd = m.groupby("method").std(numeric_only=True)
    mean.to_csv(tab / "main_mean.csv")
    sd.to_csv(tab / "main_seed_sd.csv")
    m.to_csv(tab / "main_seeds.csv", index=False)
    frames = {}
    for method in m.method.unique():
        fs = [
            pd.read_csv(final / f"{s}_{method}_events.csv")
            .groupby("uid")
            .mean(numeric_only=True)
            .assign(seed=s)
            for s in [11, 29, 47]
        ]
        frames[method] = pd.concat(fs).groupby("uid").mean(numeric_only=True)
    comparisons = []
    for a, b in PAIRS:
        if a not in frames or b not in frames:
            continue
        for metric in [
            "ndcg20",
            "recall20",
            "reward",
            "adjusted_reward",
            "evaluator_reward",
            "duration",
            "history_repeat",
            "top_decile",
            "logpop",
            "artist_distance",
            "disagreement",
        ]:
            if metric not in frames[a] or metric not in frames[b]:
                continue
            aa = frames[a][metric]
            bb = frames[b][metric]
            ix = aa.index.intersection(bb.index)
            comparisons.append(
                dict(
                    a=a, b=b, metric=metric, **ci((aa.loc[ix] - bb.loc[ix]).to_numpy())
                )
            )
    pd.DataFrame(comparisons).to_csv(tab / "paired_differences.csv", index=False)
    # Controlled cap experiment on exactly the same validation targets/users.
    f = (
        pd.read_csv(root / "pilot/fixed128/validation_per_event.csv")
        .groupby("uid")
        .mean(numeric_only=True)
    )
    d = (
        pd.read_csv(root / "pilot/dynamic128/validation_per_event.csv")
        .groupby("uid")
        .mean(numeric_only=True)
    )
    pd.DataFrame(
        [dict(metric=k, **ci(d[k] - f[k])) for k in ["ndcg20", "recall20", "ce"]]
    ).to_csv(tab / "cap_control_bootstrap.csv", index=False)
    pilots = []
    for p in sorted((root / "pilot").iterdir()):
        if not (p / "timing.json").exists():
            continue
        t = json.loads((p / "timing.json").read_text())
        log = pd.read_csv(p / "learning.csv")
        at = log.loc[log.ndcg20.idxmax()].to_dict()
        pilots.append(
            {"name": p.name, **t, **{f"selected_{k}": v for k, v in at.items()}}
        )
    pd.DataFrame(pilots).to_csv(tab / "pilot_summary.csv", index=False)
    # Real-prefix target strata; all comparisons use same targets, no conditional causal reading.
    groups = []
    for method, f in frames.items():
        events = pd.concat(
            [
                pd.read_csv(final / f"{s}_{method}_events.csv").assign(seed=s)
                for s in [11, 29, 47]
            ]
        )
        for g, mask in [
            ("all", np.ones(len(events), bool)),
            (
                "known_repeated",
                (events.in_catalog == 1) & (events.target_seen_history == 1),
            ),
            (
                "known_novel",
                (events.in_catalog == 1) & (events.target_seen_history == 0),
            ),
            ("OOV", events.in_catalog == 0),
        ]:
            sub = (
                events[mask]
                .groupby(["seed", "uid"])
                .mean(numeric_only=True)
                .groupby("uid")
                .mean(numeric_only=True)
            )
            groups.append(
                dict(
                    method=method,
                    group=g,
                    users=len(sub),
                    events_per_seed=int(mask.sum() / 3),
                    **{k: sub[k].mean() for k in ["ndcg20", "recall20", "reward"]},
                )
            )
    pd.DataFrame(groups).to_csv(tab / "target_strata.csv", index=False)
    # Candidate-signal diagnostics on observed interactions only.
    with (root / "reward/specs.pkl").open("rb") as f:
        specs = pickle.load(f)
    asset = dict(np.load(root / "reward/asset.npz"))
    fit = dict(np.load(root / "reward_data/rm_fit_raw.npz"))
    hcuts = np.quantile(fit["history_length"], [0.333, 0.667])
    ix = metadata(asset, fit["item"])
    pcuts = np.quantile(np.log1p(asset["pop"][ix]), [0.333, 0.667])
    dcuts = np.quantile(asset["duration"][ix], [0.333, 0.667])
    rmrows = []
    calrows = []
    pairrows = []
    for split, path in [
        ("validation", root / "reward_data/val_raw.npz"),
        ("final", final / "reward_data/holdout_raw.npz"),
    ]:
        raw = dict(np.load(path))
        ix = metadata(asset, raw["item"])
        groupings = dict(
            popularity=np.digitize(np.log1p(asset["pop"][ix]), pcuts),
            duration=np.digitize(asset["duration"][ix], dcuts),
            history_length=np.digitize(raw["history_length"], hcuts),
            item_response_support=np.digitize(asset["count"][ix], [1, 5, 20]),
        )
        pp = {}
        for kind, spec in specs.items():
            p = predict(raw, asset, spec)
            pp[kind] = p
            for group, labels in groupings.items():
                for level in np.unique(labels):
                    mask = labels == level
                    rmrows.append(
                        dict(
                            split=split,
                            kind=kind,
                            group=group,
                            level=int(level),
                            n=int(mask.sum()),
                            **metrics(
                                raw["y"][mask], p[mask], weights(raw["uid"][mask])
                            ),
                        )
                    )
            bins = prediction_bins(p, raw["w"])
            for k in np.unique(bins):
                mask = bins == k
                calrows.append(
                    dict(
                        split=split,
                        kind=kind,
                        bin=int(k),
                        n=int(mask.sum()),
                        prediction=np.average(p[mask], weights=raw["w"][mask]),
                        observed=np.average(raw["y"][mask], weights=raw["w"][mask]),
                    )
                )
        # Recompute per-user paired matched-context errors/accuracy for common pairs.
        pairs = {k: matched(raw, p).set_index("uid") for k, p in pp.items()}
        for candidate_a, candidate_b in [("joint", "candidate"), ("residual", "joint")]:
            common = pairs[candidate_a].index.intersection(pairs[candidate_b].index)
            for metric in ["concordance", "contrast_mse"]:
                pairrows.append(
                    dict(
                        split=split,
                        a=candidate_a,
                        b=candidate_b,
                        metric=metric,
                        **ci(
                            pairs[candidate_a].loc[common, metric]
                            - pairs[candidate_b].loc[common, metric]
                        ),
                    )
                )
        for kind in ["candidate", "joint", "residual"]:
            common = pairs[kind].index.intersection(pairs["history"].index)
            for metric in ["concordance", "contrast_mse"]:
                pairrows.append(
                    dict(
                        split=split,
                        a=kind,
                        b="history",
                        metric=metric,
                        **ci(
                            pairs[kind].loc[common, metric]
                            - pairs["history"].loc[common, metric]
                        ),
                    )
                )
            pairrows.append(
                dict(
                    split=split,
                    a=kind,
                    b="chance",
                    metric="concordance",
                    **ci(pairs[kind].concordance - 0.5),
                )
            )
    pd.DataFrame(rmrows).to_csv(tab / "reward_strata.csv", index=False)
    pd.DataFrame(calrows).to_csv(tab / "reward_calibration.csv", index=False)
    pd.DataFrame(pairrows).to_csv(tab / "reward_matched_bootstrap.csv", index=False)
    # Sequential diagnostics: same observed initial contexts, synthetic append clearly separate.
    h = pd.read_csv(final / "horizon.csv")
    agg = (
        h.groupby(["method", "mode", "step", "uid"])
        .mean(numeric_only=True)
        .groupby(["method", "mode", "step"])
        .mean()
    )
    agg.to_csv(tab / "horizon_steps.csv")
    hc = []
    for method in h.method.unique():
        a = (
            h[(h.method == method) & (h.step == 5) & (h["mode"] == "synthetic")]
            .groupby("uid")
            .mean(numeric_only=True)
        )
        b = (
            h[(h.method == method) & (h.step == 5) & (h["mode"] == "real_prefix")]
            .groupby("uid")
            .mean(numeric_only=True)
        )
        for metric in [
            "reward",
            "adjusted_reward",
            "evaluator_reward",
            "disagreement",
            "kl",
            "duration",
            "history_repeat",
            "known_fraction",
            "unmasked_list_repeat_mass",
        ]:
            hc.append(dict(method=method, metric=metric, **ci(a[metric] - b[metric])))
    pd.DataFrame(hc).to_csv(tab / "synthetic_minus_real_step5.csv", index=False)
    horizon_progress = []
    for method in h.method.unique():
        for mode in ["synthetic", "real_prefix"]:
            a = (
                h[(h.method == method) & (h["mode"] == mode) & (h.step == 5)]
                .groupby("uid")
                .mean(numeric_only=True)
            )
            b = (
                h[(h.method == method) & (h["mode"] == mode) & (h.step == 1)]
                .groupby("uid")
                .mean(numeric_only=True)
            )
            for metric in [
                "reward",
                "evaluator_reward",
                "disagreement",
                "kl",
                "duration",
                "unmasked_list_repeat_mass",
            ]:
                horizon_progress.append(
                    dict(
                        method=method,
                        mode=mode,
                        metric=metric,
                        **ci(a[metric] - b[metric]),
                    )
                )
    pd.DataFrame(horizon_progress).to_csv(tab / "horizon_step5_minus1.csv", index=False)

    beta = 0.1
    gamma = 0.9
    h["discounted_reward"] = gamma ** (h.step - 1) * h.reward
    h["discounted_evaluator_reward"] = gamma ** (h.step - 1) * h.evaluator_reward
    h["J"] = h.discounted_reward - beta * h.kl
    h["discounted_adjusted_reward"] = gamma ** (h.step - 1) * h.adjusted_reward
    h["J_adjusted"] = h.discounted_adjusted_reward - beta * h.kl
    returns = (
        h.groupby(["seed", "method", "mode", "uid"])[
            [
                "discounted_reward",
                "discounted_evaluator_reward",
                "J",
                "discounted_adjusted_reward",
                "J_adjusted",
            ]
        ]
        .sum()
        .reset_index()
    )
    returns.to_csv(tab / "horizon_returns_per_user.csv", index=False)
    returns.groupby(["method", "mode"]).mean(numeric_only=True).to_csv(
        tab / "horizon_returns_mean.csv"
    )
    bc = []
    for a, b in [
        ("rl_h1_raw", "base"),
        ("rl_h5_raw", "rl_h1_raw"),
        ("exact", "rl_h1_raw"),
        ("exact", "rl_h5_raw"),
        ("rl_h1_duration", "rl_h1_raw"),
        ("exact_duration", "rl_h1_duration"),
    ]:
        aa = (
            returns[(returns.method == a) & (returns["mode"] == "synthetic")]
            .groupby("uid")
            .mean(numeric_only=True)
        )
        bb = (
            returns[(returns.method == b) & (returns["mode"] == "synthetic")]
            .groupby("uid")
            .mean(numeric_only=True)
        )
        for metric in [
            "discounted_reward",
            "discounted_evaluator_reward",
            "J",
            "discounted_adjusted_reward",
            "J_adjusted",
        ]:
            bc.append(dict(a=a, b=b, metric=metric, **ci(aa[metric] - bb[metric])))
    pd.DataFrame(bc).to_csv(tab / "horizon_return_bootstrap.csv", index=False)
    sl = pd.read_csv(final / "slates.csv")
    sl.groupby(["method", "seed", "uid"]).mean(numeric_only=True).groupby(
        ["method", "seed"]
    ).mean(numeric_only=True).groupby("method").mean(numeric_only=True).to_csv(
        tab / "slate_mean.csv"
    )
    sc = []
    for a, b in [
        ("rl_h1_raw", "base"),
        ("rl_h5_raw", "rl_h1_raw"),
        ("exact", "rl_h1_raw"),
        ("base", "context_count"),
    ]:
        aa = sl[sl.method == a].groupby("uid").mean(numeric_only=True)
        bb = sl[sl.method == b].groupby("uid").mean(numeric_only=True)
        for metric in ["ndcg5", "recall5", "reward", "evaluator_reward", "duration"]:
            sc.append(dict(a=a, b=b, metric=metric, **ci(aa[metric] - bb[metric])))
    pd.DataFrame(sc).to_csv(tab / "slate_bootstrap.csv", index=False)
    # Timings are measured body compute vs elapsed; never equate RL actions with SFT labels.
    timings = []
    for p in root.rglob("*timing.json"):
        if "development" in str(p):
            continue
        timings.append(dict(path=str(p.relative_to(root)), **json.loads(p.read_text())))
    pd.DataFrame(timings).to_csv(tab / "timing.csv", index=False)
    from reward_domain_analysis import summarize_domain

    summarize_domain(root, tab)
    plots(mean, sd, agg, calrows)
    dump(
        out / "SUMMARY_COMPLETE.json",
        dict(
            methods=len(mean),
            final_users=int(m.n_users.iloc[0]),
            nominal_bootstrap=2000,
            seed_uncertainty="separate SD of3 policy seeds; common fitted reward models",
        ),
    )


def plots(mean, sd, h, calrows):
    plt.rcParams.update({"font.size": 10, "figure.dpi": 130})
    f, ax = plt.subplots(1, 3, figsize=(14, 4))
    for p in sorted((root / "pilot").iterdir()):
        if not (p / "learning.csv").exists():
            continue
        d = pd.read_csv(p / "learning.csv")
        ax[0].plot(d.step, d.ndcg20, "o-", label=p.name)
        ax[1].plot(d.step, d.train_probe, "o-", label=p.name)
        ax[2].plot(d.step, d.positive_items_drawn, "o-", label=p.name)
    for a, title in zip(
        ax,
        [
            "Validation NDCG@20",
            "Train-pool probe cross entropy",
            "Actually drawn positive item IDs",
        ],
    ):
        a.set_title(title)
        a.set_xlabel("Optimizer steps")
        a.grid(alpha=0.2)
    ax[0].legend(fontsize=8)
    f.tight_layout()
    f.savefig(fig / "supervised_control.png")
    f.savefig(fig / "supervised_control.svg")
    plt.close(f)
    f, ax = plt.subplots(1, 2, figsize=(11, 4))
    for s in [11, 29, 47]:
        d = pd.read_csv(root / f"seed_{s}/pure/learning.csv")
        ax[0].plot(d.step, d.train_probe, label=f"{s} train")
        ax[0].plot(d.step, d.ce, "--", label=f"{s} validation")
        ax[1].plot(d.step, d.ndcg20, "o-", label=str(s))
    ax[0].set_title("Main pure GPT: CE on fixed probes")
    ax[1].set_title("Main pure GPT: validation NDCG@20")
    for a in ax:
        a.set_xlabel("Steps")
        a.legend(fontsize=8)
        a.grid(alpha=0.2)
    f.tight_layout()
    f.savefig(fig / "supervised_main.png")
    f.savefig(fig / "supervised_main.svg")
    plt.close(f)
    f, ax = plt.subplots(figsize=(9, 5))
    methods = [
        "popularity",
        "fresh_marginal",
        "old_gpt",
        "pure",
        "base",
        "context_count",
        "fresh_count_ablation",
        "ablate_pop_tuned",
        "mixed100",
        "exact",
        "exact_duration",
        "rl_h1_raw",
        "rl_h5_raw",
        "rl_h1_duration",
        "sft",
    ]
    methods = [m for m in methods if m in mean.index]
    x = mean.loc[methods, "ndcg20"].to_numpy()
    y = mean.loc[methods, "reward"].to_numpy()
    front = [
        i
        for i in range(len(x))
        if not any(
            (x[j] >= x[i] and y[j] >= y[i] and (x[j] > x[i] or y[j] > y[i]))
            for j in range(len(x))
        )
    ]
    for i, m in enumerate(methods):
        ax.errorbar(
            x[i],
            y[i],
            xerr=sd.loc[m, "ndcg20"],
            yerr=sd.loc[m, "reward"],
            fmt="o",
            capsize=2,
            label=m,
        )
    front = sorted(front, key=lambda i: x[i])
    ax.plot(
        x[front],
        y[front],
        color="gray",
        linestyle=":",
        label="Empirical Pareto frontier",
    )
    ax.set_xlabel("NDCG@20 (next observed item)")
    ax.set_ylabel("Primary predicted reward of Top20")
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(alpha=0.2)
    f.tight_layout()
    f.savefig(fig / "pareto.png")
    f.savefig(fig / "pareto.svg")
    plt.close(f)
    f, ax = plt.subplots(1, 3, figsize=(13, 4))
    sel = [
        "base",
        "context_count",
        "mixed100",
        "exact",
        "rl_h1_raw",
        "rl_h5_raw",
        "rl_h1_duration",
    ]
    for a, col in zip(ax, ["duration", "history_repeat", "top_decile"]):
        a.barh(sel, mean.loc[sel, col])
        a.set_title(col)
        a.grid(axis="x", alpha=0.2)
    f.tight_layout()
    f.savefig(fig / "behavior.png")
    f.savefig(fig / "behavior.svg")
    plt.close(f)
    val = pd.read_csv(root / "reward/metrics.csv")
    test = pd.read_csv(root / "final/reward_metrics.csv")
    f, ax = plt.subplots(1, 3, figsize=(14, 4))
    kinds = ["mean", "history", "candidate", "joint", "residual", "evaluator"]
    positions = np.arange(len(kinds))
    for a, col in zip(ax, ["rmse", "within_mse", "between_mse"]):
        a.bar(
            positions - 0.18,
            val[val.split == "val"].set_index("kind").loc[kinds, col],
            width=0.36,
            label="validation",
        )
        a.bar(
            positions + 0.18,
            test.set_index("kind").loc[kinds, col],
            width=0.36,
            label="new final cohort",
        )
        a.set_xticks(positions, kinds, rotation=45)
        a.set_title(col)
    ax[0].legend()
    f.tight_layout()
    f.savefig(fig / "reward_action_signal.png")
    f.savefig(fig / "reward_action_signal.svg")
    plt.close(f)
    f, ax = plt.subplots(figsize=(5, 5))
    cal = pd.DataFrame(calrows)
    for k in ["history", "candidate", "joint", "residual"]:
        d = cal[(cal.split == "final") & (cal.kind == k)]
        ax.plot(d.prediction, d.observed, "o-", label=k)
    ax.plot([0, 1], [0, 1], "k:")
    ax.set_xlabel("Predicted completion")
    ax.set_ylabel("Observed completion")
    ax.legend()
    f.tight_layout()
    f.savefig(fig / "calibration.png")
    f.savefig(fig / "calibration.svg")
    plt.close(f)
    f, ax = plt.subplots(2, 4, figsize=(16, 8))
    for axis, col in zip(
        ax.ravel(),
        [
            "reward",
            "evaluator_reward",
            "disagreement",
            "kl",
            "duration",
            "logpop",
            "unmasked_list_repeat_mass",
            "history_repeat",
        ],
    ):
        for i, method in enumerate(["base", "rl_h1_raw", "rl_h5_raw", "exact"]):
            for mode, style in [("synthetic", "-"), ("real_prefix", "--")]:
                dd = h.loc[(method, mode)].reset_index()
                axis.plot(
                    dd.step,
                    dd[col],
                    style,
                    color=plt.get_cmap("tab10")(i),
                    label=f"{method} {mode}",
                )
        axis.set_title(col)
        axis.set_xlabel("Step")
        axis.grid(alpha=0.2)
    handles, labels = ax[0, 0].get_legend_handles_labels()
    f.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    f.tight_layout(rect=(0, 0.10, 1, 1))
    f.savefig(fig / "horizon.png")
    f.savefig(fig / "horizon.svg")
    plt.close(f)


if __name__ == "__main__":
    main()
