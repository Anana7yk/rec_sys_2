"""No fitting: conditional RM evaluation in the recommendable action catalog."""

import sys, pickle, json
from pathlib import Path
import numpy as np, pandas as pd

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B.parents[1] / "recsys"))
from data2 import weights, dump
from reward3 import predict, error_frame, matched, metrics, ci, Evaluator


def summarize_domain(root, out):
    ids = np.load(root / "catalog_ids.npy")
    asset = dict(np.load(root / "reward/asset.npz"))
    with (root / "reward/specs.pkl").open("rb") as f:
        specs = pickle.load(f)
    evaluator = Evaluator(root, asset, ids)
    rows = []
    effects = []
    pairs = []
    for split, path in [
        ("validation", root / "reward_data/val_raw.npz"),
        ("final", root / "final/reward_data/holdout_raw.npz"),
    ]:
        data = dict(np.load(path))
        mask = np.isin(data["item"], ids)
        d = {k: v[mask] for k, v in data.items()}
        d["w"] = weights(d["uid"])
        preds = {k: predict(d, asset, s) for k, s in specs.items()}
        preds["evaluator"] = evaluator.raw(d)
        err = {}
        pa = {}
        for kind, p in preds.items():
            e = error_frame(d, p).set_index("uid")
            err[kind] = e
            pair = matched(d, p)
            pa[kind] = pair.set_index("uid")
            rows.append(
                dict(
                    split=split,
                    kind=kind,
                    rows=len(d["uid"]),
                    users=len(e),
                    users_atleast2=int((e.n >= 2).sum()),
                    matched_users=len(pair),
                    matched_pairs=int(pair.pairs.sum()),
                    concordance=float(pair.concordance.mean()),
                    within_mse=e.within.mean(),
                    between_mse=e.between.mean(),
                    **metrics(d["y"], p, d["w"]),
                )
            )
        for a, b in [
            ("joint", "history"),
            ("residual", "history"),
            ("candidate", "history"),
            ("joint", "candidate"),
            ("residual", "joint"),
        ]:
            for metric in ["mse", "within", "between"]:
                effects.append(
                    dict(
                        split=split,
                        a=a,
                        b=b,
                        metric=metric,
                        **ci(err[a][metric] - err[b][metric]),
                    )
                )
            for metric in ["concordance", "contrast_mse"]:
                pairs.append(
                    dict(
                        split=split,
                        a=a,
                        b=b,
                        metric=metric,
                        **ci(pa[a][metric] - pa[b][metric]),
                    )
                )
        for k in ["candidate", "joint", "residual"]:
            pairs.append(
                dict(
                    split=split,
                    a=k,
                    b="chance",
                    metric="concordance",
                    **ci(pa[k].concordance - 0.5),
                )
            )
    pd.DataFrame(rows).to_csv(out / "reward_action_domain.csv", index=False)
    pd.DataFrame(effects).to_csv(
        out / "reward_action_domain_bootstrap.csv", index=False
    )
    pd.DataFrame(pairs).to_csv(out / "reward_action_domain_matched.csv", index=False)
    import matplotlib.pyplot as plt

    f, ax = plt.subplots(1, 2, figsize=(11, 4))
    data = pd.DataFrame(rows)
    kinds = list(preds)
    pos = np.arange(len(kinds))
    for a, col in zip(ax, ["rmse", "within_mse"]):
        for split, shift in [("validation", -0.18), ("final", 0.18)]:
            a.bar(
                pos + shift,
                data[data.split == split].set_index("kind").loc[kinds, col],
                width=0.36,
                label=split,
            )
        a.set_xticks(pos, kinds, rotation=45)
        a.set_title("In-catalog " + col)
    ax[0].legend()
    f.tight_layout()
    f.savefig(out.parent / "figures/reward_action_domain.png")
    f.savefig(out.parent / "figures/reward_action_domain.svg")
    plt.close(f)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=B / "runs/study_03")
    p.add_argument("--reports", type=Path, default=B.parents[1] / "validation/recomputed/study03")
    a = p.parse_args()
    (a.reports / "tables").mkdir(parents=True, exist_ok=True)
    (a.reports / "figures").mkdir(parents=True, exist_ok=True)
    summarize_domain(a.out, a.reports / "tables")
