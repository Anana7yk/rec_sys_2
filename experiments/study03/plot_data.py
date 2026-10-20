"""Train-only descriptive figures; no new fitting or policy decisions."""

import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B.parents[1] / "recsys"))
from data2 import lookup, weights

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--out',type=Path,default=B/'runs/study_03')
parser.add_argument('--reports',type=Path,default=B.parents[1]/'validation/recomputed/study03')
args=parser.parse_args()
root=args.out
out=args.reports/'figures'
out.mkdir(parents=True, exist_ok=True)
ids = np.load(root / "catalog_ids.npy")
fixed = dict(np.load(root / "data/train_raw.npz"))
target = lookup(ids, fixed["item"]) - 3
ok = target >= 0
fc = np.bincount(target[ok], minlength=len(ids))
full = np.load(root / "pool/positive_counts.npy")
users = pd.read_csv(root / "data/user_cap.csv")
rm = dict(np.load(root / "reward_data/rm_fit_raw.npz"))
asset = dict(np.load(root / "reward/asset.npz"))
cat = pd.read_csv(root / "catalog.csv")
p = np.load(root / "pool/target.npy", mmap_mode="r")
offset = np.load(root / "pool/offset.npy")
wp = np.repeat(1 / np.diff(offset), np.diff(offset))
fp = np.bincount(p, weights=wp, minlength=len(ids))
fp /= fp.sum()
fx = np.bincount(target[ok], weights=weights(fixed["uid"][ok]), minlength=len(ids))
fx /= fx.sum()
bins = np.array([0, 0.01, 0.1, 0.5, 1])
labels = ["top1%", "1–10%", "10–50%", "50–100%"]
cuts = (bins * len(ids)).astype(int)
pos = np.arange(4)
f, ax = plt.subplots(2, 3, figsize=(14, 8))
ax[0, 0].hist(
    np.log1p(users.eligible_events),
    bins=35,
    alpha=0.6,
    label="all eligible train events/user",
)
ax[0, 0].axvline(np.log1p(256), color="r", linestyle="--", label="fixed cap256")
ax[0, 0].set_xlabel("log(1 + events)")
ax[0, 0].set_ylabel("Users")
ax[0, 0].legend(fontsize=8)
for name, c in [("fixed", fc), ("full pool", full)]:
    sortedc = np.sort(c)
    cdf = np.arange(1, len(c) + 1) / len(c)
    ax[0, 1].plot(np.log1p(sortedc), cdf, label=f"{name}: {(c == 0).mean():.1%} zero")
ax[0, 1].set_xlabel("log(1 + positive target count/item)")
ax[0, 1].set_ylabel("Catalog CDF")
ax[0, 1].legend(fontsize=8)
ax[0, 2].hist(rm["y"], bins=np.linspace(0, 1, 21), weights=rm["w"] / rm["w"].sum())
ax[0, 2].set_xlabel("Clipped observed completion")
ax[0, 2].set_ylabel("User-weighted mass")
ax[1, 0].bar(
    pos - 0.18,
    [fx[a:b].sum() for a, b in zip(cuts[:-1], cuts[1:])],
    width=0.36,
    label="fixed known train",
)
ax[1, 0].bar(
    pos + 0.18,
    [fp[a:b].sum() for a, b in zip(cuts[:-1], cuts[1:])],
    width=0.36,
    label="full known train",
)
ax[1, 0].set_xticks(pos, labels, rotation=20)
ax[1, 0].set_ylabel("Target probability, uniform users")
ax[1, 0].legend(fontsize=8)
ax[1, 1].hist(
    np.log1p(cat.duration), bins=35, weights=fx, alpha=0.6, label="fixed known targets"
)
ax[1, 1].hist(
    np.log1p(cat.duration),
    bins=35,
    weights=fp,
    histtype="step",
    linewidth=2,
    label="full known targets",
)
ax[1, 1].set_xlabel("log(1 + seconds)")
ax[1, 1].set_ylabel("Target probability")
ax[1, 1].legend(fontsize=8)
prior = (asset["bin_sum"] + 1000 * asset["mu"]) / (asset["bin_count"] + 1000)
ax[1, 2].plot(np.arange(1, len(prior) + 1), prior, "o-")
ax[1, 2].axhline(
    float(asset["mu"]), linestyle=":", color="k", label="global train mean"
)
ax[1, 2].set_xlabel("Train item-duration decile")
ax[1, 2].set_ylabel("Shrunk observed completion mean")
ax[1, 2].legend(fontsize=8)
for a in ax.ravel():
    a.grid(alpha=0.2)
f.tight_layout()
f.savefig(out / "data_audit.png", dpi=140)
f.savefig(out / "data_audit.svg")
plt.close(f)
pd.DataFrame(
    dict(
        duration_bin=np.arange(len(prior)),
        bin_response_mean=prior,
        correction=prior - float(asset["mu"]),
        effective_count=asset["bin_count"],
    )
).to_csv(args.reports / "duration_reward_prior.csv", index=False)
