"""Distribution audit on aligned user-macro denominators; never reads final cohort."""

import sys, json
from pathlib import Path
import numpy as np, pandas as pd, polars as pl

B = Path(__file__).resolve().parent
R = B.parents[1]
sys.path.insert(0, str(B.parents[1] / "recsys"))
from data2 import lookup, dump

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, default=B / "runs/study_03")
args = parser.parse_args()
root = args.out
cfg = json.loads((root / "config.json").read_text())
ids = np.load(root / "catalog_ids.npy")
cat = pd.read_csv(root / "catalog.csv")
users = cfg["dev_user_ids"]
ev = (
    pl.scan_parquet(R / "data/raw/multi_event.parquet")
    .filter(
        (pl.col("event_type") == "listen")
        & pl.col("uid").is_in(users)
        & (pl.col("timestamp") < cfg["train_end"])
    )
    .collect()
    .sort(["uid", "timestamp"])
)
frames = []
userrows = []
for f in ev.partition_by("uid", maintain_order=True):
    u = int(f["uid"][0])
    t = f["timestamp"].to_numpy()
    ts = np.unique(t)
    ok = np.searchsorted(ts, t) >= cfg["min_history"]
    target = lookup(ids, f["item_id"].to_numpy()[ok]) - 3
    known = target >= 0
    userrows.append(
        dict(
            uid=u,
            eligible_events=int(ok.sum()),
            known_events=int(known.sum()),
            known_fraction=float(known.mean()) if len(known) else np.nan,
        )
    )
    frames.append(pd.DataFrame(dict(uid=u, t=t[ok], target=target)))
full = pd.concat(frames, ignore_index=True)
pd.DataFrame(userrows).to_csv(root / "data/user_cap.csv", index=False)
rows = []
sets = {"full_eligible": full}
for split in ["train", "val"]:
    d = dict(np.load(root / f"data/{split}_raw.npz"))
    sets["fixed_" + split] = pd.DataFrame(
        dict(uid=d["uid"], t=d["t"], target=lookup(ids, d["item"]) - 3)
    )
for name, f in list(sets.items()):
    sets[name + "_known"] = f[f.target >= 0]
for name, f in sets.items():
    known = f.target >= 0
    i = f.target.clip(0).to_numpy()
    temp = f.assign(
        known=known.astype(float),
        popular10=((i < len(ids) * 0.1) & known).astype(float),
        duration=np.where(known, cat.duration.to_numpy()[i], np.nan),
        time=f.t / 1e6,
    )
    m = temp.groupby("uid")[["known", "popular10", "duration", "time"]].mean().mean()
    rows.append(
        dict(
            population=name,
            events=len(f),
            users=f.uid.nunique(),
            unique_targets=f.loc[known, "target"].nunique(),
            **m.to_dict(),
        )
    )
pd.DataFrame(rows).to_csv(root / "data/target_distributions.csv", index=False)
print(rows)
