"""Read-only cumulative artist diversity for already generated sequences."""

from pathlib import Path
import numpy as np, pandas as pd, polars as pl

B = Path(__file__).resolve().parent
R = B.parents[1]
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, default=B / "runs/study_03")
parser.add_argument("--reports", type=Path, default=B.parents[1] / "validation/recomputed/study03")
args = parser.parse_args()
root = args.out
reports = args.reports
(reports / "tables").mkdir(parents=True, exist_ok=True)
ids = np.load(root / "catalog_ids.npy")
mapping = (
    pl.scan_parquet(R / "data/raw/artist_item_mapping.parquet")
    .filter(pl.col("item_id").is_in(ids.tolist()))
    .collect()
)
artists = {int(i): set() for i in ids}
for i, a in mapping.select("item_id", "artist_id").iter_rows():
    artists[int(i)].add(int(a))
h = pd.read_csv(root / "final/horizon.csv")
rows = []
for (seed, method, mode, uid), f in h.groupby(["seed", "method", "mode", "uid"]):
    previous = []
    for r in f.sort_values("step").to_dict("records"):
        previous.append(artists[int(ids[r["action"]])])
        pairs = [
            1 - len(a & b) / len(a | b)
            for j, a in enumerate(previous)
            if a
            for b in previous[j + 1 :]
            if b
        ]
        r["prefix_artist_distance"] = np.mean(pairs) if pairs else np.nan
        r["prefix_distinct_artists"] = len(set().union(*previous))
        rows.append(r)
a = pd.DataFrame(rows)
a.to_csv(reports / "tables/horizon_with_diversity.csv", index=False)
a.groupby(["method", "mode", "step", "uid"]).mean(numeric_only=True).groupby(
    ["method", "mode", "step"]
).mean().to_csv(reports / "tables/horizon_diversity_steps.csv")
