"""Actual small CPU pipeline on development users only; no research final users."""

import sys, json, time, argparse
from pathlib import Path
import numpy as np, polars as pl, torch

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B.parents[1] / "recsys"))
sys.path.insert(0, str(B))
from data2 import dump, prepare_raw
from prepare import build_pool
from reward3 import fit as fit_reward
from train import fit as fit_policy
from run import assets, policies, finalize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=B / "runs/smoke")
    a = p.parse_args()
    root = a.out
    root.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    cfg = json.loads((B.parents[1] / "configs/study03/study.json").read_text())
    dev = cfg["dev_user_ids"]
    cfg.update(
        dev_user_ids=dev[:16],
        holdout_user_ids=dev[16:24],
        seeds=[11],
        rm_per_user=32,
        inner_per_user=4,
        train_per_user=32,
        val_per_user=2,
        holdout_per_user=2,
        rm_final_per_user=4,
        dim=16,
        rl_steps=10,
    )
    dump(root / "config.json", cfg)
    prepare_raw(cfg, root / "data")
    prepare_raw(cfg, root / "reward_data")
    cat = (
        pl.read_parquet(root / "data/item_stats.parquet")
        .filter(pl.col("count") >= 2)
        .head(256)
    )
    cat.write_csv(root / "catalog.csv")
    np.save(root / "catalog_ids.npy", cat["item_id"].to_numpy())
    build_pool(cfg, root)
    fit_reward(root)
    from select_reward import select

    select(root)
    fit_policy(root, "seed_11/pure", 11, 16, 0.001, 0.01, 20, False, "cpu")
    args = assets(root, "cpu")
    cfg, ids, asset, specs, rm, items, d = args
    policies(root, cfg, rm, items, d, "cpu")
    finalize(root, *args, "cpu")
    dump(
        root / "SMOKE_COMPLETE.json",
        dict(
            seconds=time.monotonic() - start,
            scope="real development users; pseudo final only; CPU; not research evidence",
        ),
    )


if __name__ == "__main__":
    main()
