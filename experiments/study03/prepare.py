"""Prepare all eligible training targets, one strict context per timestamp group."""

import sys, json, time, shutil, hashlib
from pathlib import Path
import numpy as np, polars as pl

B = Path(__file__).resolve().parent
R = B.parents[1]
sys.path.insert(0, str(B.parents[1] / "recsys"))
from data2 import pack, lookup, dump, prepare_raw


def build_pool(cfg, out):
    ids = np.load(out / "catalog_ids.npy")
    users = cfg["dev_user_ids"]
    ev = (
        pl.scan_parquet(R / "data/raw/multi_event.parquet")
        .filter(
            (pl.col("event_type") == "listen")
            & pl.col("uid").is_in(users)
            & (pl.col("timestamp") < cfg["train_end"])
        )
        .collect()
        .sort(
            [
                "uid",
                "timestamp",
                "item_id",
                "played_ratio_pct",
                "track_length_seconds",
                "is_organic",
            ]
        )
    )
    count = ev.select(pl.struct(["uid", "timestamp"]).n_unique()).item()
    pool = out / "pool"
    if pool.exists():
        raise FileExistsError(pool)
    pool.mkdir()
    table = np.full(int(ids.max()) + 1, 2, dtype="uint32")
    table[ids] = np.arange(len(ids), dtype="uint32") + 3

    def fastmap(a):
        z = table[a.clip(0, len(table) - 1)].copy()
        z[a >= len(table)] = 2
        z[a == -1] = 0
        z[a == -2] = 1
        z[a == -3] = 2
        return z

    xx = np.lib.format.open_memmap(
        pool / "x.npy", "w+", dtype="uint32", shape=(count, cfg["context"])
    )
    gg = np.lib.format.open_memmap(
        pool / "g.npy", "w+", dtype="uint8", shape=(count, cfg["context"])
    )
    ci = []
    yy = []
    uu = []
    tt = []
    off = [0]
    available = []
    pos = 0
    min_tmargin = np.inf
    start = time.monotonic()
    for f in ev.partition_by("uid", maintain_order=True):
        uid = int(f["uid"][0])
        t = f["timestamp"].to_numpy().astype("int64")
        raw = f["item_id"].to_numpy().astype("int64")
        z = lookup(ids, raw)
        ts, first, ns = np.unique(t, return_index=True, return_counts=True)
        groups = [raw[i : i + n] for i, n in zip(first, ns)]
        gi = np.searchsorted(ts, t)
        for j in range(len(ts)):
            if j < cfg["min_history"]:
                xx[pos + j] = 0
                gg[pos + j] = 0
                continue
            pr, g, ht, _ = pack(groups, ts, j, cfg["context"])
            xx[pos + j] = fastmap(pr)
            gg[pos + j] = g
            min_tmargin = min(min_tmargin, int((ts[j] - ht[ht >= 0]).min()))
        ok = (gi >= cfg["min_history"]) & (z >= 3)
        n = int(ok.sum())
        if n:
            ci.append((pos + gi[ok]).astype("uint32"))
            yy.append((z[ok] - 3).astype("uint32"))
            uu.append(np.full(n, uid, dtype="uint32"))
            tt.append(t[ok].astype("uint32"))
            available.append(uid)
            off.append(off[-1] + n)
        pos += len(ts)
        if len(available) % 200 == 0:
            print("POOL", len(available), pos, flush=True)
    xx.flush()
    gg.flush()
    for name, v in [
        ("context", np.concatenate(ci)),
        ("target", np.concatenate(yy)),
        ("uid", np.concatenate(uu)),
        ("t", np.concatenate(tt)),
        ("offset", np.array(off)),
        ("users", np.array(available)),
    ]:
        np.save(pool / f"{name}.npy", v)
    targets = np.concatenate(yy)
    oldy = lookup(ids, dict(np.load(out / "data/train_raw.npz"))["item"]) - 3
    oldcnt = np.bincount(oldy[oldy >= 0], minlength=len(ids))
    cnt = np.bincount(targets, minlength=len(ids))
    np.save(pool / "positive_counts.npy", cnt)
    v = dict(np.load(out / "data/val_raw.npz"))
    vy = lookup(ids, v["item"]) - 3
    audit = dict(
        source_train_events=ev.height,
        eligible_known_targets=len(targets),
        groups=pos,
        known_fraction_source=float(
            (lookup(ids, ev["item_id"].to_numpy()) >= 3).mean()
        ),
        fixed_positive_items=int((oldcnt > 0).sum()),
        pool_positive_items=int((cnt > 0).sum()),
        catalog=len(ids),
        strict_min_history_time_margin=min_tmargin,
        seconds=time.monotonic() - start,
        actual_drawn_support="logged separately for each optimizer run",
    )
    for n, c in [("fixed", oldcnt), ("pool", cnt)]:
        audit[n + "_val_cold_target_mass"] = float(
            np.average((vy >= 0) & (c[vy.clip(0)] == 0), weights=v["w"])
        )
    dump(out / "data/pool_audit.json", audit)
    print(audit, flush=True)


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=B / "runs/reproduce")
    a = p.parse_args()
    cfg = json.loads((B.parents[1] / "configs/study03/study.json").read_text())
    out = a.out
    from reference import rebuild

    rebuild(cfg, out)
    dump(out / "config.json", cfg)
    prepare_raw(cfg, out / "reward_data")
    build_pool(cfg, out)


if __name__ == "__main__":
    main()
