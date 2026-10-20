"""Rebuild frozen controlled policy samples from compact row selections + public raw data."""

import sys, json, hashlib, argparse
from pathlib import Path
import numpy as np, polars as pl

B = Path(__file__).resolve().parent
R = B.parents[1]
sys.path.insert(0, str(B.parents[1] / "recsys"))
from data2 import pack, feedback, weights, dump


def array_digest(a):
    return hashlib.sha256(
        str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()
    ).hexdigest()


def rebuild(cfg, out):
    out.mkdir(parents=True, exist_ok=False)
    dest = out / "data"
    dest.mkdir()
    ids = np.load(B.parents[1] / "configs/study03/catalog_ids.npy")
    np.save(out / "catalog_ids.npy", ids)
    selected = {
        s: dict(np.load(B.parents[1] / f"configs/study03/fixed_{s}_rows.npz")) for s in ["train", "val"]
    }
    records = {s: [None] * len(d["uid"]) for s, d in selected.items()}
    ev = (
        pl.scan_parquet(R / "data/raw/multi_event.parquet")
        .filter(
            (pl.col("event_type") == "listen")
            & pl.col("uid").is_in(cfg["dev_user_ids"])
            & (pl.col("timestamp") < cfg["val_end"])
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
    early = ev.filter(pl.col("timestamp") < cfg["rm_fit_end"])
    stats = (
        early.group_by("item_id")
        .agg(
            pl.len().alias("count"),
            pl.col("track_length_seconds").median().alias("duration"),
        )
        .sort(["count", "item_id"], descending=[True, False])
    )
    stats.write_parquet(dest / "item_stats.parquet")
    cat = stats.filter(pl.col("count") >= cfg["min_count"]).head(len(ids))
    assert np.array_equal(cat["item_id"].to_numpy(), ids)
    cat.write_csv(out / "catalog.csv")
    for f in ev.partition_by("uid", maintain_order=True):
        u = int(f["uid"][0])
        t = f["timestamp"].to_numpy().astype("int64")
        it = f["item_id"].to_numpy().astype("int64")
        y = f["played_ratio_pct"].to_numpy() / 100.0
        duration = f["track_length_seconds"].to_numpy()
        ts, first, ns = np.unique(t, return_index=True, return_counts=True)
        groups = [np.sort(it[i : i + n]) for i, n in zip(first, ns)]
        cache = {}
        for split, d in selected.items():
            for j in np.flatnonzero(d["uid"] == u):
                target = int(d["t"][j])
                ix = int(np.searchsorted(ts, target))
                assert ts[ix] == target
                if target not in cache:
                    raw, g, ht, omitted = pack(groups, ts, ix, cfg["context"])
                    cr, cg, _, _ = pack(groups, ts, ix, cfg["context"], True)
                    fb, fm = feedback(t, it, y, duration, target)
                    cache[target] = (raw, g, ht, omitted, cr, cg, fb, fm)
                raw, g, ht, omitted, cr, cg, fb, fm = cache[target]
                records[split][j] = dict(
                    uid=u,
                    t=target,
                    item=int(d["item"][j]),
                    y=float(d["y"][j]),
                    raw=raw,
                    g=g,
                    ht=ht,
                    fb=fb,
                    fbmax=fm,
                    repeat=float((raw == d["item"][j]).sum()) / (cfg["context"] - 1),
                    history_length=ix,
                    omitted=omitted,
                    collapsed_raw=cr,
                    collapsed_g=cg,
                )
    expected = json.loads((B.parents[1] / "configs/study03/fixed_input_manifest.json").read_text())
    checks = []
    for split, rs in records.items():
        z = {k: np.stack([r[k] for r in rs]) for k in rs[0]}
        z["w"] = weights(z["uid"])
        for k, v in z.items():
            actual = array_digest(v)
            checks.append(
                dict(
                    split=split,
                    field=k,
                    equal=actual == expected[split][k],
                    digest=actual,
                )
            )
        np.savez_compressed(dest / f"{split}_raw.npz", **z)
    dump(out / "reference_verification.json", checks)
    if not all(r["equal"] for r in checks):
        raise ValueError(
            "Reference arrays differ; inspect verification, never silently change control population"
        )
    print("Reference rebuilt exactly", len(checks), "arrays", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["rebuild"])
    p.add_argument("--out", type=Path, default=B / "runs/reference_rebuilt")
    a = p.parse_args()
    rebuild(json.loads((B.parents[1] / "configs/study03/study.json").read_text()), a.out)
