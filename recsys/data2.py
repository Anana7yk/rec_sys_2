"""Strict group histories and user-balanced random event populations."""

import json, time
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/study03"


def dump(p, x):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(
        json.dumps(
            x,
            ensure_ascii=False,
            indent=2,
            default=lambda a: a.item() if isinstance(a, np.generic) else str(a),
        )
        + "\n"
    )


def weights(uid):
    _, ix, n = np.unique(uid, return_inverse=True, return_counts=True)
    w = 1 / n[ix]
    return (w / w.mean()).astype("float32")


def sample_rows(t, start, end, k, rng):
    ix = np.flatnonzero((t >= start) & (t < end))
    return np.sort(rng.choice(ix, min(k, len(ix)), replace=False)) if len(ix) else ix


def pack(groups, times, index, context=64, collapsed=False):
    # PAD=-1, query=-2, unrepresented timestamp group=-3. No chronology inside group.
    selected = []
    slots = context - 1
    omitted = 0
    for j in range(index - 1, -1, -1):
        group = groups[j]
        if collapsed and len(group) > 1:
            group = np.array([-3], dtype="int64")
        if len(group) > slots:
            if slots:
                selected.append((np.array([-3]), times[j]))
                omitted += len(groups[j])
            break
        selected.append((group, times[j]))
        slots -= len(group)
        if slots == 0:
            break
    selected.reverse()
    raw = np.full(context, -1, dtype="int64")
    g = np.zeros(context, dtype="int64")
    ht = np.full(context, -1, dtype="int64")
    p = 0
    for j, (members, ts) in enumerate(selected, 1):
        raw[p : p + len(members)] = members
        g[p : p + len(members)] = j
        ht[p : p + len(members)] = ts
        p += len(members)
    raw[p] = -2
    g[p] = len(selected) + 1
    return raw, g, ht, omitted


def feedback(t, it, y, duration, target_time):
    end = np.searchsorted(t, target_time - 3600, side="left")
    start = np.searchsorted(t, t[max(0, end - 512)], side="left") if end else 0
    ix = np.arange(start, end)
    availability = np.maximum(
        t[ix] + 3600, t[ix] + duration[ix] * np.maximum(1, y[ix]) + 300
    )
    ix = ix[availability < target_time]
    cut = max(0, len(ix) - 200)
    if cut and t[ix[cut]] == t[ix[cut - 1]]:
        cut = np.searchsorted(t[ix], t[ix[cut]], side="right")
    ix = ix[cut:]
    if not len(ix):
        return np.array([0.5, 0.5, 0.5, 0], dtype="float32"), -1
    yy = np.minimum(y[ix], 1)
    v = np.array(
        [yy.mean(), (yy == 0).mean(), yy.std(), np.log1p(len(ix)) / np.log(201)],
        dtype="float32",
    )
    return v, float(
        np.maximum(
            t[ix] + 3600, t[ix] + duration[ix] * np.maximum(1, y[ix]) + 300
        ).max()
    )


def prepare_raw(cfg, out, holdout=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(cfg["seed_data"] + (777 if holdout else 0))
    start_time = time.monotonic()
    userkey = "holdout_user_ids" if holdout else "dev_user_ids"
    users = np.asarray(cfg[userkey], dtype="int64")
    lf = pl.scan_parquet(ROOT / "data/raw/multi_event.parquet")
    ev = (
        lf.filter(
            (pl.col("event_type") == "listen") & pl.col("uid").is_in(users.tolist())
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
    if not holdout:
        early = ev.filter(pl.col("timestamp") < cfg["rm_fit_end"])
        stats = (
            early.group_by("item_id")
            .agg(
                pl.len().alias("count"),
                pl.col("track_length_seconds").median().alias("duration"),
            )
            .sort(["count", "item_id"], descending=[True, False])
        )
        stats.write_parquet(out / "item_stats.parquet")
    splits = (
        {"holdout": (cfg["val_end"], cfg["test_end"], cfg["holdout_per_user"])}
        if holdout
        else {
            "train": (0, cfg["train_end"], cfg["train_per_user"]),
            "rm_fit": (0, cfg["rm_fit_end"], cfg["rm_per_user"]),
            "rm_eval": (0, cfg["rm_fit_end"], cfg["rm_per_user"]),
            "rm_tune": (cfg["rm_fit_end"], cfg["rm_tune_end"], cfg["inner_per_user"]),
            "rm_cal": (cfg["rm_tune_end"], cfg["rm_cal_end"], cfg["inner_per_user"]),
            "val": (cfg["train_end"], cfg["val_end"], cfg["val_per_user"]),
        }
    )
    records = {k: [] for k in splits}
    sessions = []
    ties = 0
    for frame in ev.partition_by("uid", maintain_order=True):
        uid = int(frame["uid"][0])
        t = frame["timestamp"].to_numpy().astype("int64")
        items = frame["item_id"].to_numpy().astype("int64")
        y = frame["played_ratio_pct"].to_numpy().astype("float32") / 100
        duration = frame["track_length_seconds"].to_numpy().astype("float64")
        ts, first, count = np.unique(t, return_index=True, return_counts=True)
        groups = [np.sort(items[i : i + n]) for i, n in zip(first, count)]
        gi = np.searchsorted(ts, t)
        ties += int(count[count > 1].sum())
        for split, (low, high, k) in splits.items():
            eligible = np.flatnonzero(
                (t >= low) & (t < high) & (gi >= cfg["min_history"])
            )
            split_rng = (
                np.random.default_rng(cfg["seed_data"] + 9000 + uid)
                if split == "rm_eval"
                else rng
            )
            chosen = (
                np.sort(
                    split_rng.choice(eligible, min(k, len(eligible)), replace=False)
                )
                if len(eligible)
                else []
            )
            for row in chosen:
                ix = gi[row]
                raw, g, ht, omitted = pack(groups, ts, ix, cfg["context"])
                collapse, cg, _, _ = pack(groups, ts, ix, cfg["context"], True)
                fb, fbmax = feedback(t, items, y, duration, t[row])
                repeat = float((raw == items[row]).sum()) / (cfg["context"] - 1)
                records[split].append(
                    dict(
                        uid=uid,
                        t=int(t[row]),
                        item=int(items[row]),
                        y=float(min(y[row], 1)),
                        raw=raw,
                        g=g,
                        ht=ht,
                        fb=fb,
                        fbmax=fbmax,
                        repeat=repeat,
                        history_length=int(ix),
                        omitted=omitted,
                        collapsed_raw=collapse,
                        collapsed_g=cg,
                    )
                )
        if not holdout:
            # Session end is identifiable only after 30 minutes of observed absence.
            nextts = np.r_[ts[1:], np.inf]
            ends = np.flatnonzero(nextts - ts > 1800)
            start_indices = np.r_[0, np.flatnonzero(np.diff(ts) > 1800) + 1]
            for ix in ends:
                anchor = ts[ix] + 1800
                split = (
                    "fit"
                    if anchor < cfg["rm_fit_end"]
                    else (
                        "tune"
                        if anchor < cfg["rm_cal_end"]
                        else (
                            "val"
                            if cfg["train_end"] <= anchor < cfg["val_end"]
                            else None
                        )
                    )
                )
                if split is None:
                    continue
                boundary = {
                    "fit": cfg["rm_fit_end"],
                    "tune": cfg["rm_cal_end"],
                    "val": cfg["val_end"],
                }[split]
                if anchor + 604800 >= boundary:
                    continue
                st = start_indices[np.searchsorted(start_indices, ix, side="right") - 1]
                left = np.searchsorted(t, anchor - 7 * 86400)
                old = t[left : np.searchsorted(t, anchor)]
                fb, _ = feedback(t, items, y, duration, anchor)
                features = [
                    np.log1p(len(old)),
                    len(np.unique(old // 86400)),
                    np.log1p(count[st : ix + 1].sum()),
                    np.log1p(ts[ix] - ts[st]),
                    np.log1p(ix + 1),
                    float(fb[0]),
                    float(fb[1]),
                ]
                sessions.append(
                    dict(
                        uid=uid,
                        t=int(anchor),
                        split=split,
                        y=int(nextts[ix] <= anchor + 604800),
                        features=features,
                    )
                )

    def save(name, recs):
        fields = {k: np.stack([x[k] for x in recs]) for k in recs[0]}
        fields["w"] = weights(fields["uid"])
        np.savez_compressed(out / f"{name}_raw.npz", **fields)
        assert np.all(fields["ht"] < fields["t"][:, None])
        assert np.all(fields["fbmax"] < fields["t"])
        return dict(
            rows=len(recs),
            users=len(np.unique(fields["uid"])),
            mean_response=float(np.average(fields["y"], weights=fields["w"])),
            feedback_available=float((fields["fbmax"] >= 0).mean()),
            omitted_group_members=int(fields["omitted"].sum()),
        )

    audit = dict(
        users=len(users),
        listens=ev.height,
        items=ev["item_id"].n_unique(),
        ties_rows=ties,
        splits={},
    )
    for name, r in records.items():
        audit["splits"][name] = save(name, r)
    if sessions:
        for split in ["fit", "tune", "val"]:
            rr = [r for r in sessions if r["split"] == split]
            chosen = []
            for uid in users:
                rows = [r for r in rr if r["uid"] == uid]
                n = min(len(rows), 128 if split == "fit" else 32)
                if n:
                    chosen.extend(
                        rows[j] for j in rng.choice(len(rows), n, replace=False)
                    )
            np.savez_compressed(
                out / f"sessions_{split}.npz",
                uid=np.array([r["uid"] for r in chosen]),
                t=np.array([r["t"] for r in chosen]),
                y=np.array([r["y"] for r in chosen]),
                x=np.array([r["features"] for r in chosen]),
                w=weights(np.array([r["uid"] for r in chosen])),
            )
    audit["seconds"] = time.monotonic() - start_time
    dump(out / "audit.json", audit)
    print(audit, flush=True)


def lookup(ids, raw):
    # Vectorized sorted search avoids allocating a max-ID-dependent table.
    order = np.argsort(ids)
    sorted_ids = ids[order]
    pos = np.searchsorted(sorted_ids, np.maximum(raw, 0))
    valid = (raw >= 0) & (pos < len(ids))
    pos = pos.clip(0, len(ids) - 1)
    valid &= sorted_ids[pos] == raw
    out = np.full(raw.shape, 2, dtype="int64")
    out[raw == -1] = 0
    out[raw == -2] = 1
    out[valid] = order[pos[valid]] + 3
    return out


def map_data(d, ids, collapse=False):
    z = dict(d)
    z["x"] = lookup(ids, d["collapsed_raw"] if collapse else d["raw"])
    z["g"] = d["collapsed_g"] if collapse else d["g"]
    z["target"] = lookup(ids, d["item"])
    return z
