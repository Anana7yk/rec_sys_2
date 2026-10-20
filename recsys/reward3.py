"""Action-sensitive reward ablations with strictly past user-level offset."""

import json, pickle
from pathlib import Path
import numpy as np, pandas as pd, torch, polars as pl
from data2 import dump, map_data
from reward2 import metrics, metadata

KINDS = ["mean", "history", "candidate", "joint", "residual"]


def hfeatures(fb):
    return np.c_[fb, fb[:, 0] ** 2, fb[:, 0] * fb[:, 1]]


def cfeatures(d, asset, loo=False):
    ix = metadata(asset, d["item"])
    c = asset["candidate"][ix].copy()
    if loo:
        n = asset["count"][ix] - d["w"]
        s = asset["sum"][ix] - d["w"] * d["y"]
        mu = (asset["sum"].sum() - d["w"] * d["y"]) / (asset["count"].sum() - d["w"])
        c[:, 2] = (s + 100 * mu) / (n + 100) - mu
        b = asset["bin"][ix]
        c[:, 3] = (asset["bin_sum"][b] - d["w"] * d["y"] + 1000 * mu) / (
            asset["bin_count"][b] - d["w"] + 1000
        ) - mu
    return c


def design(d, asset, kind, loo=False):
    h = hfeatures(d["fb"])
    c = cfeatures(d, asset, loo)
    rep = d["repeat"]
    if kind == "mean":
        return np.ones((len(h), 1))
    if kind == "history":
        return np.c_[np.ones(len(h)), h]
    if kind == "candidate":
        return np.c_[np.ones(len(h)), c]
    return np.c_[
        np.ones(len(h)),
        h,
        c,
        (h[:, :, None] * c[:, None, :]).reshape(len(h), -1),
        rep,
        rep * d["fb"][:, 0],
    ]


def predict(d, asset, spec):
    x = design(d, asset, spec["kind"])
    z = x @ spec["coef"] + (d["fb"][:, 0] if spec["kind"] == "residual" else 0)
    return np.clip(spec.get("cal_a", 1) * np.clip(z, 0, 1) + spec.get("cal_b", 0), 0, 1)


def error_frame(d, p):
    rows = []
    for u in np.unique(d["uid"]):
        ix = np.flatnonzero(d["uid"] == u)
        y = d["y"][ix]
        z = p[ix]
        e = y - z
        rows.append(
            dict(
                uid=int(u),
                mse=float(np.mean(e * e)),
                within=float(np.mean((e - e.mean()) ** 2)),
                between=float(e.mean() ** 2),
                n=len(ix),
            )
        )
    return pd.DataFrame(rows)


def ci(x, seed=42, B=2000):
    rng = np.random.default_rng(seed)
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return dict(delta=float("nan"), low=float("nan"), high=float("nan"), users=0)
    m = np.array([a[rng.integers(len(a), size=len(a))].mean() for _ in range(B)])
    return dict(
        delta=float(a.mean()),
        low=float(np.quantile(m, 0.025)),
        high=float(np.quantile(m, 0.975)),
        users=len(a),
    )


def matched(d, p):
    rows = []
    for u in np.unique(d["uid"]):
        ix = np.flatnonzero(d["uid"] == u)
        ix = ix[np.argsort(d["t"][ix])]
        pairs = []
        for ii, a in enumerate(ix):
            for b in ix[ii + 1 :]:
                if d["t"][b] - d["t"][a] > 21600:
                    break
                if (
                    d["t"][a] == d["t"][b]
                    or d["item"][a] == d["item"][b]
                    or abs(d["fb"][a, 0] - d["fb"][b, 0]) > 0.05
                    or abs(d["y"][a] - d["y"][b]) < 0.2
                ):
                    continue
                delta = d["y"][b] - d["y"][a]
                pred = p[b] - p[a]
                pairs.append(
                    (float(np.sign(delta) * np.sign(pred)), float((delta - pred) ** 2))
                )
        if pairs:
            pairs = pairs[:20]
            rows.append(
                dict(
                    uid=int(u),
                    pairs=len(pairs),
                    concordance=float(np.mean([(q[0] + 1) / 2 for q in pairs])),
                    contrast_mse=float(np.mean([q[1] for q in pairs])),
                )
            )
    return pd.DataFrame(rows)


def fit(root):
    out = root / "reward"
    out.mkdir(exist_ok=False)
    dd = {
        s: dict(np.load(root / f"reward_data/{s}_raw.npz"))
        for s in ["rm_fit", "rm_tune", "rm_cal", "val"]
    }
    f = dd["rm_fit"]
    stats = pl.read_parquet(root / "data/item_stats.parquet")
    ids = stats["item_id"].to_numpy().astype("int64")
    dur = np.r_[stats["duration"].to_numpy(), stats["duration"].median()]
    pop = np.r_[stats["count"].to_numpy(), 0]
    mu = np.average(f["y"], weights=f["w"])
    cuts = np.unique(np.quantile(np.log1p(dur[:-1]), np.arange(0.1, 1, 0.1)))
    bins = np.digitize(np.log1p(dur), cuts)
    asset = dict(ids=ids, order=np.argsort(ids), mu=mu, duration=dur, pop=pop, bin=bins)
    ix = metadata(asset, f["item"])
    n = len(ids) + 1
    cnt = np.bincount(ix, weights=f["w"], minlength=n)
    su = np.bincount(ix, weights=f["w"] * f["y"], minlength=n)
    bc = np.bincount(bins[ix], weights=f["w"], minlength=len(cuts) + 1)
    bs = np.bincount(bins[ix], weights=f["w"] * f["y"], minlength=len(cuts) + 1)
    bm = (bs + 1000 * mu) / (bc + 1000)
    asset.update(count=cnt, sum=su, bin_count=bc, bin_sum=bs, correction=bm[bins] - mu)
    ld = np.log1p(dur)
    lp = np.log1p(pop)
    asset["candidate"] = np.c_[
        (ld - ld[:-1].mean()) / ld[:-1].std(),
        (lp - lp[:-1].mean()) / lp[:-1].std(),
        (su + 100 * mu) / (cnt + 100) - mu,
        bm[bins] - mu,
    ]
    specs = {}
    trials = []
    for kind in KINDS:
        X = design(f, asset, kind, loo=True)
        fm = np.average(X, axis=0, weights=f["w"])
        fs = np.sqrt(np.average((X - fm) ** 2, axis=0, weights=f["w"])).clip(0.01)
        fm[0] = 0
        fs[0] = 1
        Z = (X - fm) / fs
        y = f["y"] - (f["fb"][:, 0] if kind == "residual" else 0)
        w = f["w"]
        A = (Z.T * w) @ Z / w.sum()
        b = Z.T @ (w * y) / w.sum()
        pen = np.eye(Z.shape[1])
        pen[0, 0] = 0
        choices = []
        for ridge in [0.001, 0.01, 0.1]:
            coef = np.linalg.solve(A + ridge * pen, b) / fs
            coef[0] -= fm @ coef
            spec = dict(kind=kind, coef=coef, ridge=ridge)
            p = predict(dd["rm_tune"], asset, spec)
            met = metrics(dd["rm_tune"]["y"], p, dd["rm_tune"]["w"])
            trials.append(dict(kind=kind, ridge=ridge, **met))
            choices.append((met["rmse"], spec))
        spec = min(choices, key=lambda x: x[0])[1]
        cal = dd["rm_cal"]
        p = predict(cal, asset, spec)
        Z = np.c_[p, np.ones(len(p))]
        w = cal["w"]
        ab = np.linalg.solve(
            (Z.T * w) @ Z / w.sum() + 0.01 * np.eye(2),
            Z.T @ (w * cal["y"]) / w.sum() + 0.01 * np.array([1, 0]),
        )
        spec.update(
            cal_a=float(np.clip(ab[0], 0, 2)), cal_b=float(np.clip(ab[1], -0.5, 0.5))
        )
        specs[kind] = spec
    rows = []
    err = {}
    pairs = []
    for split, d in dd.items():
        if split == "rm_fit":
            continue
        for kind, spec in specs.items():
            p = predict(d, asset, spec)
            ef = error_frame(d, p)
            ef.to_csv(out / f"{split}_{kind}_errors.csv", index=False)
            pair = matched(d, p)
            pair.to_csv(out / f"{split}_{kind}_pairs.csv", index=False)
            rows.append(
                dict(
                    split=split,
                    kind=kind,
                    **metrics(d["y"], p, d["w"]),
                    within_mse=float(ef.within.mean()),
                    between_mse=float(ef.between.mean()),
                    matched_users=len(pair),
                    matched_pairs=int(pair.pairs.sum()) if len(pair) else 0,
                    concordance=float(pair.concordance.mean()) if len(pair) else None,
                )
            )
            if split == "val":
                err[kind] = ef
    history = next(r for r in rows if r["split"] == "val" and r["kind"] == "history")
    valid = [
        r
        for r in rows
        if r["split"] == "val"
        and r["kind"] in ["joint", "residual"]
        and r["rmse"] <= history["rmse"] * 1.005
    ]
    choice = min(
        valid
        or [
            r
            for r in rows
            if r["split"] == "val" and r["kind"] in ["joint", "residual"]
        ],
        key=lambda r: r["within_mse"],
    )
    kind = choice["kind"]
    e = err[kind].merge(err["history"], on="uid", suffixes=("_action", "_history"))
    inc = ci(e.within_action - e.within_history)
    gate = bool(valid and inc["high"] < 0)
    np.savez_compressed(out / "asset.npz", **asset)
    with (out / "specs.pkl").open("wb") as file:
        pickle.dump(specs, file)
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(trials).to_csv(out / "inner_grid.csv", index=False)
    dump(
        out / "selection.json",
        dict(
            primary=kind,
            gate=gate,
            incremental_within_mse=inc,
            choice=choice,
            history=history,
            criterion="lowest within-user MSE among joint/residual with total RMSE<=1.005 history; failing gate is exploratory proxy",
        ),
    )
    # Independent evaluator fixed architecture, disjoint half fitting rows; used only final diagnostics.
    from catboost import CatBoostRegressor

    X = design(f, asset, "joint", loo=True)[:, 1:]
    take = np.random.default_rng(9011).permutation(len(X))[::2]
    cb = CatBoostRegressor(
        iterations=300,
        depth=5,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=9011,
        thread_count=6,
        verbose=False,
        l2_leaf_reg=10,
    )
    cb.fit(X[take], f["y"][take], sample_weight=f["w"][take])
    cb.save_model(str(out / "evaluator.cbm"))
    cal = dd["rm_cal"]
    cp = np.clip(cb.predict(design(cal, asset, "joint")[:, 1:]), 0, 1)
    Z = np.c_[cp, np.ones(len(cp))]
    w = cal["w"]
    ab = np.linalg.solve(
        (Z.T * w) @ Z / w.sum() + 0.01 * np.eye(2),
        Z.T @ (w * cal["y"]) / w.sum() + 0.01 * np.array([1, 0]),
    )
    ecal = dict(a=float(np.clip(ab[0], 0, 2)), b=float(np.clip(ab[1], -0.5, 0.5)))
    dump(out / "evaluator_cal.json", ecal)
    for split in ["rm_tune", "rm_cal", "val"]:
        d = dd[split]
        p = np.clip(
            ecal["a"] * np.clip(cb.predict(design(d, asset, "joint")[:, 1:]), 0, 1)
            + ecal["b"],
            0,
            1,
        )
        e = error_frame(d, p)
        rows.append(
            dict(
                split=split,
                kind="evaluator",
                **metrics(d["y"], p, d["w"]),
                within_mse=e.within.mean(),
                between_mse=e.between.mean(),
            )
        )
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    print("RM", json.loads((out / "selection.json").read_text()), flush=True)


class Reward:
    def __init__(self, asset, spec, ids, device):
        self.asset = asset
        self.spec = spec
        self.device = device
        self.ids = ids
        ix = metadata(asset, ids)
        self.c = torch.tensor(
            asset["candidate"][ix], device=device, dtype=torch.float32
        )
        self.correction = torch.tensor(
            asset["correction"][ix], device=device, dtype=torch.float32
        )
        self.coef = torch.tensor(spec["coef"], device=device, dtype=torch.float32)

    def all(self, x, fb):
        h = torch.cat([fb, fb[:, 0:1] ** 2, fb[:, 0:1] * fb[:, 1:2]], 1)
        b = len(x)
        k = self.spec["kind"]
        z = self.coef[0].expand(b, len(self.ids)).clone()
        if k == "history":
            z += (h @ self.coef[1:])[:, None]
        elif k == "candidate":
            z += (self.c @ self.coef[1:])[None, :]
        elif k in ["joint", "residual"]:
            z += (h @ self.coef[1:7])[:, None]
            z += (self.c @ self.coef[7:11])[None, :]
            z += (h @ self.coef[11:35].reshape(6, 4)) @ self.c.T
            count = torch.zeros_like(z)
            count.scatter_add_(1, (x - 3).clamp_min(0), (x >= 3).float())
            rep = count / (x.shape[1] - 1)
            z += rep * (self.coef[35] + self.coef[36] * fb[:, 0:1])
            if k == "residual":
                z += fb[:, 0:1]
        return (
            self.spec.get("cal_a", 1) * z.clamp(0, 1) + self.spec.get("cal_b", 0)
        ).clamp(0, 1)

    def actions(self, x, fb, a):
        h = torch.cat([fb, fb[:, 0:1] ** 2, fb[:, 0:1] * fb[:, 1:2]], 1)
        c = self.c[a]
        k = self.spec["kind"]
        z = self.coef[0].expand(len(x)).clone()
        if k == "history":
            z += h @ self.coef[1:]
        elif k == "candidate":
            z += c @ self.coef[1:]
        elif k in ["joint", "residual"]:
            z += (
                h @ self.coef[1:7]
                + c @ self.coef[7:11]
                + ((h @ self.coef[11:35].reshape(6, 4)) * c).sum(1)
            )
            rep = (x == a[:, None] + 3).float().sum(1) / (x.shape[1] - 1)
            z += rep * (self.coef[35] + self.coef[36] * fb[:, 0])
            if k == "residual":
                z += fb[:, 0]
        return (
            self.spec.get("cal_a", 1) * z.clamp(0, 1) + self.spec.get("cal_b", 0)
        ).clamp(0, 1)


class Evaluator:
    def __init__(self, root, asset, ids):
        from catboost import CatBoostRegressor

        self.model = CatBoostRegressor()
        self.model.load_model(str(root / "reward/evaluator.cbm"))
        self.asset = asset
        self.ids = ids
        self.cal = json.loads((root / "reward/evaluator_cal.json").read_text())

    def raw(self, d):
        return np.clip(
            self.cal["a"]
            * np.clip(self.model.predict(design(d, self.asset, "joint")[:, 1:]), 0, 1)
            + self.cal["b"],
            0,
            1,
        )

    def candidates(self, d, a):
        b, k = a.shape
        raw = self.ids[a.reshape(-1)]
        fb = np.repeat(d["fb"], k, axis=0)
        xx = np.repeat(d["x"], k, axis=0)
        tok = np.repeat(a.reshape(-1) + 3, 1)
        repeat = (xx == tok[:, None]).sum(1) / (xx.shape[1] - 1)
        dd = dict(item=raw, fb=fb, repeat=repeat)
        return self.raw(dd).reshape(b, k)
