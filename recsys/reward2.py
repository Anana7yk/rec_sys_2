"""Weighted empirical-Bayes item baselines and small ridge residual models."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import torch
from data2 import dump


def wm(x, w):
    return np.average(x, weights=w)


def prediction_bins(p, w, n=10):
    order = np.argsort(p)
    cdf = np.cumsum(w[order]) / w.sum()
    cuts = np.unique(
        p[order][np.searchsorted(cdf, np.arange(1 / n, 1, 1 / n)).clip(0, len(p) - 1)]
    )
    return np.digitize(p, cuts)


def metrics(y, p, w):
    if len(y) == 0:
        return {
            k: float("nan")
            for k in ["rmse", "mae", "calibration", "prediction", "observed"]
        }
    mse = wm((y - p) ** 2, w)
    bins = prediction_bins(p, w)
    groups = [np.flatnonzero(bins == b) for b in np.unique(bins)]
    gap = (
        sum(w[ix].sum() * abs(wm(p[ix], w[ix]) - wm(y[ix], w[ix])) for ix in groups)
        / w.sum()
    )
    return dict(
        rmse=float(np.sqrt(mse)),
        mae=float(wm(abs(y - p), w)),
        calibration=float(gap),
        prediction=float(wm(p, w)),
        observed=float(wm(y, w)),
    )


def metadata(stats, raw):
    ids = stats["ids"]
    order = stats["order"]
    sortedids = ids[order]
    pos = np.searchsorted(sortedids, raw)
    valid = pos < len(ids)
    pos = pos.clip(0, len(ids) - 1)
    valid &= sortedids[pos] == raw
    index = np.where(valid, order[pos], len(ids))
    return index


def features(d, asset):
    ix = metadata(asset, d["item"])
    dur = asset["duration"][ix]
    pop = asset["pop"][ix]
    ld = (np.log1p(dur) - asset["dur_mu"]) / asset["dur_sd"]
    lp = (np.log1p(pop) - asset["pop_mu"]) / asset["pop_sd"]
    fb = d["fb"]
    center = fb[:, 0] - asset["mu"]
    return (
        np.stack(
            [
                center,
                fb[:, 1] - asset["skip"],
                fb[:, 2],
                fb[:, 3],
                d["repeat"],
                ld,
                lp,
                center * ld,
            ],
            1,
        ),
        ix,
    )


def raw_predict(d, asset, spec, loo=False):
    f, ix = features(d, asset)
    if spec["kind"] == "mean":
        return np.full(len(ix), asset["mu"])
    prior = asset["bin_prior"][asset["bin"][ix]]
    count = asset["count"][ix].copy()
    sums = asset["sum"][ix].copy()
    if loo:
        count -= d["w"]
        sums -= d["w"] * d["y"]
        mu_loo = (asset["sum"].sum() - d["w"] * d["y"]) / (
            asset["count"].sum() - d["w"]
        )
        b = asset["bin"][ix]
        prior = (asset["bin_sum"][b] - d["w"] * d["y"] + 1000 * mu_loo) / (
            asset["bin_count"][b] - d["w"] + 1000
        )
    p = (sums + spec["alpha"] * prior) / (count + spec["alpha"])
    if spec["kind"] == "personal":
        p += spec["intercept"] + ((f - spec["f_mu"]) / spec["f_sd"]) @ spec["coef"]
    return np.clip(p, 0, 1)


def predict(d, asset, spec):
    p = raw_predict(d, asset, spec)
    return np.clip(spec.get("cal_a", 1) * p + spec.get("cal_b", 0), 0, 1)


def fit_rm(data, stats_path, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    stats = pl.read_parquet(stats_path)
    ids = stats["item_id"].to_numpy().astype("int64")
    dur = stats["duration"].to_numpy().astype(float)
    pop = stats["count"].to_numpy().astype(float)
    fit = data["rm_fit"]
    tune = data["rm_tune"]
    cal = data["rm_cal"]
    val = data["val"]
    mu = wm(fit["y"], fit["w"])
    cuts = np.unique(np.quantile(np.log1p(dur), np.linspace(0, 1, 11)[1:-1]))
    dur = np.r_[dur, np.median(dur)]
    pop = np.r_[pop, 0.0]
    bins = np.digitize(np.log1p(dur), cuts)
    n = len(ids)
    asset = dict(
        ids=ids,
        order=np.argsort(ids),
        duration=dur,
        pop=pop,
        bin=bins,
        cuts=cuts,
        mu=mu,
        skip=wm(fit["fb"][:, 1], fit["w"]),
        dur_mu=np.log1p(dur[:-1]).mean(),
        dur_sd=np.log1p(dur[:-1]).std().clip(0.01),
        pop_mu=np.log1p(pop[:-1]).mean(),
        pop_sd=np.log1p(pop[:-1]).std().clip(0.01),
    )
    ix = metadata(asset, fit["item"])
    asset["count"] = np.bincount(ix, weights=fit["w"], minlength=n + 1)
    asset["sum"] = np.bincount(ix, weights=fit["w"] * fit["y"], minlength=n + 1)
    bc = np.bincount(bins[ix], weights=fit["w"], minlength=len(cuts) + 1)
    bs = np.bincount(bins[ix], weights=fit["w"] * fit["y"], minlength=len(cuts) + 1)
    asset["bin_sum"] = bs
    asset["bin_count"] = bc
    asset["bin_prior"] = (bs + 1000 * mu) / (bc + 1000)
    asset["duration_correction"] = asset["bin_prior"][bins] - mu
    trials = []
    specs = [dict(kind="mean")]
    ff, _ = features(fit, asset)
    fm = np.average(ff, axis=0, weights=fit["w"])
    fs = np.sqrt(np.average((ff - fm) ** 2, axis=0, weights=fit["w"])).clip(0.01)
    X = np.c_[np.ones(len(ff)), (ff - fm) / fs]
    w = fit["w"]
    pen = np.eye(X.shape[1])
    pen[0, 0] = 0
    for alpha in [10, 100, 1000]:
        item = dict(kind="item", alpha=alpha)
        specs.append(item)
        residual = fit["y"] - raw_predict(fit, asset, item, loo=True)
        for ridge in [0.01, 0.1]:
            coef = np.linalg.solve(
                (X.T * w) @ X / w.sum() + ridge * pen, (X.T @ (w * residual)) / w.sum()
            )
            specs.append(
                dict(
                    kind="personal",
                    alpha=alpha,
                    ridge=ridge,
                    intercept=float(coef[0]),
                    coef=coef[1:],
                    f_mu=fm,
                    f_sd=fs,
                )
            )
    for i, spec in enumerate(specs):
        met = metrics(tune["y"], raw_predict(tune, asset, spec), tune["w"])
        trials.append(
            dict(
                id=i,
                kind=spec["kind"],
                alpha=spec.get("alpha"),
                ridge=spec.get("ridge"),
                **met,
            )
        )
    chosen = {}
    for kind in ["mean", "item", "personal"]:
        rr = [r for r in trials if r["kind"] == kind]
        best = min(r["rmse"] for r in rr)
        r = min(
            [r for r in rr if r["rmse"] <= best + 1e-4], key=lambda r: r["calibration"]
        )
        spec = specs[r["id"]]
        # Fit a fixed ridge-to-identity affine calibration on a separate later period.
        p = raw_predict(cal, asset, spec)
        xx = np.c_[p, np.ones(len(p))]
        ww = cal["w"]
        regular = 0.01 * np.eye(2)
        ab = np.linalg.solve(
            (xx.T * ww) @ xx / ww.sum() + regular,
            (xx.T @ (ww * cal["y"])) / ww.sum() + regular @ np.array([1.0, 0.0]),
        )
        spec = spec | dict(
            cal_a=float(np.clip(ab[0], 0, 2)), cal_b=float(np.clip(ab[1], -0.5, 0.5))
        )
        chosen[kind] = spec
    rows = []
    for split in ["rm_tune", "rm_cal", "val"]:
        d = data[split]
        for kind, spec in chosen.items():
            rows.append(
                dict(
                    split=split,
                    kind=kind,
                    **metrics(d["y"], predict(d, asset, spec), d["w"]),
                )
            )
    validation = [r for r in rows if r["split"] == "val"]
    best = min(r["rmse"] for r in validation)
    bestrow = min(
        [r for r in validation if r["rmse"] <= best + 1e-4],
        key=lambda r: (r["calibration"], ["mean", "item", "personal"].index(r["kind"])),
    )
    primary = bestrow["kind"]
    mean = next(r for r in validation if r["kind"] == "mean")["rmse"]
    gate = bestrow["rmse"] < mean * 0.995 and primary != "mean"
    np.savez_compressed(out / "asset.npz", **asset)
    import pickle

    with (out / "specs.pkl").open("wb") as f:
        pickle.dump(chosen, f)
    pd.DataFrame(trials).to_csv(out / "inner_grid.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    dump(
        out / "selection.json",
        dict(
            primary=primary,
            gate_pass=bool(gate),
            validation=validation,
            criterion="weighted validation RMSE, calibration then simplicity within1e-4; hyperparameters inner18-19M; calibration19-20M",
        ),
    )
    return asset, chosen, primary, gate


class FastReward:
    def __init__(self, asset, spec, catalog_ids, device):
        self.asset = {
            k: (v.item() if isinstance(v, np.ndarray) and v.ndim == 0 else v)
            for k, v in asset.items()
        }
        self.spec = spec
        self.device = device
        ix = metadata(asset, catalog_ids)
        prior = asset["bin_prior"][asset["bin"][ix]]
        base = (
            np.full(len(ix), asset["mu"])
            if spec["kind"] == "mean"
            else (asset["sum"][ix] + spec["alpha"] * prior)
            / (asset["count"][ix] + spec["alpha"])
        )
        self.base = torch.tensor(base, dtype=torch.float32, device=device)
        self.dur = torch.tensor(
            (np.log1p(asset["duration"][ix]) - asset["dur_mu"]) / asset["dur_sd"],
            dtype=torch.float32,
            device=device,
        )
        self.pop = torch.tensor(
            (np.log1p(asset["pop"][ix]) - asset["pop_mu"]) / asset["pop_sd"],
            dtype=torch.float32,
            device=device,
        )
        self.correction = torch.tensor(
            asset["duration_correction"][ix], dtype=torch.float32, device=device
        )
        self.coef = torch.tensor(
            spec.get("coef", np.zeros(8)) / spec.get("f_sd", np.ones(8)),
            dtype=torch.float32,
            device=device,
        )
        self.intercept = spec.get("intercept", 0) - float(
            np.dot(
                spec.get("f_mu", np.zeros(8)),
                spec.get("coef", np.zeros(8)) / spec.get("f_sd", np.ones(8)),
            )
        )

    def user(self, fb):
        f = torch.stack(
            [
                fb[:, 0] - self.asset["mu"],
                fb[:, 1] - self.asset["skip"],
                fb[:, 2],
                fb[:, 3],
            ],
            1,
        )
        return f @ self.coef[:4] + self.intercept

    def all(self, x, fb):
        p = self.base[None, :].expand(len(x), -1)
        if self.spec["kind"] == "personal":
            repeat = torch.zeros((len(x), len(self.base)), device=x.device)
            repeat.scatter_add_(1, (x - 3).clamp_min(0), (x >= 3).float())
            repeat /= x.shape[1] - 1
            p = (
                p
                + self.user(fb)[:, None]
                + self.coef[4] * repeat
                + self.coef[5] * self.dur[None, :]
                + self.coef[6] * self.pop[None, :]
                + self.coef[7]
                * (fb[:, 0] - self.asset["mu"])[:, None]
                * self.dur[None, :]
            )
        return (self.spec["cal_a"] * p.clamp(0, 1) + self.spec["cal_b"]).clamp(0, 1)

    def actions(self, x, fb, a):
        p = self.base[a]
        if self.spec["kind"] == "personal":
            repeat = (x == a[:, None] + 3).float().sum(1) / (x.shape[1] - 1)
            p = (
                p
                + self.user(fb)
                + self.coef[4] * repeat
                + self.coef[5] * self.dur[a]
                + self.coef[6] * self.pop[a]
                + self.coef[7] * (fb[:, 0] - self.asset["mu"]) * self.dur[a]
            )
        return (self.spec["cal_a"] * p.clamp(0, 1) + self.spec["cal_b"]).clamp(0, 1)
