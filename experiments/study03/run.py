"""Independent study03 stages; final data opened only after seal."""

import sys, json, pickle, hashlib, time, argparse, copy
from pathlib import Path
import numpy as np, pandas as pd, torch, polars as pl

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B.parents[1] / "recsys"))
sys.path.insert(0, str(B))
from data2 import dump, map_data, prepare_raw
from models2 import load, save, freeze
from reward3 import Reward, Evaluator, predict, metrics, error_frame, matched, ci
from metrics2 import evaluate, key, latency
from fit2 import rl
from train import fit
from provenance import create_seal, verify_seal


def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def assets(root, device):
    cfg = json.loads((root / "config.json").read_text())
    ids = np.load(root / "catalog_ids.npy")
    a = dict(np.load(root / "reward/asset.npz"))
    with (root / "reward/specs.pkl").open("rb") as f:
        specs = pickle.load(f)
    sel = json.loads((root / "reward/selection.json").read_text())
    rm = Reward(a, specs[sel["primary"]], ids, device)
    rm.beta = cfg["beta"]
    cat = pd.read_csv(root / "catalog.csv")
    items = dict(
        ids=ids,
        pop=cat["count"].to_numpy(),
        duration=cat["duration"].to_numpy(),
        positive_train=np.load(root / "pool/positive_counts.npy"),
    )
    d = {
        s: map_data(dict(np.load(root / f"data/{s}_raw.npz")), ids)
        for s in ["train", "val"]
    }
    return cfg, ids, a, specs, rm, items, d


def seal(root):
    fs = [
        root / "config.json",
        root / "catalog_ids.npy",
        root / "catalog.csv",
        B.parents[1] / "docs/PROTOCOL_STUDY03.md",
        *sorted((B.parents[1] / "recsys").glob("*.py")),
        B / "run.py",
        B / "train.py",
    ]
    fs += [
        f
        for f in (root / "reward").iterdir()
        if f.suffix in [".npz", ".pkl", ".cbm", ".json"]
    ]
    for seed in json.loads((root / "config.json").read_text())["seeds"]:
        p = root / f"seed_{seed}"
        fs += (
            list(p.glob("*.pt"))
            + list(p.glob("*selection.json"))
            + [p / "pure/best.pt"]
        )
    fs += list((root / "data").glob("*.npz")) + [root / "pool/positive_counts.npy"]
    dump(
        root / "POLICY_SEAL.json",
        dict(
            final_unopened=True,
            **create_seal(fs, root),
        ),
    )


def policies(root, cfg, rm, items, d, device):
    for seed in cfg["seeds"]:
        out = root / f"seed_{seed}"
        pure = freeze(load(out / "pure/best.pt", device))
        basepure, _, _ = evaluate(
            "pure", pure, pure, d["val"], rm, items, cfg, device, quick=True
        )
        mix = []
        for alpha in [0, 0.25, 0.5, 0.75, 0.9]:
            pure.alpha = alpha
            pure.spec["alpha"] = alpha
            s, _, _ = evaluate(
                "hybrid", pure, pure, d["val"], rm, items, cfg, device, quick=True
            )
            mix.append(dict(alpha=alpha, **s))
        choice = max(mix, key=lambda r: r["ndcg20"])
        pure.alpha = choice["alpha"]
        pure.spec["alpha"] = choice["alpha"]
        save(out / "base.pt", pure)
        dump(
            out / "hybrid_selection.json",
            dict(
                choice=choice,
                grid=mix,
                criterion="validation NDCG20; same fixed gate for RL and reference",
            ),
        )
        ref = freeze(load(out / "base.pt", device))
        base, _, _ = evaluate(
            "base", ref, ref, d["val"], rm, items, cfg, device, quick=True
        )
        rows = [dict(method="pure", **basepure), dict(method="base", **base)]
        for method in ["popularity", "context_count", "ablate_pop", "ablate_uniform"]:
            s, _, _ = evaluate(
                method,
                ref,
                ref,
                d["val"],
                rm,
                items,
                cfg,
                device,
                lam=ref.alpha,
                quick=True,
            )
            rows.append(dict(method=method, lam=ref.alpha, **s))
            if method.startswith("ablate"):
                grid = []
                for alpha in [0, 0.25, 0.5, 0.75, 0.9, 1]:
                    s, _, _ = evaluate(
                        method,
                        ref,
                        ref,
                        d["val"],
                        rm,
                        items,
                        cfg,
                        device,
                        lam=alpha,
                        quick=True,
                    )
                    grid.append(dict(alpha=alpha, **s))
                choice = max(grid, key=lambda s: s["ndcg20"])
                dump(out / f"{method}_selection.json", dict(choice=choice, grid=grid))
        times = []
        for H, mod, steps in [
            (1, False, cfg["rl_steps"]),
            (5, False, cfg["rl_steps"] // 5),
            (1, True, cfg["rl_steps"]),
        ]:
            name = f"rl_h{H}_{'duration' if mod else 'raw'}"
            m, compute = rl(
                cfg, d, items, rm, ref, out / f"{name}.pt", device, seed, H, mod, steps
            )
            times.append(compute)
            s, _, _ = evaluate(
                name, m, ref, d["val"], rm, items, cfg, device, quick=True
            )
            rows.append(dict(method=name, **s))
        m = fit(
            root,
            f"seed_{seed}/sft",
            seed + 3000,
            ref.spec["d"],
            0.0001,
            0.01,
            24000,
            False,
            device,
            initial=ref,
            seconds=max(times),
        )
        # Match RL's reward-with-relevance-guard selector as well as its compute budget.
        # Preserve the conventional best-NDCG checkpoint in sft/best.pt.
        candidates = []
        for label, candidate in [
            ("initial", ref),
            ("best_ndcg", m),
            ("last", load(out / "sft/last.pt", device)),
        ]:
            summary, _, _ = evaluate(
                "sft", candidate, ref, d["val"], rm, items, cfg, device, quick=True
            )
            candidates.append((label, candidate, summary))
        label, m, s = max(candidates, key=lambda a: key(a[2], base))
        save(out / "sft.pt", m)
        dump(
            out / "sft_selection.json",
            dict(
                choice=label,
                criterion="same raw reward under95% relevance guard as RL",
                candidates=[dict(checkpoint=a, **c) for a, _, c in candidates],
            ),
        )
        rows.append(dict(method="sft", **s))
        for method in ["mixed100"]:
            grid = []
            for lam in [0, 2, 5, 10]:
                s, _, _ = evaluate(
                    method,
                    ref,
                    ref,
                    d["val"],
                    rm,
                    items,
                    cfg,
                    device,
                    lam=lam,
                    quick=True,
                )
                grid.append(dict(lam=lam, **s))
            choice = max(grid, key=lambda s: key(s, base))
            dump(out / f"{method}_selection.json", dict(choice=choice, grid=grid))
            rows.append(dict(method=method, **choice))
        s, _, _ = evaluate(
            "exact", ref, ref, d["val"], rm, items, cfg, device, quick=True
        )
        rows.append(dict(method="exact", **s))
        s, _, _ = evaluate(
            "exact_duration", ref, ref, d["val"], rm, items, cfg, device, quick=True
        )
        rows.append(dict(method="exact_duration", **s))
        pd.DataFrame(rows).to_csv(out / "validation_metrics.csv", index=False)
    seal(root)


def finalize(root, cfg, ids, a, specs, rm, items, d, device):
    sealdata = json.loads((root / "POLICY_SEAL.json").read_text())
    verify_seal(root / "POLICY_SEAL.json", root)
    out = root / "final"
    out.mkdir(exist_ok=False)
    dump(
        out / "OPENED.json",
        dict(
            utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            seal_sha256=digest(root / "POLICY_SEAL.json"),
        ),
    )
    prepare_raw(cfg, out / "policy_data", holdout=True)
    rcfg = cfg | dict(holdout_per_user=cfg["rm_final_per_user"])
    prepare_raw(rcfg, out / "reward_data", holdout=True)
    raw = dict(np.load(out / "policy_data/holdout_raw.npz"))
    hd = map_data(raw, ids)
    rd = dict(np.load(out / "reward_data/holdout_raw.npz"))
    evalrm = Evaluator(root, a, ids)
    rows = []
    errors = {}
    for kind, spec in specs.items():
        p = predict(rd, a, spec)
        e = error_frame(rd, p)
        errors[kind] = e
        e.to_csv(out / f"rm_{kind}_errors.csv", index=False)
        pa = matched(rd, p)
        pa.to_csv(out / f"rm_{kind}_pairs.csv", index=False)
        rows.append(
            dict(
                kind=kind,
                **metrics(rd["y"], p, rd["w"]),
                within_mse=e.within.mean(),
                between_mse=e.between.mean(),
                matched_users=len(pa),
                matched_pairs=int(pa.pairs.sum()) if len(pa) else 0,
                concordance=pa.concordance.mean() if len(pa) else np.nan,
            )
        )
    rdm = map_data(rd, ids)
    ix = np.arange(len(rd["uid"]))
    ep = []
    # Evaluator accepts action indices only for in-catalog candidates; raw OOV prediction separately.
    from reward3 import design

    ep = evalrm.raw(rd)
    e = error_frame(rd, ep)
    rows.append(
        dict(
            kind="evaluator",
            **metrics(rd["y"], ep, rd["w"]),
            within_mse=e.within.mean(),
            between_mse=e.between.mean(),
        )
    )
    pd.DataFrame(rows).to_csv(out / "reward_metrics.csv", index=False)
    for kind in ["candidate", "joint", "residual"]:
        e = errors[kind].merge(errors["history"], on="uid", suffixes=("_a", "_h"))
        dump(
            out / f"rm_{kind}_incremental.json",
            dict(within=ci(e.within_a - e.within_h), mse=ci(e.mse_a - e.mse_h)),
        )
    metricsrows = []
    lat = []
    hrows = []
    slaterows = []
    from horizon2 import real_contexts, diagnose, greedy_slates

    rng = np.random.default_rng(42)
    users = np.unique(hd["uid"])
    chosen = np.sort(rng.choice(users, min(200, len(users)), False))
    real = real_contexts(raw, chosen, ids, cfg, cfg["val_end"], cfg["test_end"])
    np.savez_compressed(out / "real_diagnostic_contexts.npz", **real)
    # Metadata used for descriptive diversity only; no fitting or selection.
    mapping = pl.read_parquet(B.parents[1] / "data/raw/artist_item_mapping.parquet")
    cols = mapping.columns
    if set(["item_id", "artist_id"]).issubset(cols):
        index = {int(i): j for j, i in enumerate(ids)}
        artists = [set() for _ in ids]
        for item, artist in mapping.select(["item_id", "artist_id"]).iter_rows():
            j = index.get(int(item))
            if j is not None:
                artists[j].add(int(artist))
        items["artists"] = artists
    for seed in cfg["seeds"]:
        p = root / f"seed_{seed}"
        ref = freeze(load(p / "base.pt", device))
        methods = [
            "pure",
            "base",
            "sft",
            "context_count",
            "popularity",
            "ablate_pop",
            "ablate_uniform",
            "ablate_pop_tuned",
            "ablate_uniform_tuned",
            "mixed100",
            "exact",
            "exact_duration",
            "rl_h1_raw",
            "rl_h5_raw",
            "rl_h1_duration",
        ]
        for method in methods:
            lam = ref.alpha
            actual = method.replace("_tuned", "")
            m = ref
            if method in ["pure", "sft"] or method.startswith("rl"):
                m = freeze(
                    load(
                        p / ("pure/best.pt" if method == "pure" else f"{method}.pt"),
                        device,
                    )
                )
            if method == "mixed100":
                lam = json.loads((p / "mixed100_selection.json").read_text())["choice"][
                    "lam"
                ]
            if method.endswith("_tuned"):
                lam = json.loads((p / f"{actual}_selection.json").read_text())[
                    "choice"
                ]["alpha"]
            summary, frame, rank = evaluate(
                actual, m, ref, hd, rm, items, cfg, device, lam=lam, evaluator=evalrm
            )
            frame.to_csv(out / f"{seed}_{method}_events.csv", index=False)
            np.save(out / f"{seed}_{method}_top20.npy", rank)
            metricsrows.append(dict(seed=seed, method=method, lam=lam, **summary))
            lat.append(
                dict(
                    seed=seed,
                    method=method,
                    **latency(actual, m, ref, hd, rm, items, cfg, device, lam=lam),
                )
            )
            print(
                "FINAL", seed, method, summary["ndcg20"], summary["reward"], flush=True
            )
            if method in [
                "base",
                "rl_h1_raw",
                "rl_h5_raw",
                "rl_h1_duration",
                "exact",
                "exact_duration",
            ]:
                hrows.extend(
                    diagnose(actual, m, ref, real, rm, evalrm, items, cfg, device, seed)
                )
            if method in [
                "pure",
                "base",
                "sft",
                "context_count",
                "mixed100",
                "rl_h1_raw",
                "rl_h5_raw",
                "rl_h1_duration",
                "exact",
                "exact_duration",
            ]:
                slaterows.extend(
                    [
                        dict(seed=seed, method=method, **r)
                        for r in greedy_slates(
                            actual, m, ref, hd, rm, evalrm, items, cfg, device, lam
                        )
                    ]
                )
    pd.DataFrame(metricsrows).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(lat).to_csv(out / "latency.csv", index=False)
    pd.DataFrame(hrows).to_csv(out / "horizon.csv", index=False)
    pd.DataFrame(slaterows).to_csv(out / "slates.csv", index=False)
    verify_seal(root / "POLICY_SEAL.json", root)
    dump(
        out / "COMPLETE.json",
        dict(
            policies_unchanged=True,
            users=len(users),
            targets=len(hd["uid"]),
            diagnostic_users=len(np.unique(real["uid"])),
            reward_targets=len(rd["uid"]),
            finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        ),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["reward", "policies", "final"])
    p.add_argument("--device", default="mps")
    p.add_argument("--out", type=Path, default=B / "runs/study_03")
    a = p.parse_args()
    root = a.out
    torch.set_num_threads(6)
    start = time.monotonic()
    if a.stage == "reward":
        from reward3 import fit

        fit(root)
    else:
        args = assets(root, a.device)
        if a.stage == "policies":
            cfg, ids, asset, specs, rm, items, d = args
            policies(root, cfg, rm, items, d, a.device)
        else:
            finalize(root, *args, a.device)
    dump(root / f"time_{a.stage}.json", dict(seconds=time.monotonic() - start))


if __name__ == "__main__":
    main()
