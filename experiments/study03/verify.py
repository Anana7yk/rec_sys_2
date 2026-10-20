"""Read-only leakage, rankings, frozen policy and prior-study checks."""

import sys, json, hashlib, argparse
from pathlib import Path
import numpy as np, pandas as pd, torch

B = Path(__file__).resolve().parent
R = B.parents[1]
sys.path.insert(0, str(B.parents[1] / "recsys"))
from data2 import lookup, dump
from provenance import verify_seal


def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=B / "runs/study_03")
    p.add_argument("--report", type=Path, default=B.parents[1] / "validation/study03_verification.json")
    a = p.parse_args()
    root = a.out
    cfg = json.loads((root / "config.json").read_text())
    ids = np.load(root / "catalog_ids.npy")
    checks = {}
    checks["seal_verification"] = verify_seal(root / "POLICY_SEAL.json", root)
    final = root / "final"
    raw = dict(np.load(final / "policy_data/holdout_raw.npz"))
    target = lookup(ids, raw["item"]) - 3
    for path in [
        *list((root / "data").glob("*_raw.npz")),
        *list((root / "reward_data").glob("*_raw.npz")),
        final / "policy_data/holdout_raw.npz",
        final / "reward_data/holdout_raw.npz",
    ]:
        d = dict(np.load(path))
        assert np.all(d["ht"] < d["t"][:, None])
        assert np.all(d["fbmax"] < d["t"])
        assert np.all((d["raw"] == -2).sum(1) == 1)
        assert (
            not np.isin(d["uid"], cfg["holdout_user_ids"]).any()
            if "final" not in path.parts
            else True
        )
        x = lookup(ids, d["raw"])
        assert np.array_equal(x == 0, d["raw"] == -1)
        assert np.array_equal(x == 1, d["raw"] == -2)
    checks["strict_past_arrays"] = True
    metric = pd.read_csv(final / "metrics.csv")
    ranking_checks = 0
    for row in metric.to_dict("records"):
        seed = int(row["seed"])
        method = row["method"]
        rank = np.load(final / f"{seed}_{method}_top20.npy")
        f = pd.read_csv(final / f"{seed}_{method}_events.csv")
        assert rank.shape == (len(raw["uid"]), 20)
        assert rank.min() >= 0 and rank.max() < len(ids)
        assert (np.diff(np.sort(rank, axis=1), axis=1) != 0).all()
        hit = rank == target[:, None]
        for k in [10, 20]:
            assert np.allclose(f[f"recall{k}"], hit[:, :k].any(1))
            assert np.allclose(
                f[f"ndcg{k}"], (hit[:, :k] / np.log2(np.arange(k) + 2)).sum(1)
            )
        assert np.array_equal(f.uid, raw["uid"])
        assert np.array_equal(f.t, raw["t"])
        assert row["analytic_bound_violations"] == 0
        ranking_checks += 1
    checks["independently_recomputed_rankings"] = ranking_checks
    for seed in cfg["seeds"]:
        folder = root / f"seed_{seed}"
        base = torch.load(folder / "base.pt", map_location="cpu", weights_only=False)
        for name in ["rl_h1_raw", "rl_h5_raw", "rl_h1_duration", "sft"]:
            ck = torch.load(
                folder / f"{name}.pt", map_location="cpu", weights_only=False
            )
            assert ck["spec"]["alpha"] == base["spec"]["alpha"]
            assert torch.equal(ck["state"]["popularity"], base["state"]["popularity"])
        t1 = json.loads((folder / "rl_h1_raw_timing.json").read_text())
        t5 = json.loads((folder / "rl_h5_raw_timing.json").read_text())
        assert t1["generated_actions"] == t5["generated_actions"]
    checks["fixed_copy_gate_and_equal_actions"] = True
    checks["final_policy_users"] = len(np.unique(raw["uid"]))
    checks["complete"] = json.loads((final / "COMPLETE.json").read_text())
    assert set(raw["uid"]).issubset(set(cfg["holdout_user_ids"]))
    assert not np.isin(cfg["holdout_user_ids"], cfg["dev_user_ids"]).any()
    dump(a.report, checks)
    print(checks, flush=True)


if __name__ == "__main__":
    main()
