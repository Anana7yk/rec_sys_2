"""Pre-RL selection on observed candidates in the policy's actual action catalog."""

import sys, json, pickle, time, argparse
from pathlib import Path
import numpy as np, pandas as pd

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B.parents[1] / "recsys"))
from data2 import dump, weights
from reward3 import predict, error_frame, ci, metrics, matched


def select(root):
    saved = root / "reward/selection_all_observed.json"
    if (root / "reward/ACTION_DOMAIN_SEAL.json").exists():
        return
    old = json.loads((root / "reward/selection.json").read_text())
    dump(saved, old)
    data = dict(np.load(root / "reward_data/val_raw.npz"))
    ids = np.load(root / "catalog_ids.npy")
    ok = np.isin(data["item"], ids)
    d = {k: v[ok] for k, v in data.items()}
    d["w"] = weights(d["uid"]) if len(d["uid"]) else np.array([], dtype="float32")
    asset = dict(np.load(root / "reward/asset.npz"))
    if not len(d["uid"]):
        result = old | dict(
            gate=False,
            criterion="no in-catalog validation observations; proxy unvalidated",
            eligible_rows=0,
            eligible_users=0,
            final_unopened=True,
            RL_not_started=True,
        )
        dump(root / "reward/selection.json", result)
        dump(root / "reward/ACTION_DOMAIN_SEAL.json", result)
        return

    with (root / "reward/specs.pkl").open("rb") as f:
        specs = pickle.load(f)
    rows = []
    errors = {}
    for kind, spec in specs.items():
        p = predict(d, asset, spec)
        e = error_frame(d, p)
        errors[kind] = e
        e.to_csv(root / f"reward/action_domain_{kind}_errors.csv", index=False)
        pa = matched(d, p)
        pa.to_csv(root / f"reward/action_domain_{kind}_pairs.csv", index=False)
        rows.append(
            dict(
                kind=kind,
                **metrics(d["y"], p, d["w"]),
                within_mse=float(e.within.mean()),
                between_mse=float(e.between.mean()),
                matched_users=len(pa),
                concordance=float(pa.concordance.mean()) if len(pa) else float("nan"),
            )
        )
    h = next(r for r in rows if r["kind"] == "history")
    valid = [
        r
        for r in rows
        if r["kind"] in ["joint", "residual"] and r["rmse"] <= h["rmse"] * 1.005
    ]
    choice = min(
        valid or [r for r in rows if r["kind"] in ["joint", "residual"]],
        key=lambda r: r["within_mse"],
    )
    kind = choice["kind"]
    e = errors[kind].merge(errors["history"], on="uid", suffixes=("_a", "_h"))
    delta = ci(e.within_a - e.within_h)
    gate = bool(valid and delta["high"] < 0)
    pd.DataFrame(rows).to_csv(root / "reward/action_domain_metrics.csv", index=False)
    new = dict(
        primary=kind,
        gate=gate,
        incremental_within_mse=delta,
        choice=choice,
        history=h,
        criterion="within-user MSE and1.005 RMSE history guard on observed in-catalog validation events; same frozen fitted specs",
        all_observed_initial=old,
        eligible_rows=len(d["uid"]),
        eligible_users=len(np.unique(d["uid"])),
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        final_unopened=True,
        RL_not_started=True,
    )
    dump(root / "reward/selection.json", new)
    dump(root / "reward/ACTION_DOMAIN_SEAL.json", new)
    print(new, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=B / "runs/study_03")
    a = p.parse_args()
    select(a.out)
