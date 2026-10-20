"""Frozen study02 checkpoint on the NEW common evaluation contexts; no retuning."""

import sys, json, hashlib, time, argparse
from pathlib import Path
import numpy as np, pandas as pd, torch

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B.parents[1] / "recsys"))
sys.path.insert(0, str(B))
from models2 import HybridGPT, save, load, freeze
from data2 import dump, map_data
from metrics2 import evaluate, latency
from run import assets


class MarginalControl(torch.nn.Module):
    """Replace GPT by the exact full-train user-balanced item marginal."""

    def __init__(self, prior, pop, alpha, device):
        super().__init__()
        self.alpha = alpha
        self.register_buffer(
            "prior", torch.tensor(prior, dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "pop", torch.tensor(pop / pop.sum(), dtype=torch.float32, device=device)
        )

    def forward(self, x, g):
        if self.alpha == 0:
            return self.prior.log().expand(len(x), -1)
        counts = torch.zeros((len(x), len(self.prior)), device=x.device)
        counts.scatter_add_(1, (x - 3).clamp_min(0), (x >= 3).float())
        total = counts.sum(1, keepdim=True)
        cp = torch.where(
            total > 0, 0.999 * counts / total.clamp_min(1) + 0.001 * self.pop, self.pop
        )
        return ((1 - self.alpha) * self.prior + self.alpha * cp).log()


def marginal(root):
    assert not (root / "final").exists(), "Materialize control before final access"
    p = root / "reward/policy_marginal.npz"
    if p.exists():
        return
    target = np.load(root / "pool/target.npy", mmap_mode="r")
    offset = np.load(root / "pool/offset.npy")
    w = np.repeat(1 / np.diff(offset), np.diff(offset))
    n = len(np.load(root / "catalog_ids.npy"))
    prior = np.bincount(target, weights=w, minlength=n)
    prior /= prior.sum()
    np.savez_compressed(p, prior=prior)
    dump(
        root / "reward/MARGINAL_CONTROL_SEAL.json",
        dict(
            created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            source="all eligible in-catalog train<22M; exact user-first empirical item marginal",
            sha256=hashlib.file_digest(p.open("rb"), "sha256").hexdigest(),
            final_unopened=True,
            alpha="zero, or SAME validation-selected reference alpha; no additional tuning",
        ),
    )


def assess(root, device):
    cfg, ids, asset, specs, rm, items, d = assets(root, device)
    final = root / "final"
    assert (final / "COMPLETE.json").exists()
    hd = map_data(dict(np.load(final / "policy_data/holdout_raw.npz")), ids)
    from reward3 import Evaluator

    evaluator = Evaluator(root, asset, ids)
    rows = []
    lat = []
    prior = np.load(root / "reward/policy_marginal.npz")["prior"]
    # Metadata only, never a fitting feature.
    import polars as pl

    mapping = (
        pl.scan_parquet(B.parents[1] / "data/raw/artist_item_mapping.parquet")
        .filter(pl.col("item_id").is_in(ids.tolist()))
        .collect()
    )
    index = {int(i): j for j, i in enumerate(ids)}
    artists = [set() for _ in ids]
    for item, artist in mapping.select("item_id", "artist_id").iter_rows():
        artists[index[int(item)]].add(int(artist))
    items["artists"] = artists
    for seed in cfg["seeds"]:
        ref = freeze(load(root / f"seed_{seed}/base.pt", device))
        controls = [
            ("fresh_marginal", MarginalControl(prior, items["pop"], 0, device)),
            (
                "fresh_count_ablation",
                MarginalControl(prior, items["pop"], ref.alpha, device),
            ),
        ]
        oldpath = root / f"seed_{seed}/old_reference.pt"
        if oldpath.exists():
            controls.insert(0, ("old_gpt", freeze(load(oldpath, device))))
        for method, model in controls:
            summary, frame, rank = evaluate(
                method, model, ref, hd, rm, items, cfg, device, evaluator=evaluator
            )
            frame.to_csv(final / f"{seed}_{method}_events.csv", index=False)
            np.save(final / f"{seed}_{method}_top20.npy", rank)
            rows.append(
                dict(
                    seed=seed,
                    method=method,
                    lam=ref.alpha if method == "fresh_count_ablation" else 0,
                    **summary,
                )
            )
            lat.append(
                dict(
                    seed=seed,
                    method=method,
                    **latency(method, model, ref, hd, rm, items, cfg, device),
                )
            )
            print("CONTROL_COMMON", seed, method, summary["ndcg20"], flush=True)
    existing = pd.read_csv(final / "metrics.csv")
    assert not set(r["method"] for r in rows) & set(existing.method)
    pd.concat([existing, pd.DataFrame(rows)], ignore_index=True).to_csv(
        final / "metrics.csv", index=False
    )
    pd.concat(
        [pd.read_csv(final / "latency.csv"), pd.DataFrame(lat)], ignore_index=True
    ).to_csv(final / "latency.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["marginal", "assess"])
    p.add_argument("--out", type=Path, default=B / "runs/study_03")
    p.add_argument("--device", default="mps")
    a = p.parse_args()
    torch.set_num_threads(6)
    if a.stage == "marginal":
        marginal(a.out)
    else:
        assess(a.out, a.device)
