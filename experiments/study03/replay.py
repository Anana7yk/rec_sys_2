"""Recompute scoring from saved weights on fixed contexts, never choose anything."""

import sys, json, argparse, time
from pathlib import Path
import numpy as np, pandas as pd, torch

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B.parents[1] / "recsys"))
sys.path.insert(0, str(B))
from run import assets
from models2 import load, freeze
from data2 import map_data, dump
from metrics2 import scores, tensor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=B / "runs/study_03")
    p.add_argument("--limit", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--report", type=Path, default=B.parents[1] / "validation/recomputed/study03/checkpoint_replay.json")
    args = p.parse_args()
    root = args.out
    torch.set_num_threads(6)
    cfg, ids, a, specs, rm, items, d = assets(root, args.device)
    hd = map_data(dict(np.load(root / "final/policy_data/holdout_raw.npz")), ids)
    n = min(args.limit, len(hd["uid"]))
    results = []
    with torch.no_grad():
        for seed in cfg["seeds"]:
            directory = root / f"seed_{seed}"
            ref = freeze(load(directory / "base.pt", args.device))
            for method, file in [
                ("pure", "pure/best.pt"),
                ("base", "base.pt"),
                ("sft", "sft.pt"),
                ("rl_h1_raw", "rl_h1_raw.pt"),
                ("rl_h5_raw", "rl_h5_raw.pt"),
                ("rl_h1_duration", "rl_h1_duration.pt"),
                ("old_gpt", "old_reference.pt"),
            ]:
                if not (directory / file).exists():
                    continue
                model = freeze(load(directory / file, args.device))
                x = tensor(hd["x"][:n], args.device)
                g = tensor(hd["g"][:n], args.device)
                fb = tensor(hd["fb"][:n], args.device)
                lp = scores(method, model, ref, x, g, fb, rm, items["pop"])
                rank = lp.topk(20, -1).indices.cpu().numpy()
                expected = np.load(root / f"final/{seed}_{method}_top20.npy")[:n]
                pred = (
                    rm.all(x, fb)
                    .gather(1, torch.tensor(rank, device=args.device))
                    .mean(1)
                    .cpu()
                    .numpy()
                )
                saved = pd.read_csv(
                    root / f"final/{seed}_{method}_events.csv"
                ).reward.to_numpy()[:n]
                assert float((rank == expected).mean()) >= .99, (seed, method)
                results.append(
                    dict(
                        seed=seed,
                        method=method,
                        contexts=n,
                        exact_row_fraction=float((rank == expected).all(1).mean()),
                        element_match_fraction=float((rank == expected).mean()),
                        max_reward_abs_difference=float(np.max(np.abs(pred - saved))),
                    )
                )
    dump(
        args.report,
        dict(
            device=args.device,
            scope="first fixed contexts; pure/base/SFT/RL/old models; no selection or tuning",
            results=results,
        ),
    )
    print(results, flush=True)


if __name__ == "__main__":
    main()
