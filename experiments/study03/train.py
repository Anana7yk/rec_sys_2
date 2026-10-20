"""Bounded controlled supervised search; no final cohort access."""

import sys, json, time, math, copy, argparse
from pathlib import Path
import numpy as np, pandas as pd, torch

B = Path(__file__).resolve().parent
sys.path.insert(0, str(B.parents[1] / "recsys"))
from data2 import dump, map_data
from models2 import HybridGPT, seed_all, save, load, sync


def tensor(x, device):
    return torch.as_tensor(np.asarray(x).astype("int64"), device=device)


class Pool:
    def __init__(self, root):
        for k in ["x", "g", "context", "target", "offset", "users", "uid", "t"]:
            setattr(self, k, np.load(root / f"{k}.npy", mmap_mode="r"))

    def draw(self, rng, b):
        u = rng.integers(len(self.users), size=b)
        ix = self.offset[u] + (
            rng.random(b) * (self.offset[u + 1] - self.offset[u])
        ).astype("int64")
        ci = self.context[ix]
        return self.x[ci], self.g[ci], self.target[ix]


@torch.no_grad()
def ranking(m, d, device, batch=64):
    m.eval()
    frames = []
    ce = []
    for s in range(0, len(d["x"]), batch):
        ix = slice(s, s + batch)
        z = m(tensor(d["x"][ix], device), tensor(d["g"][ix], device))
        y = d["target"][ix] - 3
        r = z.topk(20, -1).indices.cpu().numpy()
        hit = r == y[:, None]
        ok = y >= 0
        loss = np.full(len(y), np.nan)
        if ok.any():
            loss[ok] = (
                torch.nn.functional.cross_entropy(
                    z[tensor(np.flatnonzero(ok), device)],
                    tensor(y[ok], device),
                    reduction="none",
                )
                .cpu()
                .numpy()
            )
        for j in range(len(y)):
            row = dict(uid=int(d["uid"][s + j]), ce=loss[j])
            for k in [10, 20]:
                row[f"ndcg{k}"] = float((hit[j, :k] / np.log2(np.arange(k) + 2)).sum())
                row[f"recall{k}"] = float(hit[j, :k].any())
            frames.append(row)
    f = pd.DataFrame(frames)
    return f.groupby("uid").mean().mean().to_dict(), f


@torch.no_grad()
def probe_loss(m, probe, device):
    m.eval()
    loss = []
    for s in range(0, len(probe[0]), 64):
        loss.append(
            float(
                torch.nn.functional.cross_entropy(
                    m(
                        tensor(probe[0][s : s + 64], device),
                        tensor(probe[1][s : s + 64], device),
                    ),
                    tensor(probe[2][s : s + 64], device),
                )
            )
        )
    return float(np.mean(loss))


def fit(
    root,
    name,
    seed,
    d=128,
    lr=0.0003,
    wd=0.01,
    steps=8000,
    fixed=False,
    device="mps",
    initial=None,
    seconds=None,
):
    cfg = json.loads((root / "config.json").read_text())
    n = len(np.load(root / "catalog_ids.npy"))
    ids = np.load(root / "catalog_ids.npy")
    pool = Pool(root / "pool")
    rng = np.random.default_rng(seed)
    seed_all(seed)
    torch.set_num_threads(6)
    model = (
        HybridGPT(n, cfg["context"], d, 2, 4).to(device)
        if initial is None
        else copy.deepcopy(initial)
    )
    for p in model.parameters():
        p.requires_grad_(True)
    cat = pd.read_csv(root / "catalog.csv")
    model.popularity.copy_(
        torch.tensor(
            cat["count"].to_numpy() / cat["count"].sum(),
            device=device,
            dtype=torch.float32,
        )
    )
    val = map_data(dict(np.load(root / "data/val_raw.npz")), ids)
    tr = map_data(dict(np.load(root / "data/train_raw.npz")), ids)
    ok = tr["target"] >= 3
    prob = np.zeros(len(ok))
    u, ix, counts = np.unique(tr["uid"][ok], return_inverse=True, return_counts=True)
    prob[ok] = 1 / counts[ix]
    prob /= prob.sum()
    prng = np.random.default_rng(817)
    probe = pool.draw(prng, 1024)
    out = root / name
    out.mkdir(parents=True, exist_ok=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best = -1
    logs = []
    seen = np.zeros(n, bool)
    losses = []
    compute = 0
    start = time.monotonic()
    start_wall = time.time()
    step = 0
    if initial is not None:
        met, _ = ranking(model, val, device)
        best = met["ndcg20"]
        save(out / "best.pt", model, step=0)
        logs.append(
            dict(
                step=0,
                train_loss=np.nan,
                train_probe=probe_loss(model, probe, device),
                **met,
            )
        )
    while step < steps if seconds is None else compute < seconds:
        step += 1
        if fixed:
            ix = rng.choice(len(tr["x"]), 64, p=prob)
            x, g, y = tr["x"][ix], tr["g"][ix], tr["target"][ix] - 3
        else:
            x, g, y = pool.draw(rng, 64)
        seen[y] = True
        model.train()
        warm = 1 if initial is not None else min(1, step / 500)
        decay = 0.1 + 0.9 * (1 + math.cos(math.pi * min(1, step / steps))) / 2
        for group in opt.param_groups:
            group["lr"] = lr if initial is not None else lr * warm * decay
        xx, gg, yy = tensor(x, device), tensor(g, device), tensor(y, device)
        sync(device)
        t = time.perf_counter()
        z = model(xx, gg)
        loss = torch.nn.functional.cross_entropy(z, yy)
        opt.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt.step()
        sync(device)
        compute += time.perf_counter() - t
        losses.append(float(loss.detach()))
        if (
            step % 2000 == 0
            or (seconds is None and step == steps)
            or (seconds is not None and compute >= seconds)
        ):
            met, frame = ranking(model, val, device)
            trce = probe_loss(model, probe, device)
            rec = dict(
                step=step,
                train_loss=float(np.mean(losses)),
                train_probe=trce,
                positive_items_drawn=int(seen.sum()),
                compute=compute,
                **met,
            )
            logs.append(rec)
            losses = []
            print(name, seed, rec, flush=True)
            if met["ndcg20"] > best:
                best = met["ndcg20"]
                save(out / "best.pt", model, step=step)
                np.save(out / "selected_seen_positive.npy", seen)
                frame.to_csv(out / "validation_per_event.csv", index=False)
            pd.DataFrame(logs).to_csv(out / "learning.csv", index=False)
    save(out / "last.pt", model, step=step)
    np.save(out / "seen_positive.npy", seen)
    dump(
        out / "timing.json",
        dict(
            seed=seed,
            dim=d,
            lr=lr,
            weight_decay=wd,
            fixed=fixed,
            steps=step,
            compute_seconds=compute,
            total_seconds=time.monotonic() - start,
            wall_seconds=time.time() - start_wall,
            positive_items_drawn=int(seen.sum()),
            parameters=sum(p.numel() for p in model.parameters()),
            selected_step=torch.load(
                out / "best.pt", map_location="cpu", weights_only=False
            )["step"],
        ),
    )
    return load(out / "best.pt", device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["pilot", "main"])
    p.add_argument("--device", default="mps")
    p.add_argument("--out", type=Path, default=B / "runs/study_03")
    a = p.parse_args()
    root = a.out
    if a.stage == "pilot":
        specs = [
            ("fixed128", 128, 0.0003, 0.01, True),
            ("dynamic128", 128, 0.0003, 0.01, False),
            ("dynamic128_fast", 128, 0.0007, 0.01, False),
            ("dynamic192", 192, 0.0003, 0.05, False),
        ]
        res = []
        for name, d, lr, wd, fixed in specs:
            fit(root, "pilot/" + name, 11, d, lr, wd, 8000, fixed, a.device)
            log = pd.read_csv(root / "pilot" / name / "learning.csv")
            res.append(
                dict(
                    name=name,
                    d=d,
                    lr=lr,
                    wd=wd,
                    fixed=fixed,
                    best_ndcg20=float(log.ndcg20.max()),
                )
            )
        chosen = max([s for s in res if not s["fixed"]], key=lambda s: s["best_ndcg20"])
        dump(
            root / "supervised_selection.json",
            dict(choice=chosen, grid=res, final_unopened=True),
        )
        print("CHOSEN", chosen, flush=True)
    else:
        c = json.loads((root / "supervised_selection.json").read_text())["choice"]
        for seed in [11, 29, 47]:
            fit(
                root,
                f"seed_{seed}/pure",
                seed,
                c["d"],
                c["lr"],
                c["wd"],
                24000,
                False,
                a.device,
            )


if __name__ == "__main__":
    main()
