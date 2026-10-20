"""Common full-catalog protocol, with explicit distributions for inference controls."""

import time
import numpy as np
import pandas as pd
import torch
from models2 import logp, categorical_kl, sync


def tensor(x, device):
    return torch.as_tensor(x, device=device)


def scores(method, m, ref, x, g, fb, rm, pop, lam=0):
    if method == "popularity":
        lp = (
            torch.tensor(np.log(pop), dtype=torch.float32, device=x.device)
            .expand(len(x), -1)
            .log_softmax(-1)
        )
    elif method in ["context_count", "ablate_pop", "ablate_uniform"]:
        counts = ref.count(x)
        if method == "context_count":
            lp = counts.log()
        else:
            base = (
                ref.popularity.expand(len(x), -1)
                if method == "ablate_pop"
                else torch.full_like(counts, 1 / counts.shape[1])
            )
            alpha = lam
            lp = ((1 - alpha) * base + alpha * counts).log()
    else:
        lp = logp(m if m is not None else ref, x, g)
        if method in ["mixed_full", "exact", "exact_duration"]:
            factor = 1 / getattr(rm, "beta", 0.1) if method.startswith("exact") else lam
            reward = rm.all(x, fb)
            if method == "exact_duration":
                reward = reward - rm.correction[None, :]
            lp = (lp + factor * reward).log_softmax(-1)
        elif method == "mixed100":
            pool = lp.topk(min(100, lp.shape[1]), -1).indices
            b, k = pool.shape
            rr = rm.actions(
                x[:, None, :].expand(b, k, -1).reshape(b * k, -1),
                fb[:, None, :].expand(b, k, -1).reshape(b * k, -1),
                pool.reshape(-1),
            ).reshape(b, k)
            values = lp.gather(1, pool) + lam * rr
            lp = (
                torch.full_like(lp, -torch.inf).scatter(1, pool, values).log_softmax(-1)
            )
    return lp


@torch.no_grad()
def evaluate(
    method, m, ref, d, rm, items, cfg, device, lam=0, evaluator=None, quick=False
):
    if m is not None:
        m.eval()
    if ref is not None:
        ref.eval()
    rows = []
    ranks = []
    n = len(items["ids"])
    pop = items["pop"]
    dur = items["duration"]
    positive_train = items.get("positive_train", np.ones(n, dtype=np.int8))
    violations = 0
    for start in range(0, len(d["x"]), cfg["batch"]):
        sl = slice(start, start + cfg["batch"])
        x = tensor(d["x"][sl], device)
        g = tensor(d["g"][sl], device)
        fb = tensor(d["fb"][sl], device)
        lp = scores(method, m, ref, x, g, fb, rm, pop, lam)
        rank = lp.topk(20, -1).indices
        r = rm.all(x, fb)
        r20 = r.gather(1, rank).mean(1)
        adjusted20 = (r - rm.correction[None, :]).gather(1, rank).mean(1)
        lref = logp(ref, x, g) if ref is not None else lp
        kl = categorical_kl(lp, lref)
        expected = (lp.exp() * r).sum(1)
        objective = expected - cfg["beta"] * kl
        if method.startswith("rl"):
            star = (lref + r / cfg["beta"]).log_softmax(-1)
            upper = (star.exp() * r).sum(1) - cfg["beta"] * categorical_kl(star, lref)
            violations += int((objective > upper + 2e-5).sum().cpu())
            assert violations == 0, "RL exceeds exact H1 optimum on same states"
        a = rank.cpu().numpy()
        ranks.append(a)
        target = d["target"][sl] - 3
        hit = a == target[:, None]
        fbk = d["fb"][sl]
        cr20 = r20.cpu().numpy()
        cadjusted20 = adjusted20.cpu().numpy()
        cexp = expected.cpu().numpy()
        ckl = kl.cpu().numpy()
        cobj = objective.cpu().numpy()
        rank_reward = r.gather(1, rank).cpu().numpy() if evaluator is not None else None
        evalpred = None
        if evaluator is not None:
            evalpred = evaluator.candidates({k: v[sl] for k, v in d.items()}, a)
        for j in range(len(x)):
            row = dict(
                uid=int(d["uid"][start + j]),
                t=int(d["t"][start + j]),
                target_seen_history=float(
                    np.any(d["raw"][start + j] == d["item"][start + j])
                ),
                in_catalog=float(target[j] >= 0),
                reward=float(cr20[j]),
                adjusted_reward=float(cadjusted20[j]),
                expected_reward=float(cexp[j]),
                kl=float(ckl[j]),
                objective=float(cobj[j]),
            )
            if not quick and "positive_train" in items:
                row["target_positive_train"] = float(
                    target[j] >= 0 and positive_train[max(0, target[j])] > 0
                )
            for k in [10, 20]:
                row[f"recall{k}"] = float(hit[j, :k].any())
                row[f"ndcg{k}"] = float((hit[j, :k] / np.log2(np.arange(k) + 2)).sum())
            if not quick:
                hh = d["x"][start + j]
                row.update(
                    duration=float(dur[a[j]].mean()),
                    top_decile=float((a[j] < max(1, int(np.ceil(n * 0.1)))).mean()),
                    history_repeat=float(
                        (a[j, :, None] + 3 == hh[None, :]).any(1).mean()
                    ),
                    logpop=float(np.log1p(pop[a[j]]).mean()),
                )
                if "artists" in items:
                    art = [items["artists"][i] for i in a[j]]
                    pairs = [
                        1 - len(aa & bb) / len(aa | bb)
                        for ii, aa in enumerate(art)
                        if aa
                        for bb in art[ii + 1 :]
                        if bb
                    ]
                    row.update(
                        artist_distance=(
                            float(np.mean(pairs)) if pairs else float("nan")
                        ),
                        artist_known=float(np.mean([bool(aa) for aa in art])),
                    )
                if evalpred is not None:
                    row.update(
                        evaluator_reward=float(evalpred[j].mean()),
                        disagreement=float(np.abs(evalpred[j] - rank_reward[j]).mean()),
                    )
            rows.append(row)
    frame = pd.DataFrame(rows)
    macro = frame.groupby("uid").mean(numeric_only=True)
    summary = macro.mean().to_dict()
    summary = {k: float(v) for k, v in summary.items() if k not in ["t"]}
    ranks = np.concatenate(ranks)
    freq = np.bincount(
        ranks.ravel(), weights=np.repeat(d["w"], ranks.shape[1]), minlength=n
    )
    p = freq / freq.sum()
    summary.update(
        coverage=float((freq > 0).mean()),
        hhi=float((p * p).sum()),
        n_users=len(macro),
        n_events=len(frame),
        analytic_bound_violations=violations,
    )
    inc = frame[frame.in_catalog == 1].groupby("uid").mean(numeric_only=True)
    for k in [10, 20]:
        summary[f"conditional_ndcg{k}"] = float(inc[f"ndcg{k}"].mean())
        summary[f"conditional_recall{k}"] = float(inc[f"recall{k}"].mean())
    if not quick:
        _, idx = np.unique(d["uid"], return_index=True)
        rng = np.random.default_rng(4321)
        over = []
        for _ in range(min(2000, len(idx) * 4)):
            i, j = rng.choice(idx, 2, replace=False)
            aa = set(ranks[i])
            bb = set(ranks[j])
            over.append(len(aa & bb) / len(aa | bb))
        summary["user_jaccard"] = float(np.mean(over))
        durations = dur[ranks].ravel()
        order = np.argsort(durations)
        cw = np.cumsum(np.repeat(d["w"], ranks.shape[1])[order])
        cw /= cw[-1]
        summary.update(
            {
                f"duration_p{q}": float(
                    durations[
                        order[np.searchsorted(cw, q / 100).clip(0, len(order) - 1)]
                    ]
                )
                for q in [10, 50, 90]
            }
        )
    return summary, frame, ranks


def key(s, base, metric="reward"):
    return (1, s[metric]) if s["ndcg20"] >= base["ndcg20"] * 0.95 else (0, s["ndcg20"])


@torch.no_grad()
def latency(method, m, ref, d, rm, items, cfg, device, lam=0):
    x = tensor(d["x"][: cfg["batch"]], device)
    g = tensor(d["g"][: cfg["batch"]], device)
    fb = tensor(d["fb"][: cfg["batch"]], device)
    times = []
    for i in range(7):
        sync(device)
        start = time.perf_counter()
        scores(method, m, ref, x, g, fb, rm, items["pop"], lam).topk(20, -1)
        sync(device)
        if i >= 2:
            times.append(time.perf_counter() - start)
    return dict(batch=len(x), median_seconds=float(np.median(times)), repeats=times)
