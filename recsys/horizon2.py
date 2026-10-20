"""Real-prefix versus artificial-slate diagnostics with shared sampling uniforms."""

import numpy as np
import polars as pl
import torch
from data2 import ROOT, pack, feedback, map_data
from models2 import logp, append, categorical_kl, seed_all
from metrics2 import tensor, scores


def inverse_cdf_cpu(log_prob, uniform):
    """Float64 normalized CDF; right search never selects a zero-mass plateau."""
    p = np.exp(np.asarray(log_prob, dtype=np.float64))
    p /= p.sum(1, keepdims=True)
    cdf = np.cumsum(p, axis=1)
    cdf[:, -1] = 1.0
    return np.array(
        [
            np.searchsorted(row, u, side="right")
            for row, u in zip(cdf, np.asarray(uniform).ravel())
        ],
        dtype=np.int64,
    )


def real_contexts(rawdata, users, ids, cfg, start, end):
    ev = (
        pl.scan_parquet(ROOT / "data/raw/multi_event.parquet")
        .filter(
            (pl.col("event_type") == "listen") & pl.col("uid").is_in(users.tolist())
        )
        .collect()
        .sort(["uid", "timestamp"])
    )
    records = []
    for frame in ev.partition_by("uid", maintain_order=True):
        uid = int(frame["uid"][0])
        t = frame["timestamp"].to_numpy().astype("int64")
        it = frame["item_id"].to_numpy().astype("int64")
        y = frame["played_ratio_pct"].to_numpy().astype(float) / 100
        duration = frame["track_length_seconds"].to_numpy().astype(float)
        ts, first, count = np.unique(t, return_index=True, return_counts=True)
        groups = [np.sort(it[a : a + n]) for a, n in zip(first, count)]
        initialrow = np.flatnonzero(rawdata["uid"] == uid)[0]
        initial = rawdata["t"][initialrow]
        gi = np.searchsorted(ts, initial)
        if gi + 4 >= len(ts) or ts[gi + 4] >= end:
            continue
        for step in range(5):
            ix = gi + step
            raw, g, ht, lost = pack(groups, ts, ix, cfg["context"])
            fb, fbmax = feedback(t, it, y, duration, ts[ix])
            records.append(
                dict(
                    uid=uid,
                    step=step + 1,
                    t=int(ts[ix]),
                    item=int(
                        rawdata["item"][initialrow] if step == 0 else it[first[ix]]
                    ),
                    raw=raw,
                    g=g,
                    fb=fb,
                )
            )
    data = {k: np.stack([r[k] for r in records]) for k in records[0]}
    return map_data(data, ids)


@torch.no_grad()
def diagnose(method, model, ref, real, rm, evaluator, items, cfg, device, seed):
    seed_all(seed + 7000)
    users = np.unique(real["uid"])
    pos = np.array([np.flatnonzero(real["uid"] == u)[0] for u in users])
    x = tensor(real["x"][pos], device)
    g = tensor(real["g"][pos], device)
    fb = tensor(real["fb"][pos], device)
    used = torch.zeros((len(x), len(items["ids"])), dtype=torch.bool, device=device)
    rows = []
    actions = {"real_prefix": [], "synthetic": []}
    initial_real = None
    initial_action = None
    for t in range(5):
        # Same uniform implies identical first actions for identical first states.
        u = torch.rand((len(x), 1), device=device)
        for mode in ["real_prefix", "synthetic"]:
            if mode == "real_prefix":
                xx = tensor(real["x"][pos + t], device)
                gg = tensor(real["g"][pos + t], device)
                ff = tensor(real["fb"][pos + t], device)
                # Controlled state comparison: both modes exclude the same previous
                # synthetic recommendations. This is not a logged recommendation list.
                mask = used
            else:
                xx = x
                gg = g
                ff = fb
                mask = used
            premask = scores(
                method if method.startswith("exact") else "base",
                model,
                ref,
                xx,
                gg,
                ff,
                rm,
                items["pop"],
            )
            lp = (
                premask
                if mask is None
                else premask.masked_fill(mask, -torch.inf).log_softmax(-1)
            )
            lref = logp(ref, xx, gg, mask)
            if t == 0 and mode == "synthetic":
                assert initial_real is not None and initial_action is not None
                assert (
                    torch.equal(xx, initial_real[0])
                    and torch.equal(gg, initial_real[1])
                    and torch.equal(ff, initial_real[2])
                )
                a = initial_action.clone()
            else:
                a = tensor(inverse_cdf_cpu(lp.cpu().numpy(), u.cpu().numpy()), device)
                if t == 0:
                    initial_real = (xx.clone(), gg.clone(), ff.clone())
                    initial_action = a.clone()
            assert not bool(used[torch.arange(len(x), device=device), a].any())
            r = rm.actions(xx, ff, a)
            dd = dict(x=xx.cpu().numpy(), fb=ff.cpu().numpy())
            ep = evaluator.candidates(dd, a.cpu().numpy()[:, None])[:, 0]
            k = categorical_kl(lp, lref)
            repeat = (xx == a[:, None] + 3).any(1).cpu().numpy()
            known = (
                ((xx >= 3).sum(1) / ((xx != 0) & (xx != 1)).sum(1).clamp_min(1))
                .cpu()
                .numpy()
            )
            mass = (premask.exp() * used).sum(1).cpu().numpy()
            ar = a.cpu().numpy()
            actions[mode].append(ar)
            for j, uid in enumerate(users):
                rows.append(
                    dict(
                        seed=seed,
                        method=method,
                        mode=mode,
                        step=t + 1,
                        uid=int(uid),
                        reward=float(r[j]),
                        adjusted_reward=float(r[j] - rm.correction[a[j]]),
                        evaluator_reward=float(ep[j]),
                        disagreement=float(abs(r[j].item() - ep[j])),
                        kl=float(k[j]),
                        duration=float(items["duration"][ar[j]]),
                        logpop=float(np.log1p(items["pop"][ar[j]])),
                        history_repeat=float(repeat[j]),
                        known_fraction=float(known[j]),
                        unmasked_list_repeat_mass=float(mass[j]),
                        action=int(ar[j]),
                    )
                )
            if mode == "synthetic":
                used = used.clone()
                used[torch.arange(len(x), device=device), a] = True
                x, g = append(xx, gg, a)
    assert np.array_equal(actions["real_prefix"][0], actions["synthetic"][0])
    return rows


@torch.no_grad()
def greedy_slates(method, model, ref, d, rm, evaluator, items, cfg, device, lam=0):
    """Five recommendations; relevance always to the original next logged item."""
    rows = []
    for start in range(0, len(d["x"]), cfg["batch"]):
        sl = slice(start, start + cfg["batch"])
        x = tensor(d["x"][sl], device)
        g = tensor(d["g"][sl], device)
        fb = tensor(d["fb"][sl], device)
        used = torch.zeros((len(x), len(items["ids"])), device=device, dtype=torch.bool)
        actions = []
        rewards = []
        checks = []
        repeat = []
        for step in range(5):
            lp = scores(
                method, model, ref, x, g, fb, rm, items["pop"], lam
            ).masked_fill(used, -torch.inf)
            a = lp.argmax(-1)
            actions.append(a.cpu().numpy())
            rewards.append(rm.actions(x, fb, a).cpu().numpy())
            checks.append(
                evaluator.candidates(
                    dict(x=x.cpu().numpy(), fb=fb.cpu().numpy()),
                    a.cpu().numpy()[:, None],
                )[:, 0]
            )
            repeat.append((x == a[:, None] + 3).any(1).cpu().numpy())
            used = used.clone()
            used[torch.arange(len(x), device=device), a] = True
            x, g = append(x, g, a)
        aa = np.stack(actions, 1)
        rr = np.stack(rewards, 1)
        cc = np.stack(checks, 1)
        rp = np.stack(repeat, 1)
        target = d["target"][sl] - 3
        hit = aa == target[:, None]
        for j in range(len(x)):
            rows.append(
                dict(
                    uid=int(d["uid"][start + j]),
                    t=int(d["t"][start + j]),
                    recall5=float(hit[j].any()),
                    ndcg5=float((hit[j] / np.log2(np.arange(5) + 2)).sum()),
                    reward=float(rr[j].mean()),
                    evaluator_reward=float(cc[j].mean()),
                    history_repeat=float(rp[j].mean()),
                    duration=float(items["duration"][aa[j]].mean()),
                    actions=aa[j].tolist(),
                )
            )
    return rows
