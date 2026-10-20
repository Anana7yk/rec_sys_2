"""User-balanced supervised fitting and equal-action/time REINFORCE."""

import copy, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from data2 import dump, weights
from models2 import (
    GroupGPT,
    seed_all,
    freeze,
    save,
    load,
    logp,
    append,
    categorical_kl,
    sync,
)
from metrics2 import tensor, evaluate, key


def reinforce_loss(logas, rewards, kls, baselines, ema, beta, gamma):
    """J=sum gamma**t*r_t - beta*sum KL_t, including future-state KL gradients."""
    returns = []
    ret = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        ret = gamma**t * rewards[t].detach() - beta * kls[t].detach() + ret
        returns.insert(0, ret)
    loss = sum(
        (
            -(returns[t] - baselines[t] - ema[t]).detach() * logas[t] + beta * kls[t]
        ).mean()
        for t in range(len(rewards))
    )
    return loss, returns


def supervised(
    cfg,
    data,
    items,
    rm,
    out,
    device,
    seed,
    steps=None,
    seconds=None,
    reference=None,
    learning_rate=None,
):
    seed_all(seed)
    rng = np.random.default_rng(seed)
    n = len(items["ids"])
    model = (
        GroupGPT(n, cfg["context"], cfg["dim"], cfg["layers"], cfg["heads"]).to(device)
        if reference is None
        else copy.deepcopy(reference)
    )
    for p in model.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=(
            learning_rate
            if learning_rate is not None
            else (cfg["lr"] if reference is None else 0.0002)
        ),
        weight_decay=0.01,
    )
    train = data["train"]
    valid = train["target"] >= 3
    prob = train["w"] * valid
    prob /= prob.sum()
    best = -1
    logs = []
    start = time.monotonic()
    step = 0
    compute = 0
    half_checked = False
    if reference is not None:
        initial, _, _ = evaluate(
            "base", model, reference, data["val"], rm, items, cfg, device, quick=True
        )
        best = initial["ndcg20"]
        save(out, model, step=0)
    maxsteps = steps if steps is not None else cfg["base_steps"]
    while step < maxsteps if seconds is None else compute < seconds:
        step += 1
        ix = rng.choice(len(train["x"]), cfg["batch"], p=prob)
        x = tensor(train["x"][ix], device)
        g = tensor(train["g"][ix], device)
        y = tensor(train["target"][ix] - 3, device)
        model.train()
        sync(device)
        t = time.perf_counter()
        loss = torch.nn.functional.cross_entropy(model(x, g), y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        sync(device)
        compute += time.perf_counter() - t
        check = (
            seconds is None
            and (step % (min(cfg["base_check"], maxsteps)) == 0 or step == maxsteps)
        ) or (
            seconds is not None
            and ((compute >= seconds / 2 and not half_checked) or compute >= seconds)
        )
        if seconds is not None and compute >= seconds / 2:
            half_checked = True
        if check:
            summary, _, _ = evaluate(
                "base",
                model,
                reference if reference is not None else model,
                data["val"],
                rm,
                items,
                cfg,
                device,
                quick=True,
            )
            logs.append(dict(step=step, loss=float(loss), **summary))
            print("SFT", seed, Path(out).stem, step, summary["ndcg20"], flush=True)
            if summary["ndcg20"] > best:
                best = summary["ndcg20"]
                save(out, model, step=step)
    pd.DataFrame(logs).to_csv(str(out).replace(".pt", "_log.csv"), index=False)
    dump(
        str(out).replace(".pt", "_timing.json"),
        dict(
            steps=step,
            compute_seconds=compute,
            total_seconds=time.monotonic() - start,
            parameters=sum(p.numel() for p in model.parameters()),
        ),
    )
    return load(out, device), compute


def rl(cfg, data, items, rm, ref, out, device, seed, H, modified, steps, seconds=None):
    seed_all(seed + 2000)
    rng = np.random.default_rng(seed + 2000)
    model = copy.deepcopy(ref)
    for p in model.parameters():
        p.requires_grad_(True)
    model.eval()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["rl_lr"], weight_decay=0.01)
    train = data["train"]
    pool = np.flatnonzero(train["t"] >= cfg["rm_cal_end"])
    prob = weights(train["uid"][pool])
    prob /= prob.sum()
    ema = torch.zeros(H, device=device)
    compute = 0
    step = 0
    half_checked = False
    logs = []
    gradnorms = []
    start = time.monotonic()
    base, _, _ = evaluate(
        "base", ref, ref, data["val"], rm, items, cfg, device, quick=True
    )
    criterion = "adjusted_reward" if modified else "reward"
    best = key(base, base, criterion)
    save(out, model, step=0)
    while step < steps if seconds is None else compute < seconds:
        step += 1
        ix = rng.choice(pool, cfg["batch"], p=prob)
        x = tensor(train["x"][ix], device)
        g = tensor(train["g"][ix], device)
        fb = tensor(train["fb"][ix], device)
        used = torch.zeros((len(x), len(items["ids"])), dtype=torch.bool, device=device)
        logas = []
        kls = []
        rewards = []
        state_baselines = []
        sync(device)
        t0 = time.perf_counter()
        for t in range(H):
            lp = logp(model, x, g, used)
            dist = torch.distributions.Categorical(logits=lp)
            a = dist.sample()
            logas.append(dist.log_prob(a))
            with torch.no_grad():
                lref = logp(ref, x, g, used)
                r = rm.actions(x, fb, a) - (rm.correction[a] if modified else 0)
                allr = rm.all(x, fb) - (rm.correction[None, :] if modified else 0)
                state_baselines.append(
                    sum(cfg["gamma"] ** j for j in range(t, H))
                    * (lref.exp() * allr).sum(1)
                )
            kl = categorical_kl(lp, lref)
            kls.append(kl)
            rewards.append(r)
            used = used.clone()
            used[torch.arange(len(x), device=device), a] = True
            x, g = append(x, g, a)
        loss, returns = reinforce_loss(
            logas, rewards, kls, state_baselines, ema, cfg["beta"], cfg["gamma"]
        )
        opt.zero_grad()
        loss.backward()
        assert all(p.grad is None for p in ref.parameters())
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for t in range(H):
                ema[t] = 0.95 * ema[t] + 0.05 * (returns[t] - state_baselines[t]).mean()
        sync(device)
        compute += time.perf_counter() - t0
        gradnorms.append(float(norm.detach().cpu()))
        check = (seconds is None and (step in [max(1, steps // 2), steps])) or (
            seconds is not None
            and ((compute >= seconds / 2 and not half_checked) or compute >= seconds)
        )
        if seconds is not None and compute >= seconds / 2:
            half_checked = True
        if check:
            summary, _, _ = evaluate(
                "rl", model, ref, data["val"], rm, items, cfg, device, quick=True
            )
            logs.append(
                dict(
                    step=step,
                    training_reward=float(torch.stack(rewards).mean()),
                    **summary,
                )
            )
            print(
                "RL",
                seed,
                H,
                modified,
                step,
                summary["ndcg20"],
                summary["reward"],
                flush=True,
            )
            if key(summary, base, criterion) > best:
                best = key(summary, base, criterion)
                save(out, model, step=step)
    pd.DataFrame(logs).to_csv(str(out).replace(".pt", "_log.csv"), index=False)
    dump(
        str(out).replace(".pt", "_timing.json"),
        dict(
            steps=step,
            generated_actions=step * H * cfg["batch"],
            H=H,
            modified=modified,
            compute_seconds=compute,
            total_seconds=time.monotonic() - start,
            wall_budget=seconds,
            selected_includes_step0=True,
            gradient_norm_mean=float(np.mean(gradnorms)),
            clipping_fraction=float(np.mean(np.asarray(gradnorms) > 1)),
        ),
    )
    return load(out, device), compute
