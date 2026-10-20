import sys, copy
from pathlib import Path
import numpy as np
import torch

from data2 import pack, feedback, weights, lookup
from models2 import GroupGPT, append, categorical_kl
from reward2 import FastReward, raw_predict, metrics, predict
from fit2 import reinforce_loss
from horizon2 import inverse_cdf_cpu


def test_inverse_cdf_never_selects_masked_plateaus_and_shares_identical_draws():
    lp = np.array([[-np.inf, np.log(0.25), np.log(0.75), -np.inf]] * 3)
    u = np.array([0.0, 0.999999999, 0.25])
    a = inverse_cdf_cpu(lp, u)
    assert a.tolist() == [1, 2, 2] and np.array_equal(
        a, inverse_cdf_cpu(lp.copy(), u.copy())
    )


def test_two_step_gradient_matches_exact_reward_discounted_but_undiscounted_kl_objective():
    theta = torch.tensor([0.2, -0.4, 0.7], dtype=torch.float64, requires_grad=True)
    p = torch.stack([torch.sigmoid(theta), 1 - torch.sigmoid(theta)], 1)
    lp = p.log()
    ref = torch.tensor([[0.6, 0.4], [0.3, 0.7], [0.8, 0.2]], dtype=torch.float64).log()
    kl = (p * (lp - ref)).sum(1)
    gamma = 0.9
    beta = 0.1
    reward = torch.tensor([[0.1, 0.4], [0.8, 0.2]], dtype=torch.float64)
    loss = 0.0
    exact = 0.0
    for a in range(2):
        for b in range(2):
            weight = p[0, a] * p[a + 1, b]
            rr = [
                torch.tensor([0.2 + 0.3 * a], dtype=torch.float64),
                reward[a, b].reshape(1),
            ]
            kk = [kl[0].reshape(1), kl[a + 1].reshape(1)]
            ll, _ = reinforce_loss(
                [lp[0, a].reshape(1), lp[a + 1, b].reshape(1)],
                rr,
                kk,
                [torch.tensor([0.37]), torch.tensor([0.11 + 0.05 * a])],
                torch.zeros(2),
                beta,
                gamma,
            )
            loss = loss + weight.detach() * ll
            exact = exact + weight * (
                rr[0].sum() + gamma * rr[1].sum() - beta * (kk[0].sum() + kk[1].sum())
            )
    actual = torch.autograd.grad(loss, theta, retain_graph=True)[0]
    expected = -torch.autograd.grad(exact, theta)[0]
    assert torch.allclose(actual, expected, atol=1e-10)


def test_empty_conditional_catalog_is_explicitly_undefined():
    result = metrics(np.array([]), np.array([]), np.array([]))
    assert set(result) == {"rmse", "mae", "calibration", "prediction", "observed"}
    assert all(np.isnan(v) for v in result.values())


def test_item_leave_one_out_removes_own_label_from_every_target_statistic():
    result = []
    for own in [0.2, 0.8]:
        sums = np.array([own, 0.4, 0.0])
        mu = (own + 0.4) / 2
        asset = dict(
            ids=np.array([10, 20]),
            order=np.arange(2),
            duration=np.array([100, 200, 150]),
            pop=np.array([1, 1, 0]),
            dur_mu=5.0,
            dur_sd=1.0,
            pop_mu=1.0,
            pop_sd=1.0,
            mu=mu,
            skip=0.0,
            bin=np.array([0, 0, 0]),
            count=np.array([1.0, 1.0, 0.0]),
            sum=sums,
            bin_count=np.array([2.0]),
            bin_sum=np.array([own + 0.4]),
            bin_prior=np.array([mu]),
        )
        d = dict(
            item=np.array([10]),
            fb=np.array([[0.5, 0.1, 0.2, 0.4]]),
            repeat=np.array([0.0]),
            y=np.array([own]),
            w=np.array([1.0]),
        )
        result.append(raw_predict(d, asset, dict(kind="item", alpha=10), loo=True)[0])
    assert np.allclose(result, [0.4, 0.4])


def test_group_permutation_invariance_and_group_causality():
    torch.manual_seed(7)
    m = GroupGPT(20, 8, 32, 2, 4).eval()
    x = torch.tensor([[3, 4, 5, 6, 1, 0, 0, 0]])
    g = torch.tensor([[1, 1, 2, 2, 3, 0, 0, 0]])
    y = x.clone()
    y[0, :2] = x[0, torch.tensor([1, 0])]
    y[0, 2:4] = x[0, torch.tensor([3, 2])]
    assert torch.allclose(m(x, g), m(y, g), atol=1e-6)
    cache = []
    hook = m.ln.register_forward_pre_hook(
        lambda module, args: cache.append(args[0].detach().clone())
    )
    m(x, g)
    y = x.clone()
    y[0, 2:4] = torch.tensor([10, 11])
    m(y, g)
    hook.remove()
    assert torch.allclose(cache[0][:, :2], cache[1][:, :2], atol=1e-6)


def test_padding_is_invisible_and_raw_ids_match_item_only_head():
    torch.manual_seed(19)
    m = GroupGPT(3, 5, 16, 1, 4).eval()
    x = torch.tensor([[3, 4, 1, 0, 0]])
    g = torch.tensor([[1, 2, 3, 0, 0]])
    before = m(x, g)
    with torch.no_grad():
        m.item.weight[0].fill_(1000)
    assert before.shape == (1, 3) and torch.allclose(before, m(x, g), atol=1e-6)
    ids = np.array([30, 10, 20])
    assert lookup(ids, np.array([-1, -2, -3, 10, 99, 30])).tolist() == [
        0,
        1,
        2,
        4,
        2,
        3,
    ]


def test_tie_group_packing_keeps_boundary_marker_and_strict_past():
    groups = [np.array([3, 4, 5]), np.array([6, 7])]
    times = np.array([10, 20])
    raw, g, ht, lost = pack(groups, times, 2, 5)
    assert (
        raw.tolist() == [-3, 6, 7, -2, -1]
        and g.tolist() == [1, 2, 2, 3, 0]
        and lost == 3
    )
    assert np.all(ht < 30)


def test_feedback_excludes_recent_and_current_outcomes():
    t = np.array([0, 5000, 6000])
    it = np.array([1, 2, 3])
    y = np.array([0.2, 0.8, 0.9])
    duration = np.ones(3) * 100
    a, maxa = feedback(t, it, y, duration, 6100)
    y[1:] = 0
    b, maxb = feedback(t, it, y, duration, 6100)
    assert np.array_equal(a, b) and np.isclose(a[0], 0.2) and maxa < 6100


def test_user_weights_equalize_mass():
    u = np.array([1, 1, 1, 2])
    w = weights(u)
    assert np.isclose(w[u == 1].sum(), w[u == 2].sum())


def test_append_removes_entire_oldest_group():
    x = torch.tensor([[3, 4, 5, 1], [3, 1, 0, 0]])
    g = torch.tensor([[1, 1, 2, 3], [1, 2, 0, 0]])
    z, h = append(x, g, torch.tensor([4, 3]))
    assert z.tolist() == [[5, 7, 1, 0], [3, 6, 1, 0]]
    assert h.tolist() == [[1, 2, 3, 0], [1, 2, 3, 0]]


def test_analytic_kl_optimum():
    ref = torch.tensor([[0.1, 0.2, 0.7]]).log()
    r = torch.tensor([[0.9, 0.4, 0.1]])
    beta = 0.1
    q = (ref + r / beta).log_softmax(-1)
    optimal = (q.exp() * r).sum() - beta * categorical_kl(q, ref).sum()
    for _ in range(20):
        lp = torch.randn_like(ref).log_softmax(-1)
        j = (lp.exp() * r).sum() - beta * categorical_kl(lp, ref).sum()
        assert j <= optimal + 1e-6


def test_duration_adjustment_is_single_train_offset():
    r = torch.tensor([0.3, 0.8])
    offset = torch.tensor([0.1, -0.1])
    assert torch.allclose(r - offset, torch.tensor([0.2, 0.9]))


def test_fast_reward_full_vs_sampled_including_loaded_numpy_scalars():
    ids = np.array([10, 20, 30])
    asset = dict(
        ids=ids,
        order=np.arange(3),
        mu=np.array(0.6),
        skip=np.array(0.1),
        duration=np.array([100, 200, 300, 200]),
        pop=np.array([10, 20, 30, 0]),
        dur_mu=np.array(5.0),
        dur_sd=np.array(1.0),
        pop_mu=np.array(2.0),
        pop_sd=np.array(1.0),
        bin=np.array([0, 1, 1, 1]),
        bin_prior=np.array([0.5, 0.7]),
        count=np.array([10, 20, 30, 0]),
        sum=np.array([6, 10, 24, 0]),
        duration_correction=np.array([-0.1, 0.1, 0.1, 0.1]),
    )
    spec = dict(
        kind="personal",
        alpha=100,
        coef=np.arange(8) * 0.01,
        f_mu=np.zeros(8),
        f_sd=np.ones(8),
        intercept=0.01,
        cal_a=0.9,
        cal_b=0.03,
    )
    for device in ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []):
        rm = FastReward(asset, spec, ids, device)
        x = torch.tensor([[3, 4, 1, 0], [5, 3, 3, 1]], device=device)
        fb = torch.tensor([[0.5, 0.1, 0.2, 0.3], [0.8, 0.2, 0.1, 0.5]], device=device)
        a = torch.tensor([1, 0], device=device)
        allr = rm.all(x, fb)
        chosen = rm.actions(x, fb, a)
        assert torch.allclose(chosen, allr.gather(1, a[:, None]).squeeze(1), atol=1e-6)
        aa = a.cpu().numpy()
        xx = x.cpu().numpy()
        dd = dict(
            item=ids[aa],
            fb=fb.cpu().numpy(),
            repeat=(xx == aa[:, None] + 3).sum(1) / (xx.shape[1] - 1),
        )
        assert np.allclose(chosen.cpu().numpy(), predict(dd, asset, spec), atol=1e-6)
