import sys
from pathlib import Path
import numpy as np, torch, pytest

B = Path(__file__).resolve().parents[1]
from models2 import HybridGPT, append, logp, seed_all, categorical_kl
from data2 import pack, lookup
from train import ranking
from reward3 import Reward, predict


def test_strict_group_history_boundary():
    groups = [np.array([8, 9]), np.array([11]), np.array([15, 14]), np.array([77])]
    t = np.array([10, 15, 20, 25])
    raw, g, ht, _ = pack(groups, t, 2, 8)
    assert set(raw[raw >= 0]) == {8, 9, 11}
    assert (ht[ht >= 0] < 20).all()
    assert not np.isin(raw, [14, 15, 77]).any()


def test_tie_permutation_and_padding():
    seed_all(2)
    m = HybridGPT(20, 8, 16, 2, 4).eval()
    x = torch.tensor([[3, 4, 5, 1, 0, 0, 0, 0]])
    g = torch.tensor([[1, 1, 2, 3, 0, 0, 0, 0]])
    with torch.no_grad():
        y = m(x, g)
        z = m(x[:, [1, 0, 2, 3, 4, 5, 6, 7]], g)
        assert torch.allclose(y, z, atol=1e-6)
        # Padding table cannot influence valid query through attention.
        m.item.weight[0].fill_(1000)
        assert torch.allclose(y, m(x, g), atol=1e-6)


def test_synthetic_append_keeps_whole_groups():
    x = torch.tensor([[3, 4, 5, 6, 1]])
    g = torch.tensor([[1, 1, 2, 3, 4]])
    z, h = append(x, g, torch.tensor([7]))
    assert z.tolist() == [[5, 6, 10, 1, 0]]
    assert h.tolist() == [[1, 2, 3, 4, 0]]


def test_mapping_and_rank_metrics():
    ids = np.array([80, 4, 912])
    raw = np.array([[-1, -2, -3, 4, 912, 80, 7]])
    assert lookup(ids, raw).tolist() == [[0, 1, 2, 4, 5, 3, 2]]

    class Model(torch.nn.Module):
        def forward(self, x, g):
            return torch.arange(25, dtype=torch.float32).expand(len(x), -1)

    d = dict(
        x=np.ones((2, 2), int),
        g=np.ones((2, 2), int),
        target=np.array([27, 2]),
        uid=np.array([1, 2]),
    )
    met, f = ranking(Model(), d, "cpu")
    assert met["ndcg20"] == 0.5
    assert met["recall20"] == 0.5


def test_hybrid_sampling_mask_logprobs_and_gradient():
    seed_all(7)
    m = HybridGPT(23, 8, 16, 1, 4, alpha=0.75)
    x = torch.tensor([[3, 3, 4, 1, 0, 0, 0, 0]])
    g = torch.tensor([[1, 2, 3, 4, 0, 0, 0, 0]])
    used = torch.zeros((1, 23), dtype=torch.bool)
    used[:, 5] = True
    lp = logp(m, x, g, used)
    dist = torch.distributions.Categorical(logits=lp)
    a = dist.sample((1000,))
    assert not (a == 5).any()
    assert torch.allclose(dist.log_prob(a), lp[0, a[:, 0]].reshape(-1, 1), atol=1e-6)
    assert lp.shape[-1] == 23
    loss = -lp[0, 0]
    loss.backward()
    assert m.item.weight.grad.abs().sum() > 0


def test_tiny_copy_task_learns():
    seed_all(11)
    torch.set_num_threads(2)
    m = HybridGPT(12, 4, 16, 1, 4)
    x = torch.tensor([[3, 3, 1, 0], [4, 4, 1, 0], [5, 5, 1, 0], [6, 6, 1, 0]])
    g = torch.tensor([[1, 2, 3, 0]] * 4)
    target = torch.arange(4)
    opt = torch.optim.AdamW(m.parameters(), lr=0.01)
    with torch.no_grad():
        before = torch.nn.functional.cross_entropy(m(x, g), target).item()
    for _ in range(100):
        loss = torch.nn.functional.cross_entropy(m(x, g), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        assert torch.nn.functional.cross_entropy(m(x, g), target) < before * 0.05
        assert torch.equal(m(x, g).argmax(-1), target)


def test_exact_h1_is_upper_bound():
    torch.manual_seed(4)
    ref = torch.randn(8, 13).log_softmax(-1)
    p = torch.randn(8, 13).log_softmax(-1)
    r = torch.rand(8, 13)
    beta = 0.1
    q = (ref + r / beta).log_softmax(-1)
    j = lambda lp: (lp.exp() * r).sum(1) - beta * categorical_kl(lp, ref)
    assert (j(q) >= j(p) - 1e-6).all()


def test_reward_action_full_and_numpy_prediction_match():
    # Independent NumPy prediction vs batched Torch action/full-catalog paths.
    rng = np.random.default_rng(100)
    ids = np.array([10, 20, 30])
    asset = dict(
        ids=ids,
        order=np.arange(3),
        candidate=rng.normal(size=(4, 4)),
        correction=np.array([-0.1, 0, 0.1, 0]),
    )
    x = torch.tensor([[3, 3, 4, 1, 0], [5, 4, 1, 0, 0]])
    fb = torch.tensor([[0.7, 0.1, 0.2, 0.5], [0.3, 0.5, 0.2, 0.9]])
    a = torch.tensor([1, 2])
    dd = dict(
        item=ids[a.numpy()],
        fb=fb.numpy(),
        repeat=((x == a[:, None] + 3).sum(1).numpy() / 4),
    )
    for kind, n in [
        ("mean", 1),
        ("history", 7),
        ("candidate", 5),
        ("joint", 37),
        ("residual", 37),
    ]:
        spec = dict(
            kind=kind, coef=rng.normal(scale=0.1, size=n), cal_a=0.95, cal_b=0.02
        )
        r = Reward(asset, spec, ids, "cpu")
        act = r.actions(x, fb, a)
        full = r.all(x, fb).gather(1, a[:, None])[:, 0]
        assert torch.allclose(act, full, atol=2e-6)
        assert np.allclose(act.numpy(), predict(dd, asset, spec), atol=2e-6)


def test_new_reward_loo_removes_own_label_from_item_bin_and_global():
    from reward3 import cfeatures

    asset = dict(
        ids=np.array([10, 20]),
        order=np.array([0, 1]),
        mu=0.5,
        count=np.array([3.0, 5.0, 2.0]),
        sum=np.array([2.0, 2.0, 1.0]),
        bin=np.array([0, 1, 0]),
        bin_count=np.array([5.0, 5.0]),
        bin_sum=np.array([3.0, 2.0]),
        candidate=np.zeros((3, 4)),
    )
    d = dict(item=np.array([10]), w=np.array([1.0]), y=np.array([0.2]))
    before = cfeatures(d, asset, loo=True)
    import copy

    other = copy.deepcopy(asset)
    other["sum"][0] += 0.6
    other["bin_sum"][0] += 0.6
    other["mu"] += 0.06
    d["y"][0] += 0.6
    assert np.allclose(before, cfeatures(d, other, loo=True), atol=1e-12)


def test_feedback_is_invariant_when_window_cuts_a_tied_group():
    from data2 import feedback

    t = np.r_[np.arange(100) * 5, np.full(8, 501), np.arange(108, 302) * 5].astype(
        "int64"
    )
    y = np.linspace(0, 1, len(t))
    duration = np.full(len(t), 100)
    it = np.arange(len(t))
    perm = np.arange(len(t))
    perm[100:108] = perm[100:108][::-1]
    a, m = feedback(t, it, y, duration, 10000)
    b, n = feedback(t[perm], it[perm], y[perm], duration[perm], 10000)
    assert np.allclose(a, b)
    assert m == n


def test_history_only_ignores_candidate_even_when_candidate_changes():
    from reward3 import predict

    asset = dict(
        ids=np.array([10, 20]),
        order=np.array([0, 1]),
        candidate=np.array([[0, 1, 0.2, 0.1], [3, 4, -0.1, -0.2], [0, 0, 0, 0]]),
    )
    d = dict(
        item=np.array([10, 20]),
        fb=np.array([[0.7, 0.1, 0.2, 0.5], [0.7, 0.1, 0.2, 0.5]]),
        repeat=np.array([0.5, 0.0]),
    )
    spec = dict(kind="history", coef=np.array([0.1, 0.4, -0.1, 0.1, 0.1, 0.1, 0.1]))
    p = predict(d, asset, spec)
    assert p[0] == p[1]
