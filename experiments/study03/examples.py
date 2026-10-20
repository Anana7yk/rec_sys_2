"""Posthoc illustrations; never select/tune models from these rows."""

import json
from pathlib import Path
import numpy as np, pandas as pd

B = Path(__file__).resolve().parent
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, default=B / "runs/study_03")
parser.add_argument("--reports", type=Path, default=B.parents[1] / "validation/recomputed/study03")
args = parser.parse_args()
root = args.out
out = args.reports
out.mkdir(parents=True, exist_ok=True)
d = dict(np.load(root / "final/policy_data/holdout_raw.npz"))
ids = np.load(root / "catalog_ids.npy")
base = np.load(root / "final/11_base_top20.npy")
rl = np.load(root / "final/11_rl_h1_raw_top20.npy")
q = np.load(root / "final/11_exact_top20.npy")
a = pd.read_csv(root / "final/11_base_events.csv")
b = pd.read_csv(root / "final/11_rl_h1_raw_events.csv")
chosen = []
for label, mask in [
    ("repeated_target", a.target_seen_history == 1),
    ("novel_known_target", (a.target_seen_history == 0) & (a.in_catalog == 1)),
]:
    ix = np.flatnonzero(mask.to_numpy())
    changed = [i for i in ix if not np.array_equal(base[i], rl[i])]
    chosen.append((label, int((changed or ix.tolist())[0])))
chosen.append(("largest_proxy_increase", int((b.reward - a.reward).idxmax())))
rows = []
text = [
    "# Примеры до/после RL, seed11",
    "Выбор иллюстративный: первый изменившийся список для повторной/новой known цели и максимум прироста primary proxy. Третий пример экстремальный, не типичный. Выбор не меняет модели и основные выводы. Все ID анонимны; нельзя приписывать им жанры или названия.\n",
]
for label, i in chosen:
    history = [
        dict(time=int(t), items=d["raw"][i][d["ht"][i] == t].tolist())
        for t in np.unique(d["ht"][i][d["ht"][i] >= 0])[-3:]
    ]
    row = dict(
        rule=label,
        uid=int(d["uid"][i]),
        t=int(d["t"][i]),
        historical_target=int(d["item"][i]),
        observed_target_response=float(d["y"][i]),
        recent_timestamp_groups=history,
        base_top5=ids[base[i, :5]].tolist(),
        rl_top5=ids[rl[i, :5]].tolist(),
        analytic_top5=ids[q[i, :5]].tolist(),
        top20_overlap=len(set(base[i]) & set(rl[i])) / 20,
        base={
            k: float(a.iloc[i][k])
            for k in [
                "ndcg20",
                "reward",
                "evaluator_reward",
                "duration",
                "history_repeat",
                "top_decile",
            ]
        },
        rl={
            k: float(b.iloc[i][k])
            for k in [
                "ndcg20",
                "reward",
                "evaluator_reward",
                "duration",
                "history_repeat",
                "top_decile",
            ]
        },
    )
    rows.append(row)
    text += [
        "## " + label,
        f"User{row['uid']}, target{row['historical_target']}, t={row['t']}. Наблюдённый отклик относится только к этому историческому target: {row['observed_target_response']:.2f}.",
        f"Последние3 временные группы: `{history}`",
        f"До: `{row['base_top5']}`; RL: `{row['rl_top5']}`; q: `{row['analytic_top5']}`.",
        f"Совпадение Top20: {row['top20_overlap']:.0%}; primary reward {row['base']['reward']:.4f}→{row['rl']['reward']:.4f}; evaluator {row['base']['evaluator_reward']:.4f}→{row['rl']['evaluator_reward']:.4f}; duration {row['base']['duration']:.1f}→{row['rl']['duration']:.1f}s; NDCG20 {row['base']['ndcg20']:.4f}→{row['rl']['ndcg20']:.4f}.\n",
    ]
(out / "examples.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
)
(out / "EXAMPLES.md").write_text("\n\n".join(text) + "\n")
