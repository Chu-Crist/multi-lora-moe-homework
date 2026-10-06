"""Shared constants and numpy-only evaluation helpers for Person 4 (MoE / router).

No torch needed here, so these functions can be tested without a GPU.
"""
import numpy as np
from sklearn.metrics import f1_score

# The three experts are Person 3's adapters. Their heads have different sizes,
# so every expert's output is placed in its own slice of one 12-class space:
#   AG News -> classes 0-3, SST-2 -> 4-5, TREC (coarse) -> 6-11
TASK_ORDER = ["agnews", "sst2", "trec"]
TASKS = {
    "agnews": dict(path="fancyzhx/ag_news", text="text",     label="label",        n=4, test="test"),
    "sst2":   dict(path="stanfordnlp/sst2", text="sentence", label="label",        n=2, test="validation"),
    "trec":   dict(path="CogComp/trec",     text="text",     label="coarse_label", n=6, test="test",
                   rev="refs/convert/parquet"),
}
OFFSET = {}
_o = 0
for _t in TASK_ORDER:
    OFFSET[_t] = _o
    _o += TASKS[_t]["n"]
NUM_CLASSES = _o  # 12


def predict_variants(g, P, task):
    """Predictions for every way of combining the experts.

    g:    (N, 3)     router weights (softmax output)
    P:    (N, 3, 12) each expert's class probabilities, padded into the 12-class space
    task: (N,)       true task id (used only by the oracle)
    """
    idx = np.arange(len(P))
    return {
        "oracle_task_id":  P[idx, task].argmax(1),            # upper bound: true task label is given
        "uniform_average": P.mean(1).argmax(1),               # multi-LoRA with no router
        "moe_soft":        (g[:, :, None] * P).sum(1).argmax(1),  # y = sum_i g_i * E_i(x)
        "moe_top1":        P[idx, g.argmax(1)].argmax(1),     # hard routing: only the best expert
    }


def metrics(pred, y, task):
    """Overall accuracy, plus accuracy and macro F1 per task (all in the 12-class space)."""
    out = {"overall_acc": float((pred == y).mean())}
    accs = []
    for i, t in enumerate(TASK_ORDER):
        m = task == i
        acc = float((pred[m] == y[m]).mean())
        labels = list(range(OFFSET[t], OFFSET[t] + TASKS[t]["n"]))
        f1 = float(f1_score(y[m], pred[m], labels=labels, average="macro", zero_division=0))
        out[t] = {"acc": acc, "macro_f1": f1}
        accs.append(acc)
    out["mean_task_acc"] = float(np.mean(accs))  # not dominated by AG News (7,600 of 8,972 test examples)
    return out


def routing_summary(g, task):
    """How often the router picks the right expert, and the average gate weights per true task."""
    pick = g.argmax(1)
    summary = {"overall_routing_acc": float((pick == task).mean())}
    for i, t in enumerate(TASK_ORDER):
        m = task == i
        summary[t] = {
            "routing_acc": float((pick[m] == i).mean()),
            "mean_gate": {TASK_ORDER[j]: float(g[m][:, j].mean()) for j in range(len(TASK_ORDER))},
        }
    return summary
