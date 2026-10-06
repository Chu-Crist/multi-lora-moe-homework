"""Person 4: MoE router over Person 3's three LoRA experts.

Pipeline:   input -> router -> (LoRA expert 1 | 2 | 3) -> combination -> prediction
    g = softmax(W_r x)                      x = frozen DistilBERT [CLS] vector (768)
    y = sum_i g_i * E_i(x)                  E_i = expert i's class probabilities, padded to 12 classes

Experts and the base model stay frozen. Only the router (768 -> 3, 2,307 parameters) is trained.
Because the experts are frozen, their outputs are computed once and cached, so training the router
takes seconds.

Run (GPU recommended for the caching step):
    python moe_router.py
    python moe_router.py --reuse-cache      # skip the model passes, retrain/evaluate the router only
"""
import argparse, json, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import (AutoModel, AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding)

from moe_common import (TASK_ORDER, TASKS, OFFSET, NUM_CLASSES,
                        predict_variants, metrics, routing_summary)

MODEL = "distilbert-base-uncased"
HERE = Path(__file__).resolve().parent
ADAPTER_DIR = HERE.parent / "person3" / "adapters"
P3_RESULTS = HERE.parent / "person3" / "results"
OUT_DIR = HERE / "results"
CACHE_DIR = HERE / "cache"
BATCH = 128


# ----------------------------------------------------------------------------- data
def load_split(name, split, tokenizer, max_n=None, seed=0):
    cfg = TASKS[name]
    ds = load_dataset(cfg["path"], revision=cfg.get("rev"))[split]
    if max_n is not None and len(ds) > max_n:
        ds = ds.shuffle(seed=seed).select(range(max_n))
    ds = ds.rename_column(cfg["label"], "labels")
    ds = ds.map(lambda x: tokenizer(x[cfg["text"]], truncation=True, max_length=128), batched=True)
    keep = {"input_ids", "attention_mask", "labels"}
    return ds.remove_columns([c for c in ds.column_names if c not in keep])


# ----------------------------------------------------------------------------- models
def load_expert(name, device):
    # Person 3's adapters include their own classifier heads (4 / 2 / 6 outputs), so each
    # expert is its own model instance. The "newly initialized weights" warning is expected:
    # the adapter's saved pre_classifier / classifier weights replace them.
    base = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=TASKS[name]["n"])
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR / name))
    return model.to(device).eval()


@torch.no_grad()
def cache_split(ds, name, experts, encoder, tokenizer, device):
    """For every example: [CLS] vector, the 3 experts' padded probabilities, and the label."""
    loader = DataLoader(ds, batch_size=BATCH, collate_fn=DataCollatorWithPadding(tokenizer))
    cls_all, P_all, y_all = [], [], []
    use_amp = device.type == "cuda"
    for batch in loader:
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            cls = encoder(**batch).last_hidden_state[:, 0].float()
            per_expert = []
            for t in TASK_ORDER:
                probs = torch.softmax(experts[t](**batch).logits.float(), dim=-1)
                padded = torch.zeros(probs.size(0), NUM_CLASSES, device=device)
                padded[:, OFFSET[t]:OFFSET[t] + TASKS[t]["n"]] = probs
                per_expert.append(padded)
        cls_all.append(cls.cpu())
        P_all.append(torch.stack(per_expert, dim=1).cpu())          # (B, 3, 12)
        y_all.append(labels + OFFSET[name])                          # label in the 12-class space
    y = torch.cat(y_all)
    return dict(cls=torch.cat(cls_all), P=torch.cat(P_all), y=y,
                task=torch.full_like(y, TASK_ORDER.index(name)))


def build_cache(kind, n_per_task, tokenizer, experts, encoder, device, seed):
    parts = []
    for t in TASK_ORDER:
        split = "train" if kind == "train" else TASKS[t]["test"]
        ds = load_split(t, split, tokenizer, max_n=n_per_task if kind == "train" else None, seed=seed)
        print(f"  caching {kind}/{t}: {len(ds)} examples")
        parts.append(cache_split(ds, t, experts, encoder, tokenizer, device))
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}


# ----------------------------------------------------------------------------- router
class Router(nn.Module):
    """g = softmax(W_r x), with the input standardized using training-set statistics."""
    def __init__(self, dim, n_experts, mean, std):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.linear = nn.Linear(dim, n_experts)

    def forward(self, x):
        return torch.softmax(self.linear((x - self.mean) / self.std), dim=-1)


def mixture(g, P):
    """y = sum_i g_i * E_i(x)   (g: (B,3), P: (B,3,12)) -> (B,12) class probabilities."""
    return torch.einsum("be,bec->bc", g, P)


def train_router(cls, P, y, task, epochs, lr, seed, device):
    n = cls.size(0)
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    n_val = n // 10
    val_idx, tr_idx = perm[:n_val].to(device), perm[n_val:].to(device)
    cls, P, y, task = cls.to(device), P.to(device), y.to(device), task.to(device)

    mean = cls[tr_idx].mean(0)
    std = cls[tr_idx].std(0).clamp_min(1e-6)
    router = Router(cls.size(1), len(TASK_ORDER), mean, std).to(device)
    opt = torch.optim.Adam(router.parameters(), lr=lr)
    n_params = sum(p.numel() for p in router.parameters() if p.requires_grad)
    print(f"Router trainable parameters: {n_params:,}")

    for ep in range(1, epochs + 1):
        router.train()
        order = tr_idx[torch.randperm(len(tr_idx), device=device)]
        total = 0.0
        for i in range(0, len(order), 256):
            b = order[i:i + 256]
            y_mix = mixture(router(cls[b]), P[b])
            loss = -torch.log(y_mix.gather(1, y[b][:, None]).squeeze(1).clamp_min(1e-8)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        if ep % 5 == 0 or ep == 1:
            router.eval()
            with torch.no_grad():
                val_acc = (router(cls[val_idx]).argmax(1) == task[val_idx]).float().mean().item()
            print(f"  epoch {ep:3d}  train loss {total / len(order):.4f}  held-out routing acc {val_acc:.4f}")
    return router.eval(), n_params


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-per-task", type=int, default=2000, help="router training examples per task")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reuse-cache", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    OUT_DIR.mkdir(exist_ok=True); CACHE_DIR.mkdir(exist_ok=True)

    train_f, test_f = CACHE_DIR / "train.pt", CACHE_DIR / "test.pt"
    t0 = time.time()
    if args.reuse_cache and train_f.exists() and test_f.exists():
        train, test = torch.load(train_f), torch.load(test_f)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        encoder = AutoModel.from_pretrained(MODEL).to(device).eval()      # router features: no adapters
        experts = {t: load_expert(t, device) for t in TASK_ORDER}
        train = build_cache("train", args.train_per_task, tokenizer, experts, encoder, device, args.seed)
        test = build_cache("test", None, tokenizer, experts, encoder, device, args.seed)
        torch.save(train, train_f); torch.save(test, test_f)
    cache_time = time.time() - t0

    t0 = time.time()
    router, n_params = train_router(train["cls"], train["P"], train["y"], train["task"],
                                    args.epochs, args.lr, args.seed, device)
    router_time = time.time() - t0

    with torch.no_grad():
        g = router(test["cls"].to(device)).cpu().numpy()
    P, y, task = test["P"].numpy(), test["y"].numpy(), test["task"].numpy()

    results = {name: metrics(pred, y, task) for name, pred in predict_variants(g, P, task).items()}
    routing = routing_summary(g, task)

    # Sanity check: the oracle numbers must match Person 3's results, otherwise the adapters were loaded wrongly.
    print("\nSanity check against Person 3 (oracle task ID should equal their accuracy):")
    for t in TASK_ORDER:
        f = P3_RESULTS / f"{t}.json"
        ref = json.load(open(f))["accuracy"] if f.exists() else float("nan")
        got = results["oracle_task_id"][t]["acc"]
        flag = "" if abs(got - ref) < 0.01 else "   <-- MISMATCH, check adapter loading"
        print(f"  {t:7s} ours {got:.4f}  Person 3 {ref:.4f}{flag}")

    print(f"\nTest examples: {len(y)}  (AG News {int((task==0).sum())}, SST-2 {int((task==1).sum())}, TREC {int((task==2).sum())})")
    print(f"{'method':18s} {'overall':>8s} {'AG News':>8s} {'SST-2':>8s} {'TREC':>8s} {'mean/task':>10s}")
    for name, r in results.items():
        print(f"{name:18s} {r['overall_acc']:8.4f} {r['agnews']['acc']:8.4f} {r['sst2']['acc']:8.4f} "
              f"{r['trec']['acc']:8.4f} {r['mean_task_acc']:10.4f}")
    print(f"\nRouting accuracy (router picks the right expert): {routing['overall_routing_acc']:.4f}")
    for t in TASK_ORDER:
        gate = ", ".join(f"{k} {v:.3f}" for k, v in routing[t]["mean_gate"].items())
        print(f"  true task {t:7s}: routing acc {routing[t]['routing_acc']:.4f} | mean gate: {gate}")

    out = dict(config=vars(args), router_trainable_params=n_params, test_size=int(len(y)),
               cache_time_s=cache_time, router_train_time_s=router_time,
               results=results, routing=routing)
    json.dump(out, open(OUT_DIR / "moe_results.json", "w"), indent=2)
    torch.save(router.state_dict(), HERE / "router.pt")
    print(f"\nSaved results/moe_results.json and router.pt")


if __name__ == "__main__":
    main()
