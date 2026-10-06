# Person 4: MoE / Router

A router that weights Person 3's three LoRA adapters (AG News, SST-2, TREC), so the model does not need to be told which task an input belongs to.

## How it works

    input -> router -> LoRA expert 1 / 2 / 3 -> combination -> prediction

- Base model: `distilbert-base-uncased`, frozen. Experts: Person 3's adapters (with their own classifier heads), frozen.
- Router: `g = softmax(W_r x)`, where `x` is the [CLS] vector of the frozen pretrained DistilBERT (768 values). `W_r` is a single linear layer (768 -> 3, 2,307 trainable parameters). Inputs are standardized with training-set statistics.
- Combination: `y = sum_i g_i * E_i(x)`. Each expert's class probabilities are placed in its own slice of one 12-class output (AG News 0-3, SST-2 4-5, TREC 6-11) before mixing. Probabilities are mixed instead of raw logits because the experts' logits are on different scales.
- Training: cross-entropy on the mixed output, router only. The experts are frozen, so their outputs are computed once and cached.
- Router training data: a random subsample of each training set (2,000 per task by default; the reported run used 500). Note that the experts were trained on these same sets.
- Evaluation: Person 3's splits (AG News test 7,600, SST-2 validation 872, TREC test 500; 8,972 in total).

## Methods compared

| Method | Meaning |
|---|---|
| `oracle_task_id` | Multi-LoRA when the true task is given. Upper bound; must match Person 3's numbers. |
| `uniform_average` | Multi-LoRA with no router (all experts weighted 1/3) |
| `moe_soft` | Multi-LoRA + MoE, soft router weights |
| `moe_top1` | Multi-LoRA + MoE, only the highest-weighted expert is used |

## Results

The router was trained using 500 examples per task.

Router trainable parameters: 2,307
Routing accuracy: 99.00%

| Method | Overall | AG News | SST-2 | TREC | Mean/task |
|---|---:|---:|---:|---:|---:|
| `oracle_task_id` | 92.05% | 92.79% | 88.65% | 86.80% | 89.41% |
| `uniform_average` | 80.87% | 83.29% | 82.22% | 41.80% | 69.10% |
| `moe_soft` | 91.31% | 92.09% | 87.16% | 86.60% | 88.62% |
| `moe_top1` | 91.22% | 92.01% | 86.93% | 86.60% | 88.51% |

The soft MoE improves overall accuracy from 80.87% with
uniform averaging to 91.31%, a gain of 10.44 percentage points.
It is 0.74 percentage points below the oracle-task-ID result.

Macro F1 (the metric Person 3 also reports):

| Method | AG News | SST-2 | TREC |
|---|---:|---:|---:|
| `oracle_task_id` | 92.79% | 88.63% | 73.69% |
| `uniform_average` | 89.14% | 86.80% | 48.14% |
| `moe_soft` | 92.47% | 88.08% | 73.58% |
| `moe_top1` | 92.43% | 87.91% | 73.58% |

Routing accuracy by task:
- AG News: 99.09%
- SST-2: 97.71%
- TREC: 99.80%

Sanity check against Person 3:
- AG News: 92.79% (matches Person 3)
- SST-2: 88.65% (matches Person 3)
- TREC: 86.80% (matches Person 3)

## Files

- `moe_router.py`: caches expert outputs, trains the router, evaluates
- `moe_common.py`: constants and the evaluation functions
- `results/moe_results.json`, `router.pt`: written by the script

## Reproduce

Requires `person3/adapters` next to this folder.

    pip install peft datasets transformers accelerate scikit-learn
    python moe_router.py --train-per-task 500   # settings of the reported run; GPU recommended for the caching step
    python moe_router.py --reuse-cache   # retrain/evaluate the router without rerunning the models

The "some weights were newly initialized" warnings are expected: the adapters' saved classifier weights replace them. The script prints a sanity check comparing the oracle numbers with Person 3's results.

## Limitations

- The experts were trained on the same training sets that the router is trained on.
- The mixed test set is dominated by AG News (7,600 of 8,972 examples), so per-task accuracies and the mean over tasks are also reported.
- Single run, single seed.
- Inference needs three expert passes (plus one for the router features), so it costs about 3x a single adapter.
- The three tasks are very different (news, movie reviews, questions), so routing may be easy. In this setup the oracle is the upper bound, and the MoE's advantage is that it needs no task label at inference.
