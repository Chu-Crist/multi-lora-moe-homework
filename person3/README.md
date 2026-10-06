# Person 3: Multi-LoRA

LoRA adapters on DistilBERT (`distilbert-base-uncased`), one adapter per task, trained with Hugging Face PEFT.

## Setup

- LoRA: rank r=8, alpha=16, dropout=0.1, applied to `q_lin` and `v_lin` in all 6 transformer layers
- Training: 1 epoch, learning rate 2e-4, batch size 32, dynamic padding, max length 128, weight decay 0.01, FP16
- Hardware: Google Colab, T4 GPU
- Single run per task, no repeated seeds

## Results

| Task | Classes | Eval split | Accuracy | Macro F1 | Trainable params | Training time |
|---|---|---|---|---|---|---|
| AG News | 4 | test (7,600) | 92.79% | 92.79% | 741,124 | 275.8 s |
| SST-2 | 2 | validation (872) | 88.65% | 88.63% | 739,586 | 67.7 s |
| TREC (coarse) | 6 | test (500) | 86.80% | 73.69% | 742,662 | 7.0 s |

Notes:
- SST-2 is evaluated on the validation split because the official test labels are hidden.
- TREC uses the `coarse_label` column (6 classes).
- TREC macro F1 (73.69%) is far below its accuracy (86.80%). This suggests weak performance on the rare classes. TREC is small (5,452 training samples) and was trained for only 1 epoch, so it is likely under-trained. We did not confirm the cause.

## Comparison with the Person 2 baseline (AG News)

| Model | Accuracy | Macro F1 | Trainable params | Training time |
|---|---|---|---|---|
| Baseline (full fine-tuning) | 94.45% | 94.45% | about 67M | 3178.0 s |
| LoRA (r=8) | 92.79% | 92.79% | 741,124 (1.09%) | 275.8 s |

LoRA reaches 1.66 points lower accuracy while training about 1.1% of the parameters.

Caveat: this is not a like-for-like comparison. The baseline used learning rate 2e-5, batch size 4 and fixed padding to 128 tokens. The LoRA runs used learning rate 2e-4, batch size 32 and dynamic padding. Most of the roughly 11x speedup comes from the larger batch size and dynamic padding, not from LoRA itself. Both results are from a single run, so a difference of about one point should not be over-interpreted.

## Trainable parameters

Of the roughly 741k trainable parameters per task, only 147,456 are LoRA matrices (6 layers x 2 modules x 12,288). The remaining roughly 594k are the classifier layers (`pre_classifier` and `classifier`), which are trained from scratch for each task. Reporting only the LoRA matrices would give a much smaller number.

## How LoRA works (W' = W + BA)

A pretrained weight matrix W (768 x 768 in DistilBERT) is kept frozen. LoRA learns two small matrices instead: B (768 x r) and A (r x 768), with r=8. Their product BA is a full 768 x 768 matrix of corrections, and the adapted weight is:

W' = W + (alpha / r) * BA

With alpha=16 and r=8 the scaling factor is 2. W has 589,824 numbers, but A and B together have only 12,288, so we train about 2% of the numbers for each adapted matrix while still being able to change the whole matrix. Only A and B are saved, which is why the adapter files are small.

## Note for Person 4 (MoE / router)

Each adapter includes its own classifier head, and the heads have different output sizes (4, 2 and 6 classes). The three adapters therefore cannot simply be switched on one shared model with one head. A router has to either handle the three heads separately or use a shared head design.

## Files

- `multi_lora.py`: trains and evaluates all three adapters
- `adapters/agnews`, `adapters/sst2`, `adapters/trec`: saved PEFT adapters (including classifier heads)
- `results/*.json`: metrics per task

## Reproduce

Requires an NVIDIA GPU (the script uses FP16).

    pip install peft datasets transformers accelerate scikit-learn
    python multi_lora.py

TREC is loaded from the Hub's parquet conversion (`refs/convert/parquet`) because the original dataset script is no longer supported by recent versions of `datasets`.
