# Person 2 — Baseline Transformer

This folder contains the standard Transformer baseline experiment for the Multi-LoRA project.

## Model

- Model: DistilBERT (`distilbert-base-uncased`)
- Dataset: AG News
- Task: 4-class text classification
- Training samples: 120,000
- Test samples: 7,600
- Epochs: 1
- Batch size: 4
- Maximum sequence length: 128
- Learning rate: 2e-5
- FP16: Enabled

## Results

| Metric | Result |
| Accuracy | 94.45% |
| Macro F1 | 94.45% |
| Training loss | 0.2964 |
| Evaluation loss | 0.2365 |
| Training time | 3178.02 seconds |

This baseline will be used for comparison with LoRA, Multi-LoRA, and Multi-LoRA + MoE approaches.
