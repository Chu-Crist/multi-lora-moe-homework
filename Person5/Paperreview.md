# Person 5: Multi-LoRA Research Paper Review & Analysis
## 1. Overview of Evaluated Architectures
* **MultiLoRA:** Addresses multi-task parameter interference by horizontally splitting low-rank adaptation matrices across parameter subspaces.
* **MALoRA (Mixture of Asymmetric LoRA):** Implements dynamic asymmetric low-rank adapters paired with a Softmax top-k MoE gating network to reduce memory usage during multi-task adaptation.
* **MeteoRA:** Utilizes token-level dynamic routing across specialized LoRA adapter pools, enabling real-time intention and task switching during inference.
---
## 2. Comparative Summary Table

| Model | Core Adaptation Strategy | Router Gating Mechanism | Key Advantage |
| :--- | :--- | :--- | :--- |
| **MultiLoRA** | Subspace Rank Splitting | Deterministic Task Gating | Low task cross-talk |
| **MALoRA** | Asymmetric Matrix Decomposition | Softmax Top-k MoE | Low GPU Memory footprint |
| **MeteoRA** | Dynamic Token Adapter Pool | Token-level Load Balancer | SOTA Multi-classification accuracy |
