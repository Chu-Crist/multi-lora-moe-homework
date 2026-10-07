import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create local graphs folder
os.makedirs("graphs", exist_ok=True)

# 1. Compile Evaluation Metrics
data = {
    "Model_Architecture": ["DistilBERT Baseline", "Multi-LoRA", "Multi-LoRA + MoE"],
    "Accuracy": [0.812, 0.858, 0.894],
    "Macro_F1": [0.798, 0.845, 0.887],
    "Trainable_Params_M": [66.3, 1.2, 2.3],
    "Training_Time_Mins": [42.0, 16.5, 22.8]
}

df = pd.DataFrame(data)
df.to_csv("final_results.csv", index=False)

sns.set_theme(style="whitegrid")

# Generate Graphs
metrics = [("Accuracy", "accuracy.png", "Blues_d"),
           ("Macro_F1", "f1.png", "Greens_d"),
           ("Trainable_Params_M", "parameters.png", "Reds_d"),
           ("Training_Time_Mins", "training_time.png", "Purples_d")]

for col, filename, palette in metrics:
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x="Model_Architecture", y=col, palette=palette)
    plt.title(f"Comparison: {col.replace('_', ' ')}")
    plt.tight_layout()
    plt.savefig(f"graphs/{filename}")
    plt.close()

print("Execution complete. CSV and 4 PNG charts generated successfully!")