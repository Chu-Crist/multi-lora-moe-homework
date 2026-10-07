from datasets import load_dataset

# Load SST-2
print("Loading SST-2 dataset...")
dataset = load_dataset("stanfordnlp/sst2")

# Dataset information
print("\nDataset information:")
print(dataset)

# Number of examples
print("\nNumber of examples:")
print("Training:", len(dataset["train"]))
print("Validation:", len(dataset["validation"]))
print("Test:", len(dataset["test"]))

# Dataset features
print("\nDataset features:")
print(dataset["train"].features)

# First 5 examples
print("\nFirst 5 training examples:")
for i in range(5):
    print(dataset["train"][i])

# Label distribution
print("\nSST-2 label distribution:")

train_labels = dataset["train"]["label"]

negative = train_labels.count(0)
positive = train_labels.count(1)

total = len(train_labels)

print("Negative (0):", negative)
print("Positive (1):", positive)

print("\nPercentages:")
print("Negative:", round(negative / total * 100, 2), "%")
print("Positive:", round(positive / total * 100, 2), "%")
# ==========================================
# AG NEWS DATASET ANALYSIS
# ==========================================

print("\n\nLoading AG News dataset...")

ag_news = load_dataset("fancyzhx/ag_news")

# Dataset information
print("\nAG News dataset information:")
print(ag_news)

# Number of examples
print("\nNumber of examples:")
print("Training:", len(ag_news["train"]))
print("Test:", len(ag_news["test"]))

# Dataset features
print("\nDataset features:")
print(ag_news["train"].features)

# First 5 examples
print("\nFirst 5 AG News training examples:")
for i in range(5):
    print(ag_news["train"][i])

# Class distribution
print("\nAG News class distribution:")

ag_labels = ag_news["train"]["label"]

class_0 = ag_labels.count(0)
class_1 = ag_labels.count(1)
class_2 = ag_labels.count(2)
class_3 = ag_labels.count(3)

total_ag = len(ag_labels)

print("Class 0:", class_0)
print("Class 1:", class_1)
print("Class 2:", class_2)
print("Class 3:", class_3)

print("\nPercentages:")
print("Class 0:", round(class_0 / total_ag * 100, 2), "%")
print("Class 1:", round(class_1 / total_ag * 100, 2), "%")
print("Class 2:", round(class_2 / total_ag * 100, 2), "%")
print("Class 3:", round(class_3 / total_ag * 100, 2), "%")
# AG News label names
print("\nAG News label names:")
print(ag_news["train"].features["label"].names)
# ==========================================
# TREC DATASET ANALYSIS
# ==========================================

print("\n\nLoading TREC dataset...")

trec = load_dataset("lukasgarbas/trec")

# Dataset information
print("\nTREC dataset information:")
print(trec)

# Number of examples
print("\nNumber of examples:")
print("Training:", len(trec["train"]))
print("Test:", len(trec["test"]))

# Dataset features
print("\nDataset features:")
print(trec["train"].features)

# First 5 examples
print("\nFirst 5 TREC training examples:")
for i in range(5):
    print(trec["train"][i])
    # TREC coarse label distribution
print("\nTREC coarse label distribution:")

trec_labels = trec["train"]["coarse_label"]

from collections import Counter

trec_counts = Counter(trec_labels)

for label, count in sorted(trec_counts.items()):
    percentage = count / len(trec_labels) * 100
    print(f"{label}: {count} ({percentage:.2f}%)")