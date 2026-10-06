import time, json, os
import numpy as np
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, f1_score

MODEL = "distilbert-base-uncased"
TASKS = {
    "agnews": dict(path="fancyzhx/ag_news",  text="text",     label="label",        n=4, test="test"),
    "sst2":   dict(path="stanfordnlp/sst2",  text="sentence", label="label",        n=2, test="validation"),
    "trec":   dict(path="CogComp/trec",      text="text",     label="coarse_label", n=6, test="test",
                   rev="refs/convert/parquet"),
}

tokenizer = AutoTokenizer.from_pretrained(MODEL)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=-1)
    return {"accuracy": accuracy_score(p.label_ids, preds),
            "f1": f1_score(p.label_ids, preds, average="macro")}

def run(name):
    cfg = TASKS[name]
    ds = load_dataset(cfg["path"], revision=cfg.get("rev"))
    ds = ds.rename_column(cfg["label"], "labels")
    ds = ds.map(lambda x: tokenizer(x[cfg["text"]], truncation=True, max_length=128),
                batched=True)

    base = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=cfg["n"])
    lora_cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16,
                          lora_dropout=0.1, target_modules=["q_lin", "v_lin"])
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    args = TrainingArguments(
        output_dir=f"./models/{name}", eval_strategy="epoch", save_strategy="no",
        learning_rate=2e-4, per_device_train_batch_size=32, per_device_eval_batch_size=64,
        num_train_epochs=1, weight_decay=0.01, logging_steps=50,
        report_to="none", fp16=True)

    trainer = Trainer(model=model, args=args, train_dataset=ds["train"],
                      eval_dataset=ds[cfg["test"]], compute_metrics=compute_metrics,
                      data_collator=DataCollatorWithPadding(tokenizer))

    t = time.time()
    trainer.train()
    train_time = time.time() - t
    r = trainer.evaluate()

    model.save_pretrained(f"adapters/{name}")
    out = dict(task=name, accuracy=r["eval_accuracy"], macro_f1=r["eval_f1"],
               trainable_params=trainable, total_params=total, train_time_s=train_time)
    print(out)
    json.dump(out, open(f"results/{name}.json", "w"), indent=2)

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    for n in TASKS: run(n)
