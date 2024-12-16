import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict, Dataset
import numpy as np
import json, torch
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import time
from transformers import EarlyStoppingCallback
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader 
import torch.nn as nn
import torch.nn.functional as F


early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=2,  # stop if no update in 3 continuous epoches
    early_stopping_threshold=0.01  # Threshold
)

#get the current time stamp: month-day-hour-minute-second
current_time = time.strftime("%m-%d-%H-%M-%S", time.localtime())

# load and process dataset
raw_dataset = pd.read_json('./trec06c_clean.json', lines=True)
dataset = Dataset.from_pandas(raw_dataset)

# Split dataset
train_test_split = dataset.train_test_split(test_size=0.3)
val_test_split = train_test_split["test"].train_test_split(test_size=0.5)
dataset = DatasetDict({
    "train": train_test_split["train"],
    "validation": val_test_split["train"],
    "test": val_test_split["test"]
})

# Tokenizer and model
model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize and preprocess labels
def tokenize_and_process(examples):
    examples["labels"] = [1 if x == "spam" else 0 for x in examples["Category"]]
    tokenized = tokenizer(
        examples["Message"],
        add_special_tokens=True,
        max_length=128,
        padding="max_length",
        truncation=True,
    )
    tokenized["labels"] = examples["labels"]
    return tokenized

tokenized_datasets = dataset.map(tokenize_and_process, batched=True, remove_columns=["Message", "Category"])

# Evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    cm = confusion_matrix(labels, predictions)
    return {
        "accuracy": (predictions == labels).mean(),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist()
    }


# Training arguments
training_args = TrainingArguments(
    output_dir=f"./results_{current_time}",
    per_device_train_batch_size=32,  # per device batch size
    per_device_eval_batch_size=32,  # validation batch size
    num_train_epochs=2, # epoch num for training
    learning_rate=1e-5,  # initial lr
    warmup_ratio=0.2,  # 20% steps for warming up
    lr_scheduler_type="cosine",  # cosine strategy
    weight_decay=0.01,  # weight decay
    evaluation_strategy="steps",  # per epoch evaluation
    save_strategy="steps",  # per epoch checkpoint
    eval_steps=50,  # 10 steps for evaluation
    save_steps=50,
    save_total_limit=3,  # 3 checkpoints left
    load_best_model_at_end=True,  # load best result model
    metric_for_best_model="eval_accuracy",  # use accuracy as metric
    greater_is_better=True,  
    logging_steps=10,  # log every 10 steps
    fp16=True,  # mix precision to speed up
)


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
    callbacks=[early_stopping_callback],
)

# Train and save
trainer.train()
trainer.save_model(f"./spam_detect_{current_time}")

# Evaluate
final_results = trainer.evaluate(tokenized_datasets["test"])
print(final_results)
with open(f'./results_{current_time}/test_results.json', 'w') as result_file:
    json.dump(final_results, result_file)


# tensorboard --logdir=./results
