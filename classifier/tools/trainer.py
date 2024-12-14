import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from skmultilearn.model_selection import iterative_train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments)

from classifier.tools.common import MAX_LENGTH, MODEL_NAME, SNAPSHOT, \
    get_unique_categories, plot_analysis, preprocess_abstract


METRICS_THRESHOLD = 0.3


def train_model(model_name=MODEL_NAME, sample_size=1):
    all_categories, train_df, val_df = prepare_data(sample_size=sample_size, plot=True)

    print("---> Showing category distribution...")
    plot_analysis(train_df)
    plot_analysis(val_df)

    print("---> Converting to HuggingFace datasets...")
    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    val_dataset = Dataset.from_pandas(val_df, preserve_index=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def preprocess_function(examples):
        return tokenizer(
            examples["abstract"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

    print("---> Tokenizing datasets...")
    train_dataset = train_dataset.map(preprocess_function, batched=True, num_proc=4)
    val_dataset = val_dataset.map(preprocess_function, batched=True, num_proc=4)
    train_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )
    val_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )

    print("---> Initializing model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(all_categories),
        problem_type="multi_label_classification",
    )

    training_args = TrainingArguments(
        output_dir="./bert_finetuned_multilabel",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        num_train_epochs=4,
        learning_rate=5e-5,  # was 2e-5, try 4e-5
        weight_decay=0.01,
        logging_steps=1000,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        # does not show anything on tensorboard --logdir=~/Documents/Projects/iris-task/classifier/tools/bert_miniXX
        # report_to="tensorboard",
        save_total_limit=5,
    )

    print("---> Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    eval_results = trainer.evaluate()
    print("---> Evaluation results:", eval_results)

    model_dir = Path(__file__).parent.parent / "models" / f"{model_name.split('/')[1]}{str(sample_size)}"
    trainer.save_model(str(model_dir))
    print(f"---> Model saved successfully at {model_dir}.")

    training_metrics = model_dir / "training_metrics.json"
    with open(training_metrics, "w") as f:
        json.dump(trainer.state.log_history, f, indent=4)

    print("---> Saved training Metrics to a JSON File")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > METRICS_THRESHOLD).astype(int)

    return {
        "f1_micro": f1_score(labels, preds, average="micro"),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "avg_label_accuracy": (labels == preds).mean(),
        "precision_macro": precision_score(labels, preds, average="macro"),
        "precision_weighted": precision_score(labels, preds, average="weighted"),
        "recall_macro": recall_score(labels, preds, average="macro"),
        "recall_weighted": recall_score(labels, preds, average="weighted"),
        # "roc_auc_macro": roc_auc_score(labels, probs, average='macro')
    }


def prepare_data(sample_size=1, plot=False):
    if sample_size == 100:
        sample_size = 87

    resource_json_path = Path(__file__).parent.parent.parent / "data" / f"{SNAPSHOT}-sample_{str(sample_size)}%-json.json"
    df = pd.read_json(resource_json_path)

    print(f"---> File {resource_json_path} loaded successfully:")
    print(df.info())

    if plot:
        print("---> Analyzing and creating plots...")
        plot_analysis(df)

    all_categories = get_unique_categories(df["categories"])
    cat2id = {c: i for i, c in enumerate(all_categories)}
    print(f"---> Generated {len(cat2id)} unique categories.")

    cat2id_path = Path(__file__).parent / f"cat2id-sample_{str(sample_size)}%.json"
    with open(cat2id_path, "w") as f:
        json.dump(cat2id, f, indent=2)
    print(f"---> Saved category to id mapping to {cat2id_path}.")

    df["abstract"] = df["abstract"].apply(preprocess_abstract)
    print("---> Pre-processed abstracts...")

    def encode_labels(cats_str):
        labels = np.zeros(len(all_categories), dtype=np.float32)
        for c in cats_str.split():
            if c in cat2id:
                labels[cat2id[c]] = 1.0
        return labels

    df["label"] = df["categories"].apply(encode_labels)
    print("---> Encoded labels into multi-hot arrays...")

    X = df["abstract"].values
    Y = np.stack(df["label"].values)  # Ensure shape (num_samples, num_labels)
    X = X.reshape(-1, 1)

    print("---> Performing iterative stratified split...")
    X_train, Y_train, X_val, Y_val = iterative_train_test_split(X, Y, test_size=0.1)
    X_train = X_train.reshape(-1)
    X_val = X_val.reshape(-1)

    train_df = pd.DataFrame({"abstract": X_train})
    val_df = pd.DataFrame({"abstract": X_val})
    train_df["label"] = list(Y_train)
    val_df["label"] = list(Y_val)
    print("---> Return train and validation sets")

    return all_categories, train_df, val_df


def show_category_distribution(df, categories):
    """
    Display the distribution of categories in a DataFrame.

    Parameters:
    - df: A pandas DataFrame with a 'label' column containing multi-hot arrays (e.g. np.array([1,0,1,...])).
    - categories: A list of category names corresponding to each label index.

    The function prints out a DataFrame showing:
    - category name
    - count of how many samples have that category
    - percentage of samples that have that category
    """
    label_array = np.vstack(df["label"].values)
    counts = label_array.sum(axis=0)
    total_samples = label_array.shape[0]

    dist_df = pd.DataFrame(
        {
            "category": categories,
            "count": counts,
            "percentage": (counts / total_samples) * 100,
        }
    )

    dist_df = dist_df.sort_values("count", ascending=False)
    print(dist_df)



if __name__ == "__main__":
    # plot_analysis()
    train_model()
