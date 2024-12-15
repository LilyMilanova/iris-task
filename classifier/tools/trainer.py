import json
import os
from pathlib import Path
from typing import Dict, Tuple, Union
import logging
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from skmultilearn.model_selection import iterative_train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments)

from classifier.tools.common import MAX_LENGTH, MODEL_NAME, SNAPSHOT, \
    get_unique_categories, plot_analysis, preprocess_abstract

METRICS_THRESHOLD = 0.3


_model_trainer = None


def get_trainer(sample_size: Union[str, int] = 87):
    """Singleton instance for TrainerPipeline."""
    global _model_trainer
    if _model_trainer is None:
        _model_trainer = MultiLabelClassificationModelTrainer(
            full_model_name=MODEL_NAME, sample_size=sample_size
        )
    return _model_trainer


class MultiLabelClassificationModelTrainer:
    def __init__(self,
                 full_model_name: str = MODEL_NAME,
                 sample_size: Union[str, int] = 1,
                 plot: bool = False,
                 enable_logging: bool = True
                 ):
        self.plot = plot
        self.logger = logging.getLogger(__name__)
        self.full_model_name = full_model_name
        if enable_logging:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
            self.logger.setLevel(logging.INFO)
        else:
            logging.disable(logging.CRITICAL)

        if int(sample_size) == 100:
            sample_size = 87
        self.sample_size = str(sample_size)

        self.data_dir = Path(__file__).resolve().parents[2] / "data"
        self.models_dir = self.data_dir / "models"
        self.out_dir = self.models_dir / "bert_finetuned_multilabel"
        self.datasets_dir = self.data_dir / "datasets"
        self.configs_dir = self.data_dir / "configs"
        self.model_name = self.full_model_name.split('/')[1] if '/' in self.full_model_name else self.full_model_name

        self.df = self.load_training_dataset()
        self.all_categories = get_unique_categories(self.df["categories"])
        self.num_labels = len(self.all_categories)

        self.tokenizer = AutoTokenizer.from_pretrained(self.full_model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.full_model_name, num_labels=self.num_labels, problem_type="multi_label_classification"
        )
        self.cat2id = self.save_cat2id()
        self.id2cat = {int(v): k for k, v in self.cat2id.items()}

    def load_training_dataset(self) -> pd.DataFrame:
        training_dataset = self.datasets_dir / f"{SNAPSHOT}-sample_{self.sample_size}%-json.json"
        df = pd.read_json(training_dataset)
        self.log(f"File {training_dataset} loaded successfully:")
        self.log(df.info())

        if self.plot:
            self.log("Analyzing and creating plots...")
            plot_analysis(df)
        return df

    def unload_training_dataset(self):
        self.df.drop(self.df.index, inplace=True)
        self.df = None
        import gc
        gc.collect()

    def train(self, hyp_search: bool = False, n_trials: int = 0):
        self.logger.warning("Start model training...")
        train_df, val_df = self.split_data()

        self.log("Plot category distribution of train and validation sets...")
        if self.logger.isEnabledFor(logging.INFO):
            plot_analysis(train_df, id2cat=self.id2cat, df_title="Train Dataset")
            plot_analysis(val_df, id2cat=self.id2cat, df_title="Validation Dataset")

        train_dataset = self.preprocess_data(train_df)
        val_dataset = self.preprocess_data(val_df)

        training_args = TrainingArguments(
            output_dir=str(self.out_dir),
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

        best_trial = None
        if hyp_search:
            trainer = Trainer(
                model_init=self.model_init,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=self.tokenizer,
                compute_metrics=self.compute_metrics,
            )
            self.logger.info("Performing hyperparameter search...")
            best_trial = trainer.hyperparameter_search(
                direction="maximize",
                hp_space=self.hp_space,
                backend="optuna",
                n_trials=n_trials,
            )
            self.logger.info(f"Best trial: {best_trial}")
            self.logger.info(f"Best hyperparameters: {best_trial.hyperparameters}")
        else:
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=self.tokenizer,
                compute_metrics=self.compute_metrics,
            )
            self.log("Start training...")
            trainer.train()

        model_dir = self.models_dir / f"{self.model_name}{self.sample_size}"
        if hyp_search and best_trial:
            self.logger.info(f"Load the best model's checkpoint: {best_trial} from hyperparameter search...")
            best_run_dir = self.models_dir / self.out_dir / f"run-{best_trial.run_id}"

            checkpoints = [d for d in os.listdir(best_run_dir) if d.startswith("checkpoint-")]
            checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
            best_checkpoint_dir = best_run_dir / checkpoints[-1]

            tokenizer = AutoTokenizer.from_pretrained(str(best_checkpoint_dir))
            best_model = AutoModelForSequenceClassification.from_pretrained(
                str(best_checkpoint_dir), local_files_only=True
            )
            tokenizer.save_pretrained(str(model_dir))
            best_model.save_pretrained(str(model_dir))
            self.logger.info(f"Best model saved successfully at {model_dir}.")

            # Do we need the other files?

            trainer = Trainer(
                model=best_model,
                args=training_args,
                eval_dataset=val_dataset,
                tokenizer=tokenizer,
                compute_metrics=self.compute_metrics,
            )
        else:
            trainer.save_model(str(model_dir))
            self.log(f"Model saved successfully at {model_dir}.")

        eval_results = trainer.evaluate()
        self.logger.warning(f"Evaluation results: {eval_results}")

        training_metrics = model_dir / "training_metrics.json"
        with open(training_metrics, "w") as f:
            json.dump(trainer.state.log_history, f, indent=4)

        self.log("Saved training metrics to a JSON File")

    def preprocess_data(self, data: pd.DataFrame) -> Dataset:
        """Preprocess input data for training/testing."""
        def tokenize(example):
            return self.tokenizer(
                example["abstract"], max_length=MAX_LENGTH,
                truncation=True, padding="max_length", return_tensors="pt"
            )
        dataset = Dataset.from_pandas(data, preserve_index=False)
        dataset = dataset.map(tokenize, batched=True, num_proc=4)
        dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )
        return dataset

    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataset into train and test sets."""
        self.df["abstract"] = self.df["abstract"].apply(preprocess_abstract)
        self.log("Pre-processed abstracts...")

        self.df["label"] = self.df["categories"].apply(self.encode_labels)
        self.log("Encoded labels into multi-hot arrays...")

        X = self.df["abstract"].values
        y = np.stack(self.df["label"].values)  # Ensure shape (num_samples, num_labels)
        X = X.reshape(-1, 1)

        self.unload_training_dataset()

        self.log("Performing iterative stratified split...")
        X_train, y_train, X_val, y_val = iterative_train_test_split(X, y, test_size=0.1)
        train_df = pd.DataFrame({"abstract": X_train.reshape(-1)})
        val_df = pd.DataFrame({"abstract": X_val.reshape(-1)})
        train_df["label"] = list(y_train)
        val_df["label"] = list(y_val)

        return train_df, val_df

    def save_cat2id(self):
        cat2id = {c: i for i, c in enumerate(self.all_categories)}
        self.log(f"Found {len(cat2id)} unique categories.")

        cat2id_path = self.configs_dir / f"cat2id-sample_{self.sample_size}%.json"
        with open(cat2id_path, "w") as f:
            json.dump(cat2id, f, indent=2)
        self.log(f"Saved category to id mapping to {cat2id_path}.")

        return cat2id

    def encode_labels(self, categories):
        labels = np.zeros(self.num_labels, dtype=np.float32)
        for category in categories.split():
            if category in self.cat2id:
                labels[self.cat2id[category]] = 1.0
        return labels

    @staticmethod
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

    def model_init(self):
        """Initialize a new instance of the model for each hyperparameter trial."""
        return AutoModelForSequenceClassification.from_pretrained(
            self.full_model_name,
            num_labels=self.num_labels,
            problem_type="multi_label_classification"
        )

    @staticmethod
    def hp_space(trial) -> Dict[str, float]:
        """Define the hyperparameter search space for tuning."""
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64]),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 5),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
        }

    def log(self, message: object) -> None:
        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(message)


if __name__ == "__main__":
    # plot_analysis()
    model_trainer = get_trainer(sample_size="24")
    # model_trainer.train()
    model_trainer.train(hyp_search=True, n_trials=20)

