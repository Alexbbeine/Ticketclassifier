import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Standardmäßig werden Titel und Beschreibung zu einem gemeinsamen Eingabetext zusammengeführt.
TEXT_COLUMNS_DEFAULT = ["Titel", "Beschreibung"]


# Standardmäßig werden Titel und Beschreibung zu einem gemeinsamen Eingabetext zusammengeführt.
def read_table(path: Path, sheet_name: str = "Tabelle") -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name)
    raise ValueError(f"Nicht unterstütztes Dateiformat: {suffix}")


# Baut aus den angegebenen Textspalten einen gemeinsamen Text pro Zeile.
def build_text(df: pd.DataFrame, text_cols: list[str]) -> pd.Series:
    return (
        df[text_cols]
        .fillna("")
        .astype(str)
        .agg("\n\n".join, axis=1)
        .str.strip()
    )


# Bereinigt den Datensatz, prüft Pflichtspalten und erzeugt die Zielstruktur mit Text und Label.
def prepare_dataframe(df: pd.DataFrame, label_col: str, text_cols: list[str]) -> pd.DataFrame:
    required = text_cols + [label_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Fehlende Spalten: {missing}")

    work = df.copy()
    work["text"] = build_text(work, text_cols)
    work[label_col] = work[label_col].fillna("").astype(str).str.strip()

    work = work[(work["text"] != "") & (work[label_col] != "")].copy()
    work = work.rename(columns={label_col: "label_text"})[["text", "label_text"]]
    work = work.drop_duplicates(subset=["text", "label_text"]).reset_index(drop=True)
    return work


# Prüft, ob jede Klasse im jeweiligen Split mindestens eine definierte Mindestanzahl an Beispielen hat.
def ensure_min_class_counts(df: pd.DataFrame, split_name: str, minimum: int = 1) -> None:
    counts = df["label_text"].value_counts()
    too_small = counts[counts < minimum]
    if not too_small.empty:
        raise ValueError(
            f"Im Split '{split_name}' sind Klassen mit weniger als {minimum} Beispiel(en) enthalten:\n"
            + too_small.to_string()
        )


# Wandelt Textlabels in numerische Label-IDs um und prüft, ob unbekannte Labels vorkommen.
def add_label_ids(df: pd.DataFrame, label2id: dict[str, int]) -> pd.DataFrame:
    out = df.copy()
    out["label"] = out["label_text"].map(label2id)
    if out["label"].isna().any():
        missing = out.loc[out["label"].isna(), "label_text"].unique().tolist()
        raise ValueError(
            "Mindestens ein Label in Val/Test kommt im Trainingssplit nicht vor: "
            f"{missing}"
        )
    out["label"] = out["label"].astype(int)
    return out


# Entfernt exakte Dubletten zwischen Train-, Validierungs- und Testsplit, um Dopplungen zu vermeiden.
def drop_cross_split_overlaps(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    train_keys = set(zip(train_df["text"], train_df["label_text"]))
    val_keys_before = set(zip(val_df["text"], val_df["label_text"]))
    test_keys_before = set(zip(test_df["text"], test_df["label_text"]))

    val_before = len(val_df)
    test_before = len(test_df)

    val_df = val_df[~val_df.apply(lambda r: (r["text"], r["label_text"]) in train_keys, axis=1)].reset_index(drop=True)

    updated_val_keys = set(zip(val_df["text"], val_df["label_text"]))
    forbidden_test = train_keys | updated_val_keys
    test_df = test_df[~test_df.apply(lambda r: (r["text"], r["label_text"]) in forbidden_test, axis=1)].reset_index(drop=True)
    updated_test_keys = set(zip(test_df["text"], test_df["label_text"]))

    overlap_info = {
        "val_removed_due_to_train_overlap": val_before - len(val_df),
        "test_removed_due_to_train_or_val_overlap": test_before - len(test_df),
        "train_val_exact_overlap_before_cleanup": len(train_keys & val_keys_before),
        "train_test_exact_overlap_before_cleanup": len(train_keys & test_keys_before),
        "val_test_exact_overlap_before_cleanup": len(val_keys_before & test_keys_before),
        "val_test_exact_overlap_after_cleanup": len(updated_val_keys & updated_test_keys),
    }
    return train_df, val_df, test_df, overlap_info



# Eigener Trainer, der optional Klassengewichte für unausgeglichene Klassenverteilungen berücksichtigt.
class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# Berechnet Klassengewichte auf Basis der Häufigkeit der Labels im Trainingssplit.
def compute_class_weights(y: pd.Series, label2id: dict[str, int]) -> torch.Tensor:
    counts = y.value_counts().to_dict()
    total = len(y)
    num_classes = len(label2id)
    weights = []
    for label, idx in sorted(label2id.items(), key=lambda x: x[1]):
        c = counts[label]
        weights.append(total / (num_classes * c))
    return torch.tensor(weights, dtype=torch.float)


# Berechnet Klassengewichte auf Basis der Häufigkeit der Labels im Trainingssplit.
def to_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df[["text", "label_text", "label"]], preserve_index=False)


# Hauptfunktion: lädt Daten, bereitet sie auf, trainiert das Modell und speichert die Ergebnisse.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True, help="Pfad zur Trainingsdatei")
    parser.add_argument("--val-data", required=True, help="Pfad zur Validierungsdatei")
    parser.add_argument("--test-data", required=True, help="Pfad zur Testdatei")
    parser.add_argument("--label-col", required=True, help="Zielspalte, z. B. Typ, Bereich, Prio oder Schweregrad")
    parser.add_argument("--text-cols", nargs="+", default=TEXT_COLUMNS_DEFAULT, help="Textspalten, standardmäßig Titel Beschreibung")
    parser.add_argument("--model", default="deepset/gbert-base")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sheet-name", default="Tabelle")
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-class-weights", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_train_df = read_table(Path(args.train_data), sheet_name=args.sheet_name)
    raw_val_df = read_table(Path(args.val_data), sheet_name=args.sheet_name)
    raw_test_df = read_table(Path(args.test_data), sheet_name=args.sheet_name)

    train_df = prepare_dataframe(raw_train_df, label_col=args.label_col, text_cols=args.text_cols)
    val_df = prepare_dataframe(raw_val_df, label_col=args.label_col, text_cols=args.text_cols)
    test_df = prepare_dataframe(raw_test_df, label_col=args.label_col, text_cols=args.text_cols)

    train_df, val_df, test_df, overlap_info = drop_cross_split_overlaps(train_df, val_df, test_df)

    ensure_min_class_counts(train_df, "train", minimum=2)
    ensure_min_class_counts(val_df, "val", minimum=1)
    ensure_min_class_counts(test_df, "test", minimum=1)

    labels = sorted(train_df["label_text"].unique().tolist())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    # Sicherstellen, dass Val/Test keine unbekannten Labels enthalten
    _ = add_label_ids(val_df, label2id)
    _ = add_label_ids(test_df, label2id)

    train_df = add_label_ids(train_df, label2id)
    val_df = add_label_ids(val_df, label2id)
    test_df = add_label_ids(test_df, label2id)

    train_ds = to_dataset(train_df)
    val_ds = to_dataset(val_df)
    test_ds = to_dataset(test_df)

    if "gbert" in args.model.lower() or "dbmdz" in args.model.lower():
        tokenizer = BertTokenizer.from_pretrained(args.model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Tokenisierung des Eingabetexts für das Transformer-Modell.
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    tokenized_train = train_ds.map(tokenize, batched=True, remove_columns=["text", "label_text"])
    tokenized_val = val_ds.map(tokenize, batched=True, remove_columns=["text", "label_text"])
    tokenized_test = test_ds.map(tokenize, batched=True, remove_columns=["text", "label_text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if "gbert" in args.model.lower() or "bert-base-german" in args.model.lower():
        config = BertConfig.from_pretrained(
            args.model,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
        )
        model = BertForSequenceClassification.from_pretrained(
            args.model,
            config=config,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
        )

    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(train_df["label_text"], label2id)

    # Bewertungsmetriken für Validierung und Test.
    def compute_metrics(eval_pred):
        logits, labels_np = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels_np, preds),
            "macro_f1": f1_score(labels_np, preds, average="macro"),
            "weighted_f1": f1_score(labels_np, preds, average="weighted"),
        }

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=25,
        report_to="none",
        dataloader_num_workers=0,
        use_cpu=True,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    test_metrics = trainer.evaluate(tokenized_test, metric_key_prefix="test")
    pred_output = trainer.predict(tokenized_test)
    y_true = pred_output.label_ids
    y_pred = np.argmax(pred_output.predictions, axis=1)

    report = classification_report(
        y_true,
        y_pred,
        target_names=[id2label[i] for i in range(len(id2label))],
        output_dict=True,
        zero_division=0,
    )

    (output_dir / "metrics_test.json").write_text(
        json.dumps(test_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "classification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "label_mapping.json").write_text(
        json.dumps(
            {
                "label2id": label2id,
                "id2label": {str(k): v for k, v in id2label.items()},
                "raw_rows": {
                    "train": len(raw_train_df),
                    "val": len(raw_val_df),
                    "test": len(raw_test_df),
                },
                "prepared_rows_after_cleaning_and_dedup": {
                    "train": len(train_df),
                    "val": len(val_df),
                    "test": len(test_df),
                },
                "overlap_cleanup": overlap_info,
                "use_class_weights": bool(args.use_class_weights),
                "base_model": args.model,
                "max_length": args.max_length,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Training abgeschlossen.")
    print(f"Modell gespeichert in: {output_dir}")
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
