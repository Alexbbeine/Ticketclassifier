import argparse
import json
from pathlib import Path

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from train_ticket_classifier import read_table, prepare_dataframe, add_label_ids

TEXT_COLUMNS_DEFAULT = ["Titel", "Beschreibung"]

# Hauptfunktion zur Auswertung eines bereits trainierten Modells auf einem separaten Testdatensatz.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--label-col", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--sheet-name", default="Tabelle")
    parser.add_argument("--text-cols", nargs="+", default=TEXT_COLUMNS_DEFAULT)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    raw_test_df = read_table(Path(args.test_data), sheet_name=args.sheet_name)
    test_df = prepare_dataframe(raw_test_df, label_col=args.label_col, text_cols=args.text_cols)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    label2id = {str(k): int(v) for k, v in model.config.label2id.items()}
    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}

    # Prüft, ob alle Labels im Testdatensatz auch im trainierten Modell bekannt sind.
    test_df = add_label_ids(test_df, label2id)

    test_ds = Dataset.from_pandas(test_df[["text", "label_text", "label"]], preserve_index=False)

    # Tokenisiert den Testtext im gleichen Format wie beim Training.
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    tokenized_test = test_ds.map(tokenize, batched=True, remove_columns=["text", "label_text"])

    # Erstellt einen Trainer nur für die Evaluation und Vorhersage auf dem Testdatensatz.
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(model_dir / "_eval_tmp"),
            per_device_eval_batch_size=args.eval_batch_size,
            report_to="none",
            use_cpu=True,
        ),
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    eval_metrics = trainer.evaluate(tokenized_test, metric_key_prefix="test")
    pred_output = trainer.predict(tokenized_test)

    y_true = pred_output.label_ids
    y_pred = np.argmax(pred_output.predictions, axis=1)

    # Berechnet die wichtigsten Gütemaße für das Modell auf dem Testsplit.
    metrics = {
        **eval_metrics,
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_macro_f1": f1_score(y_true, y_pred, average="macro"),
        "test_weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }

    report = classification_report(
        y_true,
        y_pred,
        target_names=[id2label[i] for i in range(len(id2label))],
        output_dict=True,
        zero_division=0,
    )

    # Speichert Metriken und Klassifikationsbericht im Modellordner.
    (model_dir / "metrics_test.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (model_dir / "classification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nGespeichert in: {model_dir}")


if __name__ == "__main__":
    main()
