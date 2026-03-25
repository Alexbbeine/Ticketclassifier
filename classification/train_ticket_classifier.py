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
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Standardmäßig werden Titel und Beschreibung zu einem gemeinsamen Eingabetext zusammengeführt.
TEXT_COLUMNS_DEFAULT = ["Titel", "Beschreibung"]

# Mindestanzahl von Beispielen pro Klasse in den einzelnen Splits.
# Diese Werte werden sowohl beim automatischen Split als auch bei der späteren Prüfung verwendet.
MIN_TRAIN_PER_CLASS = 8
MIN_VAL_PER_CLASS = 1
MIN_TEST_PER_CLASS = 1


# Liest Daten aus einer CSV oder Excel-Datei ein.
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


# Teilt einen bereits vorbereiteten Datensatz stratifiziert in Train, Validierung und Test auf.
# Dabei wird je Klasse so nah wie möglich an 80:10:10 geblieben, ohne die Mindestanforderungen
# pro Split zu verletzen.
def split_prepared_dataframe(
    df: pd.DataFrame,
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
):
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Die Split-Verhältnisse müssen zusammen 1.0 ergeben.")

    min_total_per_class = MIN_TRAIN_PER_CLASS + MIN_VAL_PER_CLASS + MIN_TEST_PER_CLASS
    label_counts = df["label_text"].value_counts()
    too_small = label_counts[label_counts < min_total_per_class]
    if not too_small.empty:
        raise ValueError(
            "Für den automatischen Split werden pro Klasse mindestens "
            f"{min_total_per_class} Beispiele benötigt. Zu kleine Klassen:\n"
            + too_small.to_string()
        )

    train_parts = []
    val_parts = []
    test_parts = []

    for _, group in df.groupby("label_text", sort=False):
        group = group.sample(frac=1, random_state=seed).reset_index(drop=True)
        n = len(group)

        n_train = int(np.floor(n * train_ratio))
        n_val = int(np.floor(n * val_ratio))
        n_test = n - n_train - n_val

        n_train = max(n_train, MIN_TRAIN_PER_CLASS)
        n_val = max(n_val, MIN_VAL_PER_CLASS)
        n_test = max(n_test, MIN_TEST_PER_CLASS)

        overflow = n_train + n_val + n_test - n

        if overflow > 0:
            reducible_train = max(0, n_train - MIN_TRAIN_PER_CLASS)
            take = min(overflow, reducible_train)
            n_train -= take
            overflow -= take

        if overflow > 0:
            reducible_test = max(0, n_test - MIN_TEST_PER_CLASS)
            take = min(overflow, reducible_test)
            n_test -= take
            overflow -= take

        if overflow > 0:
            reducible_val = max(0, n_val - MIN_VAL_PER_CLASS)
            take = min(overflow, reducible_val)
            n_val -= take
            overflow -= take

        if overflow > 0:
            raise ValueError(
                f"Klasse '{label}' konnte nicht in einen gültigen 80:10:10 Split überführt werden."
            )

        train_parts.append(group.iloc[:n_train])
        val_parts.append(group.iloc[n_train:n_train + n_val])
        test_parts.append(group.iloc[n_train + n_val:n_train + n_val + n_test])

    train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

    return train_df, val_df, test_df

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


# Entfernt exakte Dubletten zwischen Train-, Validierungs- und Testsplit.
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


# Eigene Trainer-Klasse, die optional Klassengewichte berücksichtigt.
# Das ist hilfreich, wenn einzelne Ticketklassen deutlich seltener vorkommen als andere.
class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    # Überschreibt die Verlustberechnung, um bei Bedarf gewichtete Fehlerkosten zu nutzen.
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.get("logits")

        if not hasattr(self, "_printed_device"):
            print("Trainer-Device:", self.args.device, flush=True)
            print("Modell-Device im Trainer:", next(model.parameters()).device, flush=True)
            print("Aktuelles Rechengerät im Forward-Pass:", logits.device, flush=True)
            if torch.cuda.is_available():
                print("GPU-Speicher belegt (MB):", torch.cuda.memory_allocated() / 1024**2, flush=True)
            self._printed_device = True

        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# Berechnet Klassengewichte auf Basis der Häufigkeit der Labels im Trainingssplit.
def compute_class_weights(y: pd.Series, label2id: dict[str, int]) -> torch.Tensor:
    counts = y.value_counts().to_dict()
    total = len(y)
    num_classes = len(label2id)
    weights = []
    for label, _ in sorted(label2id.items(), key=lambda x: x[1]):
        c = counts[label]
        weights.append(total / (num_classes * c))
    return torch.tensor(weights, dtype=torch.float)


# Wandelt einen Pandas DataFrame in ein Hugging Face Dataset um.
def to_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df[["text", "label_text", "label"]], preserve_index=False)


# Führt den vollständigen Trainingsablauf für einen Ticketklassifikator aus.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Pfad zu einer einzelnen Datei, die intern in Train, Val und Test aufgeteilt wird")
    parser.add_argument("--train-data", help="Pfad zur Trainingsdatei")
    parser.add_argument("--val-data", help="Pfad zur Validierungsdatei")
    parser.add_argument("--test-data", help="Pfad zur Testdatei")
    parser.add_argument("--label-col", required=True, help="Zielspalte, z. B. Typ, Bereich, Prio oder Schweregrad")
    parser.add_argument("--text-cols", nargs="+", default=TEXT_COLUMNS_DEFAULT, help="Textspalten, standardmäßig Titel Beschreibung")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sheet-name", default="Tabelle")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--do-hpo", action="store_true")
    parser.add_argument("--hpo-trials", type=int, default=10)
    args = parser.parse_args()

    single_file_mode = args.data is not None
    pre_split_mode = all(
        value is not None for value in (args.train_data, args.val_data, args.test_data)
    )

    if single_file_mode and any(
        value is not None for value in (args.train_data, args.val_data, args.test_data)
    ):
        parser.error(
            "Bitte entweder --data oder --train-data/--val-data/--test-data verwenden, nicht beides gleichzeitig."
        )

    if not single_file_mode and not pre_split_mode:
        parser.error(
            "Bitte entweder --data für eine einzelne Datei oder alle drei Parameter "
            "--train-data, --val-data und --test-data angeben."
        )

    if args.do_hpo:
        try:
            import optuna
        except ImportError as exc:
            raise ImportError(
                "Für --do-hpo muss das Paket 'optuna' installiert sein. Installiere es mit: pip install optuna"
            ) from exc

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if single_file_mode:
        # Eine einzelne Datei einlesen, bereinigen und danach stratifiziert in Train, Val und Test aufteilen.
        raw_all_df = read_table(Path(args.data), sheet_name=args.sheet_name)
        prepared_all_df = prepare_dataframe(raw_all_df, label_col=args.label_col, text_cols=args.text_cols)
        train_df, val_df, test_df = split_prepared_dataframe(prepared_all_df, seed=args.seed)
        overlap_info = {
            "val_removed_due_to_train_overlap": 0,
            "test_removed_due_to_train_or_val_overlap": 0,
            "train_val_exact_overlap_before_cleanup": 0,
            "train_test_exact_overlap_before_cleanup": 0,
            "val_test_exact_overlap_before_cleanup": 0,
            "val_test_exact_overlap_after_cleanup": 0,
        }
        input_metadata = {
            "input_mode": "single_file_auto_split",
            "raw_rows_source": len(raw_all_df),
            "prepared_rows_after_cleaning_and_dedup_source": len(prepared_all_df),
            "split_rows": {
                "train": len(train_df),
                "val": len(val_df),
                "test": len(test_df),
            },
        }
    else:
        # Rohdaten für Training, Validierung und Test einlesen.
        raw_train_df = read_table(Path(args.train_data), sheet_name=args.sheet_name)
        raw_val_df = read_table(Path(args.val_data), sheet_name=args.sheet_name)
        raw_test_df = read_table(Path(args.test_data), sheet_name=args.sheet_name)

        # Alle drei Splits in die gemeinsame Trainingsstruktur überführen.
        train_df = prepare_dataframe(raw_train_df, label_col=args.label_col, text_cols=args.text_cols)
        val_df = prepare_dataframe(raw_val_df, label_col=args.label_col, text_cols=args.text_cols)
        test_df = prepare_dataframe(raw_test_df, label_col=args.label_col, text_cols=args.text_cols)

        # Exakte Überschneidungen zwischen den Splits entfernen.
        train_df, val_df, test_df, overlap_info = drop_cross_split_overlaps(train_df, val_df, test_df)

        input_metadata = {
            "input_mode": "pre_split_files",
            "raw_rows": {
                "train": len(raw_train_df),
                "val": len(raw_val_df),
                "test": len(raw_test_df),
            },
            "prepared_rows_after_cleaning_dedup_and_overlap_cleanup": {
                "train": len(train_df),
                "val": len(val_df),
                "test": len(test_df),
            },
        }
    # Mindestanzahl pro Klasse sicherstellen.
    ensure_min_class_counts(train_df, "train", minimum=MIN_TRAIN_PER_CLASS)
    ensure_min_class_counts(val_df, "val", minimum=MIN_VAL_PER_CLASS)
    ensure_min_class_counts(test_df, "test", minimum=MIN_TEST_PER_CLASS)

    # Alle im Trainingssplit vorhandenen Klassen in numerische IDs übersetzen.
    labels = sorted(train_df["label_text"].unique().tolist())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    all_label_ids = list(range(len(labels)))

    # Sicherstellen, dass Val/Test keine unbekannten Labels enthalten.
    _ = add_label_ids(val_df, label2id)
    _ = add_label_ids(test_df, label2id)

    # Die Label-IDs in allen Splits ergänzen.
    train_df = add_label_ids(train_df, label2id)
    val_df = add_label_ids(val_df, label2id)
    test_df = add_label_ids(test_df, label2id)

    # Daten in das Dataset-Format für die Transformers-Bibliothek überführen.
    train_ds = to_dataset(train_df)
    val_ds = to_dataset(val_df)
    test_ds = to_dataset(test_df)

    # Für einige BERT-Modelle wird explizit der BertTokenizer genutzt.
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

    # Der Data Collator sorgt dafür, dass unterschiedlich lange Texte innerhalb eines Batches automatisch auf eine gemeinsame Länge aufgefüllt werden.
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Optional: Klassengewichte berechnen, damit seltene Klassen im Loss stärker gewichtet werden.
    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(train_df["label_text"], label2id)

    # Baut ein frisches Klassifikationsmodell auf.
    def build_model():
        if "gbert" in args.model.lower() or "bert-base-german" in args.model.lower():
            config = BertConfig.from_pretrained(
                args.model,
                num_labels=len(labels),
                id2label=id2label,
                label2id=label2id,
            )
            return BertForSequenceClassification.from_pretrained(
                args.model,
                config=config,
            )

        return AutoModelForSequenceClassification.from_pretrained(
            args.model,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
        )

    # Für die Hyperparameter-Suche wird pro Trial ein frisches Modell benötigt.
    def model_init(trial=None):
        return build_model()

    # Bewertungsmetriken für Validierung und Test.
    def compute_metrics(eval_pred):
        logits, labels_np = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels_np, preds),
            "macro_f1": f1_score(
                labels_np,
                preds,
                labels=all_label_ids,
                average="macro",
                zero_division=0,
            ),
            "weighted_f1": f1_score(
                labels_np,
                preds,
                labels=all_label_ids,
                average="weighted",
                zero_division=0,
            ),
        }

    # Erzeugt die TrainingArguments für einen Trainingslauf.
    def make_training_args(
        run_output_dir,
        learning_rate,
        train_batch_size,
        eval_batch_size,
        grad_accum,
        num_train_epochs,
        weight_decay,
        warmup_ratio=0.0,
    ):
        return TrainingArguments(
            output_dir=str(run_output_dir),
            learning_rate=learning_rate,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=grad_accum,
            num_train_epochs=num_train_epochs,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_macro_f1",
            greater_is_better=True,
            save_total_limit=2,
            logging_steps=25,
            report_to="none",
            dataloader_num_workers=0,
            fp16=torch.cuda.is_available(),
            seed=args.seed,
            data_seed=args.seed,
        )

    # Hier werden die besten Hyperparameter aus der Suche gespeichert.
    best_hyperparams = {}

    if args.do_hpo:
        # Definiert den Suchraum für Optuna.
        def hp_space(trial):
            return {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
                "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 5),
                "per_device_train_batch_size": trial.suggest_categorical(
                    "per_device_train_batch_size", [4, 8, 16]
                ),
                "gradient_accumulation_steps": trial.suggest_categorical(
                    "gradient_accumulation_steps", [1, 2, 4]
                ),
                "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
                "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
            }

        # Legt fest, dass die Macro-F1 maximiert werden soll.
        def compute_objective(metrics):
            return metrics["eval_macro_f1"]

        # Trainingsargumente für die Suchläufe.
        hpo_args = make_training_args(
            run_output_dir=output_dir / "hpo_runs",
            learning_rate=args.lr,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            grad_accum=args.grad_accum,
            num_train_epochs=args.epochs,
            weight_decay=args.weight_decay,
            warmup_ratio=0.0,
        )

        # Eigener Trainer für die Hyperparameter-Suche.
        hpo_trainer = WeightedTrainer(
            class_weights=class_weights,
            model=None,
            model_init=model_init,
            args=hpo_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        # Führt die eigentliche Hyperparameter-Suche aus.
        best_trial = hpo_trainer.hyperparameter_search(
            direction="maximize",
            backend="optuna",
            hp_space=hp_space,
            n_trials=args.hpo_trials,
            compute_objective=compute_objective,
        )

        best_hyperparams = best_trial.hyperparameters
        print("Beste Hyperparameter:")
        print(json.dumps(best_hyperparams, ensure_ascii=False, indent=2), flush=True)

    # Erstellt die finalen Trainingsargumente.
    # Falls HPO aktiv war, werden die besten gefundenen Werte verwendet, andernfalls die Standardwerte aus den Argumenten.
    final_training_args = make_training_args(
        run_output_dir=output_dir,
        learning_rate=float(best_hyperparams.get("learning_rate", args.lr)),
        train_batch_size=int(best_hyperparams.get("per_device_train_batch_size", args.train_batch_size)),
        eval_batch_size=args.eval_batch_size,
        grad_accum=int(best_hyperparams.get("gradient_accumulation_steps", args.grad_accum)),
        num_train_epochs=int(best_hyperparams.get("num_train_epochs", args.epochs)),
        weight_decay=float(best_hyperparams.get("weight_decay", args.weight_decay)),
        warmup_ratio=float(best_hyperparams.get("warmup_ratio", 0.0)),
    )

    final_hyperparams = {
        "learning_rate": float(best_hyperparams.get("learning_rate", args.lr)),
        "per_device_train_batch_size": int(best_hyperparams.get("per_device_train_batch_size", args.train_batch_size)),
        "per_device_eval_batch_size": int(args.eval_batch_size),
        "gradient_accumulation_steps": int(best_hyperparams.get("gradient_accumulation_steps", args.grad_accum)),
        "num_train_epochs": int(best_hyperparams.get("num_train_epochs", args.epochs)),
        "weight_decay": float(best_hyperparams.get("weight_decay", args.weight_decay)),
        "warmup_ratio": float(best_hyperparams.get("warmup_ratio", 0.0)),
        "max_length": int(args.max_length),
        "seed": int(args.seed),
    }

    # Finaler Trainer für das eigentliche Endmodell.
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=build_model(),
        args=final_training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("CUDA verfügbar:", torch.cuda.is_available(), flush=True)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Trainer-Device:", trainer.args.device, flush=True)
    print("Trainer-Modell-Device vor train():", next(trainer.model.parameters()).device, flush=True)

    # Modell trainieren und danach Modellgewichte sowie Tokenizer im Ausgabeordner speichern.
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Das beste gespeicherte Modell abschließend auf dem Testsplit auswerten.
    test_metrics = trainer.evaluate(tokenized_test, metric_key_prefix="test")
    pred_output = trainer.predict(tokenized_test)
    y_true = pred_output.label_ids
    y_pred = np.argmax(pred_output.predictions, axis=1)

    # Detaillierten Klassifikationsbericht pro Klasse erstellen.
    report = classification_report(
        y_true,
        y_pred,
        labels=all_label_ids,
        target_names=[id2label[i] for i in all_label_ids],
        output_dict=True,
        zero_division=0,
    )

    # Testmetriken, Klassifikationsbericht und Label-Mapping speichern.
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
                **input_metadata,
                "overlap_cleanup": overlap_info,
                "minimum_examples_per_class": {
                    "train": MIN_TRAIN_PER_CLASS,
                    "val": MIN_VAL_PER_CLASS,
                    "test": MIN_TEST_PER_CLASS,
                },
                "use_class_weights": bool(args.use_class_weights),
                "base_model": args.model,
                "max_length": args.max_length,
                "best_hyperparameters": best_hyperparams if best_hyperparams else None,
                "final_hyperparameters_used": final_hyperparams,
                "hpo_enabled": bool(args.do_hpo),
                "hpo_trials": args.hpo_trials if args.do_hpo else 0,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.do_hpo:
        (output_dir / "best_hyperparameters.json").write_text(
            json.dumps(best_hyperparams, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("Training abgeschlossen.")
    print(f"Modell gespeichert in: {output_dir}")
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()