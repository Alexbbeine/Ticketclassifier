from __future__ import annotations

import shutil
from pathlib import Path
from statistics import mean
from typing import Any

from config import (
    RPA_INBOX_DIR,
    STREAMLIT_AREA_OPTIONS,
    STREAMLIT_ENVIRONMENT_OPTIONS,
    STREAMLIT_IMPACT_OPTIONS,
    STREAMLIT_PRIORITY_OPTIONS,
    STREAMLIT_TICKET_TYPE_OPTIONS,
    TICKETS_DIR,
)
from storage import load_json, utc_now_iso, write_json_atomic

EDITABLE_TICKET_FIELDS = (
    "Title",
    "Area",
    "Iteration",
    "Description",
    "Ticket-Type",
    "Environment",
    "Prio",
    "Impact",
)

CLASSIFICATION_LABELS = {
    "ticket_type": "Ticket-Typ",
    "ticket_area": "Bereich",
    "ticket_priority": "Priorität",
    "ticket_impact": "Schweregrad",
}

FIXED_OPTION_MAP = {
    "ticket_type": list(STREAMLIT_TICKET_TYPE_OPTIONS),
    "area": list(STREAMLIT_AREA_OPTIONS),
    "priority": list(STREAMLIT_PRIORITY_OPTIONS),
    "impact": list(STREAMLIT_IMPACT_OPTIONS),
    "environment": list(STREAMLIT_ENVIRONMENT_OPTIONS),
    "iteration": [],
}


# Stellt sicher, dass das Verzeichnis für Ticketdateien existiert und gibt es zurück.
def ensure_ticket_directory(ticket_dir: Path | None = None) -> Path:
    directory = Path(ticket_dir or TICKETS_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


# Stellt sicher, dass das Zielverzeichnis für die RPA-Inbox existiert und gibt es zurück.
def ensure_rpa_inbox_directory(target_dir: Path | None = None) -> Path:
    directory = Path(target_dir or RPA_INBOX_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


# Liefert alle Ticket-JSON-Dateien in absteigender Reihenfolge zurück.
def iter_ticket_files(ticket_dir: Path | None = None) -> list[Path]:
    directory = ensure_ticket_directory(ticket_dir)
    return sorted(directory.glob("TICKET-*.json"), reverse=True)


# Wandelt einen beliebigen Wert sicher in einen String um.
def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


# Normalisiert einen Bereichswert für die interne Verarbeitung.
def normalize_area_value(value: Any) -> str:
    return str(value or "").strip()


# Bereitet einen Bereichswert für die Anzeige in der Oberfläche auf.
def format_area_display(value: Any) -> str:
    return str(value or "").strip()


# Normalisiert einen Ticketfeldwert vor Vergleich oder Speicherung.
def normalize_ticket_field(field: str, value: Any) -> str:
    text = str(value or "")
    if field == "Description":
        return text.strip("\n")
    return text.strip()


# Extrahiert alle numerischen Konfidenzwerte aus dem Klassifikationsergebnis.
def _extract_confidence_map(classification: dict[str, Any]) -> dict[str, float]:
    confidences: dict[str, float] = {}

    for classifier_key, payload in classification.items():
        score = payload.get("softmax_confidence")
        if isinstance(score, (int, float)):
            confidences[classifier_key] = float(score)

    return confidences


# Kürzt einen Text für die Vorschau und entfernt dabei überflüssige Leerzeichen.
def _truncate(value: str, limit: int = 160) -> str:
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


# Verdichtet einen Ticketdatensatz zu einer normalisierten Zeile für Übersicht und Filterung.
def normalize_ticket_record(record: dict[str, Any], source_path: Path) -> dict[str, Any]:
    email = record.get("email", {})
    ticket = record.get("ticket", {})
    meta = record.get("meta", {})
    manual_review = record.get("manual_review", {})

    ticket_id = _as_text(meta.get("message_id") or source_path.stem)
    received_utc = _as_text(email.get("received_utc"))
    created_utc = _as_text(meta.get("ticket_created_at_utc"))
    description = _as_text(ticket.get("Description") or email.get("body_cleaned") or email.get("body"))

    confidence_map = _extract_confidence_map(record.get("classification", {}))
    confidence_values = list(confidence_map.values())
    average_confidence = mean(confidence_values) if confidence_values else None
    minimum_confidence = min(confidence_values) if confidence_values else None

    history = manual_review.get("history", [])
    manually_edited = isinstance(history, list) and len(history) > 0

    return {
        "ticket_id": ticket_id,
        "file_name": source_path.name,
        "file_path": str(source_path),
        "title": _as_text(ticket.get("Title") or email.get("subject")),
        "sender": _as_text(email.get("sender")),
        "received_utc": received_utc,
        "ticket_created_at_utc": created_utc,
        "ticket_type": _as_text(ticket.get("Ticket-Type")),
        "area": normalize_area_value(ticket.get("Area")),
        "iteration": _as_text(ticket.get("Iteration")),
        "environment": _as_text(ticket.get("Environment")),
        "priority": _as_text(ticket.get("Prio")),
        "impact": _as_text(ticket.get("Impact")),
        "description": description,
        "description_preview": _truncate(description),
        "average_confidence": average_confidence,
        "minimum_confidence": minimum_confidence,
        "confidence_map": confidence_map,
        "manually_edited": manually_edited,
        "edit_count": len(history) if isinstance(history, list) else 0,
        "last_manual_save_utc": _as_text(manual_review.get("last_saved_at_utc")),
    }


# Lädt alle Ticketdateien und baut daraus den Index für die Übersicht auf.
def load_ticket_index(ticket_dir: Path | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for ticket_file in iter_ticket_files(ticket_dir):
        try:
            record = load_json(ticket_file)
            rows.append(normalize_ticket_record(record, ticket_file))
        except Exception as error:
            rows.append(
                {
                    "ticket_id": ticket_file.stem,
                    "file_name": ticket_file.name,
                    "file_path": str(ticket_file),
                    "title": "Fehler beim Laden",
                    "sender": "",
                    "received_utc": "",
                    "ticket_created_at_utc": "",
                    "ticket_type": "",
                    "area": "",
                    "iteration": "",
                    "environment": "",
                    "priority": "",
                    "impact": "",
                    "description": "",
                    "description_preview": f"Datei konnte nicht gelesen werden: {error}",
                    "average_confidence": None,
                    "minimum_confidence": None,
                    "confidence_map": {},
                    "manually_edited": False,
                    "edit_count": 0,
                    "last_manual_save_utc": "",
                    "load_error": str(error),
                }
            )

    rows.sort(
        key=lambda row: (
            row.get("received_utc", ""),
            row.get("ticket_created_at_utc", ""),
            row.get("file_name", ""),
        ),
        reverse=True,
    )
    return rows


# Sucht einen Ticketdatensatz anhand seiner Ticket-ID und lädt die passende Datei.
def load_ticket_record_by_id(
    ticket_id: str,
    ticket_dir: Path | None = None,
) -> tuple[Path, dict[str, Any]] | None:
    for ticket_file in iter_ticket_files(ticket_dir):
        record = load_json(ticket_file)
        message_id = _as_text(record.get("meta", {}).get("message_id") or ticket_file.stem)
        if message_id == ticket_id:
            return ticket_file, record
    return None


# Erstellt aus einem Datensatz die editierbaren Ticketfelder mit sinnvollen Fallbacks.
def build_editable_ticket(record: dict[str, Any]) -> dict[str, str]:
    ticket = record.get("ticket", {})
    email = record.get("email", {})

    editable = {
        "Title": _as_text(ticket.get("Title") or email.get("subject")),
        "Area": normalize_area_value(ticket.get("Area")),
        "Iteration": _as_text(ticket.get("Iteration")),
        "Description": _as_text(ticket.get("Description") or email.get("body_cleaned") or email.get("body")),
        "Ticket-Type": _as_text(ticket.get("Ticket-Type")),
        "Environment": _as_text(ticket.get("Environment")),
        "Prio": _as_text(ticket.get("Prio")),
        "Impact": _as_text(ticket.get("Impact")),
    }
    return editable


# Bereitet die Modellvorhersagen inklusive Konfidenzen und Alternativen für die Detailansicht auf.
def build_classification_overview(record: dict[str, Any]) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []

    for classifier_key, payload in record.get("classification", {}).items():
        top_3 = payload.get("top_3", []) or []
        alternatives = [item.get("label", "") for item in top_3[1:3] if isinstance(item, dict)]
        predicted_value = _as_text(payload.get("label"))
        if classifier_key == "ticket_area":
            predicted_value = normalize_area_value(predicted_value)

        rows.append(
            {
                "Modell": CLASSIFICATION_LABELS.get(classifier_key, classifier_key),
                "Vorhersage": predicted_value,
                "Konfidenz": float(payload.get("softmax_confidence", 0.0) or 0.0),
                "Alternative 1": normalize_area_value(alternatives[0]) if classifier_key == "ticket_area" and len(alternatives) > 0 else (alternatives[0] if len(alternatives) > 0 else ""),
                "Alternative 2": normalize_area_value(alternatives[1]) if classifier_key == "ticket_area" and len(alternatives) > 1 else (alternatives[1] if len(alternatives) > 1 else ""),
                "Modellpfad": _as_text(payload.get("model_dir")),
            }
        )

    return rows


# Sammelt alle auswählbaren Feldoptionen aus festen Vorgaben und vorhandenen Ticketwerten.
def collect_options(index_rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    options: dict[str, list[str]] = {key: list(values) for key, values in FIXED_OPTION_MAP.items()}

    for row in index_rows:
        for key in options:
            value = _as_text(row.get(key)).strip()
            if key == "area":
                value = normalize_area_value(value)
            if value and value not in options[key]:
                options[key].append(value)

    for key, values in options.items():
        options[key] = sorted(values)

    return options


# Übernimmt manuelle Änderungen an einem Ticket, protokolliert sie und speichert den Datensatz.
def update_ticket_record(
    ticket_id: str,
    updated_ticket: dict[str, Any],
    ticket_dir: Path | None = None,
) -> dict[str, dict[str, str]]:
    loaded = load_ticket_record_by_id(ticket_id, ticket_dir=ticket_dir)
    if loaded is None:
        raise FileNotFoundError(f"Ticket mit der ID {ticket_id} wurde nicht gefunden.")

    target_path, record = loaded
    ticket = record.setdefault("ticket", {})
    original_ticket = {field: normalize_ticket_field(field, ticket.get(field)) for field in EDITABLE_TICKET_FIELDS}
    changed_fields: dict[str, dict[str, str]] = {}

    for field in EDITABLE_TICKET_FIELDS:
        new_value = normalize_ticket_field(field, updated_ticket.get(field))
        old_value = original_ticket.get(field, "")
        if old_value != new_value:
            changed_fields[field] = {"old": old_value, "new": new_value}
            ticket[field] = new_value

    if not changed_fields:
        return changed_fields

    manual_review = record.setdefault("manual_review", {})
    manual_review.setdefault("initial_ticket", original_ticket)
    history = manual_review.setdefault("history", [])
    history.append(
        {
            "edited_at_utc": utc_now_iso(),
            "source": "streamlit_ui",
            "changed_fields": changed_fields,
        }
    )
    manual_review["last_saved_at_utc"] = utc_now_iso()

    meta = record.setdefault("meta", {})
    meta["last_updated_by_ui_at_utc"] = utc_now_iso()

    write_json_atomic(target_path, record)
    return changed_fields


# Kennzeichnet ausgewählte Tickets als an RPA übergeben und verschiebt sie in die RPA-Inbox.
def move_tickets_to_rpa_inbox(
    ticket_ids: list[str],
    *,
    ticket_dir: Path | None = None,
    target_dir: Path | None = None,
) -> dict[str, list[dict[str, str]]]:
    source_directory = ensure_ticket_directory(ticket_dir)
    destination_directory = ensure_rpa_inbox_directory(target_dir)

    moved: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []

    for ticket_id in ticket_ids:
        try:
            loaded = load_ticket_record_by_id(ticket_id, ticket_dir=source_directory)
            if loaded is None:
                raise FileNotFoundError(f"Ticket mit der ID {ticket_id} wurde nicht gefunden.")

            source_path, record = loaded
            destination_path = destination_directory / source_path.name
            if destination_path.exists():
                raise FileExistsError(
                    f"Im Zielordner existiert bereits eine Datei mit dem Namen {destination_path.name}."
                )

            submitted_at_utc = utc_now_iso()
            meta = record.setdefault("meta", {})
            status = record.setdefault("status", {})
            timing = record.setdefault("timing", {})
            meta["submitted_to_rpa_at_utc"] = submitted_at_utc
            meta["rpa_target_path"] = str(destination_directory)
            timing["submitted_to_rpa_at_utc"] = submitted_at_utc
            status["submitted_to_rpa"] = True

            write_json_atomic(source_path, record)
            shutil.move(str(source_path), str(destination_path))

            moved.append(
                {
                    "ticket_id": ticket_id,
                    "file_name": source_path.name,
                    "source_path": str(source_path),
                    "target_path": str(destination_path),
                }
            )
        except Exception as error:
            errors.append(
                {
                    "ticket_id": ticket_id,
                    "error": str(error),
                }
            )

    return {"moved": moved, "errors": errors}
