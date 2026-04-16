import argparse
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter

from config import (
    DEFAULT_ENVIRONMENT,
    ITERATION_ROOT,
    RELEASE_CUTOFF_DAYS,
    RELEASE_PRODUCTIVE_DATES,
)
from preprocessing import preprocess_email
from storage import (
    append_stored_email_id,
    append_ticketed_id,
    iter_email_json_files,
    load_json,
    load_stored_email_ids,
    load_ticketed_ids,
    save_email_json,
    save_error_report,
    save_ticket_json,
    utc_now_iso,
)
from classification.predict_ticket_classifier import (
    build_predicted_ticket,
    classify_email_text,
    load_classifiers,
)

VALID_PIPELINE_MODES = {"all", "fetch", "classify"}


# Liest den Ausführungsmodus der Pipeline aus den Kommandozeilenargumenten ein.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=sorted(VALID_PIPELINE_MODES),
        default="all",
        help="all = Outlook lesen und anschliessend inbox JSON klassifizieren, fetch = nur Outlook nach emails_inbox, classify = nur vorhandene inbox JSON verarbeiten",
    )
    return parser.parse_args()


# Wandelt einen ISO-Zeitstempel in ein UTC-Datetime-Objekt um.
def parse_utc_datetime(value: str) -> datetime:
    if not value:
        raise ValueError("Kein received_utc vorhanden.")

    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


# Erzeugt aus Jahr und Release-Nummer den vollständigen Iterationspfad.
def build_iteration_name(year: int, release_number: int) -> str:
    return f"{ITERATION_ROOT}\\REL{year}_{release_number}"


# Ermittelt anhand des Empfangszeitpunkts die passende Zieliteration aus dem Release-Kalender.
def determine_iteration(received_utc: str) -> str:
    received_dt = parse_utc_datetime(received_utc)
    year = received_dt.year

    if year not in RELEASE_PRODUCTIVE_DATES:
        raise ValueError(f"Kein Release-Kalender für das Jahr {year} konfiguriert.")

    productive_dates = RELEASE_PRODUCTIVE_DATES[year]

    for release_number, productive_date_str in enumerate(productive_dates, start=1):
        productive_dt = datetime.fromisoformat(productive_date_str).replace(tzinfo=timezone.utc)
        cutoff_dt = productive_dt - timedelta(days=RELEASE_CUTOFF_DAYS)

        if received_dt < cutoff_dt:
            return build_iteration_name(year, release_number)

    next_year = year + 1
    if next_year not in RELEASE_PRODUCTIVE_DATES:
        raise ValueError(
            f"Bitte den Release-Kalender für {next_year} in config.py ergänzen."
        )

    return build_iteration_name(next_year, 1)


# Baut aus E-Mail-Daten und Klassifikation die Ticketfelder für das Zielsystem auf.
def build_ticket(inbox_record: dict, processed: dict, predicted_ticket: dict) -> dict:
    email = inbox_record.get("email", {})
    received_utc = email.get("received_utc", utc_now_iso())

    cleaned_description = processed.get("body_cleaned", "")
    if not cleaned_description.strip():
        cleaned_description = email.get("body", "")

    return {
        "Title": email.get("subject", ""),
        "Area": predicted_ticket.get("area", ""),
        "Iteration": determine_iteration(received_utc),
        "Description": cleaned_description,
        "Ticket-Type": predicted_ticket.get("type", ""),
        "Environment": DEFAULT_ENVIRONMENT,
        "Prio": predicted_ticket.get("priority", ""),
        "Impact": predicted_ticket.get("impact", ""),
    }


# Überführt eine gelesene E-Mail in das interne Inbox-JSON-Format.
def build_inbox_record(email: dict) -> dict:
    return {
        "email": {
            "subject": email.get("subject", ""),
            "sender": email.get("sender", ""),
            "received_utc": email.get("received_utc", utc_now_iso()),
            "body": email.get("body", ""),
        },
        "meta": {
            "source": "outlook_desktop",
            "message_id": email.get("message_id", "UNKNOWN_MESSAGE_ID"),
            "stored_at_utc": utc_now_iso(),
        },
        "status": {
            "stored_in_inbox": True,
            "ticket_created": False,
        },
        "timing": dict(email.get("timing", {})),
    }


# Erstellt den vollständigen Ticket-Datensatz mit E-Mail-, Vorverarbeitungs-, Klassifikations- und Metadaten.
def build_ticket_record(
    inbox_record: dict,
    source_email_path: Path,
    processed: dict,
    classifications: dict,
    predicted_ticket: dict,
    timing: dict | None = None,
) -> dict:
    return {
        "email": {
            "subject": inbox_record.get("email", {}).get("subject", ""),
            "subject_cleaned": processed.get("subject_cleaned", ""),
            "sender": inbox_record.get("email", {}).get("sender", ""),
            "received_utc": inbox_record.get("email", {}).get("received_utc", utc_now_iso()),
            "body": inbox_record.get("email", {}).get("body", ""),
            "body_cleaned": processed.get("body_cleaned", ""),
            "text_for_classification": processed.get("text_for_classification", ""),
        },
        "ticket": build_ticket(inbox_record, processed, predicted_ticket),
        "classification": classifications,
        "preprocessing": processed.get("preprocessing", {}),
            "timing": timing or {},
        "meta": {
            "source": inbox_record.get("meta", {}).get("source", "outlook_desktop"),
            "message_id": inbox_record.get("meta", {}).get("message_id", "UNKNOWN_MESSAGE_ID"),
            "source_email_file": str(source_email_path),
            "ticket_created_at_utc": utc_now_iso(),
        },
    }

# Erstellt einen standardisierten Fehlerreport mit Zeitstempel, Kontext und Traceback.
def build_error_report(stage: str, message_id: str, error: Exception, context: dict | None = None) -> dict:
    return {
        "timestamp_utc": utc_now_iso(),
        "stage": stage,
        "message_id": message_id,
        "error": {
            "type": error.__class__.__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        },
        "context": context or {},
    }


# Liest neue E-Mails aus Outlook, überspringt bereits bekannte Nachrichten und speichert neue Inbox-Dateien.
def fetch_and_store_new_emails() -> dict:
    try:
        from outlook_reader import fetch_emails
    except ModuleNotFoundError as error:
        if error.name == "win32com":
            raise ModuleNotFoundError(
                "Das Paket 'pywin32' ist nicht installiert. Fuer Outlook-Zugriff bitte in der aktiven venv 'python -m pip install pywin32' ausfuehren. Fuer 'python main.py --mode classify' wird Outlook nicht benoetigt."
            ) from error
        raise

    stored_ids = load_stored_email_ids()

    summary = {
        "read": 0,
        "stored": 0,
        "skipped": 0,
        "errors": 0,
    }

    try:
        emails = fetch_emails()
        summary["read"] = len(emails)
        print(f"[INFO] {summary['read']} Mail(s) aus Outlook ausgelesen.")
    except Exception as error:
        summary["errors"] += 1
        report = build_error_report(
            stage="fetch_mailbox",
            message_id="OUTLOOK_FETCH_STAGE",
            error=error,
            context={},
        )
        error_path = save_error_report(report)
        print(f"[ERROR][FETCH] Outlook konnte nicht gelesen werden: {error} -> {error_path.name}")
        return summary

    for email in emails:
        message_id = email.get("message_id", "UNKNOWN_MESSAGE_ID")

        try:
            if message_id in stored_ids:
                summary["skipped"] += 1
                print(f"[SKIP][FETCH] Bereits in emails_inbox vorhanden: {message_id}")
                continue

            inbox_record = build_inbox_record(email)
            target_path = save_email_json(inbox_record)
            append_stored_email_id(message_id)
            stored_ids.add(message_id)

            summary["stored"] += 1
            print(f"[OK][FETCH] Gespeichert: {target_path.name}")

        except Exception as error:
            summary["errors"] += 1
            report = build_error_report(
                stage="fetch_to_inbox",
                message_id=message_id,
                error=error,
                context={
                    "email_preview": {
                        "subject": email.get("subject", ""),
                        "sender": email.get("sender", ""),
                        "received_utc": email.get("received_utc", ""),
                    }
                },
            )
            error_path = save_error_report(report)
            print(f"[ERROR][FETCH] {message_id}: {error} -> {error_path.name}")

    return summary


# Bereitet Betreff und Inhalt für die Klassifikation auf.
def get_processed_payload(inbox_record: dict) -> dict:
    email = inbox_record.get("email", {})
    subject = email.get("subject", "")
    body = email.get("body", "")

    processed = preprocess_email(subject, body)

    existing_text = email.get("text_for_classification", "")
    if existing_text and processed.get("text_for_classification", "").strip() == "":
        processed["subject_cleaned"] = email.get("subject_cleaned", processed.get("subject_cleaned", ""))
        processed["body_cleaned"] = email.get("body_cleaned", processed.get("body_cleaned", ""))
        processed["text_for_classification"] = existing_text
        processed["preprocessing"] = inbox_record.get("preprocessing", processed.get("preprocessing", {}))

    return processed


# Klassifiziert alle noch nicht verarbeiteten Inbox-Dateien und speichert daraus Ticket-JSON-Dateien.
def classify_pending_emails() -> dict:
    ticketed_ids = load_ticketed_ids()
    email_files = list(iter_email_json_files())

    summary = {
        "checked": len(email_files),
        "ticketed": 0,
        "skipped": 0,
        "errors": 0,
    }

    try:
        classifiers = load_classifiers()
        print(f"[INFO] {len(classifiers)} Klassifikator(en) geladen.")
    except Exception as error:
        summary["errors"] += 1
        report = build_error_report(
            stage="load_classifiers",
            message_id="CLASSIFIER_LOAD_STAGE",
            error=error,
            context={},
        )
        error_path = save_error_report(report)
        print(f"[ERROR][CLASSIFY] Klassifikatoren konnten nicht geladen werden: {error} -> {error_path.name}")
        return summary

    for email_file in email_files:
        message_id = "UNKNOWN_MESSAGE_ID"

        try:
            inbox_record = load_json(email_file)
            message_id = inbox_record.get("meta", {}).get("message_id", "UNKNOWN_MESSAGE_ID")
            received_utc = inbox_record.get("email", {}).get("received_utc", utc_now_iso())

            classification_started_at_utc = utc_now_iso()
            classification_started = perf_counter()

            preprocessing_started = perf_counter()
            processed = get_processed_payload(inbox_record)
            preprocessing_seconds = round(perf_counter() - preprocessing_started, 6)

            if message_id in ticketed_ids:
                summary["skipped"] += 1
                print(f"[SKIP][CLASSIFY] Ticket bereits vorhanden: {message_id}")
                continue
            
            processed = get_processed_payload(inbox_record)
            text_for_classification = processed.get("text_for_classification", "")

            if not text_for_classification.strip():
                raise ValueError("Der klassifizierbare Text ist leer.")
            
            inference_started = perf_counter()
            classifications = classify_email_text(text_for_classification, classifiers)
            inference_seconds = round(perf_counter() - inference_started, 6)
            predicted_ticket = build_predicted_ticket(classifications)

            timing = dict(inbox_record.get("timing", {}))
            timing["classification_started_at_utc"] = classification_started_at_utc
            timing["preprocessing_duration_seconds"] = preprocessing_seconds
            timing["model_inference_duration_seconds"] = inference_seconds
            timing["classification_finished_at_utc"] = utc_now_iso()
            timing["classification_duration_seconds"] = round(
                perf_counter() - classification_started, 6
            )

            ticket_record = build_ticket_record(
                inbox_record=inbox_record,
                source_email_path=email_file,
                processed=processed,
                classifications=classifications,
                predicted_ticket=predicted_ticket,
                timing=timing,
            )

            ticket_path = save_ticket_json(
                ticket_record,
                received_utc=received_utc,
                message_id=message_id,
            )
            append_ticketed_id(message_id)
            ticketed_ids.add(message_id)

            summary["ticketed"] += 1
            print(f"[OK][CLASSIFY] Ticket gespeichert: {ticket_path.name}")

        except Exception as error:
            summary["errors"] += 1
            report = build_error_report(
                stage="classification",
                message_id=message_id,
                error=error,
                context={
                    "source_email_file": str(email_file),
                },
            )
            error_path = save_error_report(report)
            print(f"[ERROR][CLASSIFY] {message_id}: {error} -> {error_path.name}")

    return summary


# Gibt eine kompakte Zusammenfassung der Fetch- und Klassifikationsstufe in der Konsole aus.
def print_summary(fetch_summary: dict, classification_summary: dict) -> None:
    print("\n--- Zusammenfassung Fetch-Stufe ---")
    print(f"Ausgelesen:      {fetch_summary['read']}")
    print(f"Gespeichert:     {fetch_summary['stored']}")
    print(f"Übersprungen:   {fetch_summary['skipped']}")
    print(f"Fehler:          {fetch_summary['errors']}")

    print("\n--- Zusammenfassung Klassifikations-Stufe ---")
    print(f"Geprüft:        {classification_summary['checked']}")
    print(f"Ticket erstellt: {classification_summary['ticketed']}")
    print(f"Übersprungen:   {classification_summary['skipped']}")
    print(f"Fehler:          {classification_summary['errors']}")


# Führt die Pipeline abhängig vom gewählten Modus aus und liefert die Ergebnisübersicht zurück.
def run_pipeline(mode: str = "all") -> dict:
    if mode not in VALID_PIPELINE_MODES:
        valid_modes = ", ".join(sorted(VALID_PIPELINE_MODES))
        raise ValueError(f"Ungültiger Modus '{mode}'. Erlaubt sind: {valid_modes}.")

    fetch_summary = {"read": 0, "stored": 0, "skipped": 0, "errors": 0}
    classification_summary = {"checked": 0, "ticketed": 0, "skipped": 0, "errors": 0}

    if mode in {"all", "fetch"}:
        fetch_summary = fetch_and_store_new_emails()

    if mode in {"all", "classify"}:
        classification_summary = classify_pending_emails()

    return {
        "mode": mode,
        "fetch": fetch_summary,
        "classification": classification_summary,
        "finished_at_utc": utc_now_iso(),
    }


def main() -> None:
    args = parse_args()
    pipeline_result = run_pipeline(args.mode)
    print_summary(pipeline_result["fetch"], pipeline_result["classification"])


if __name__ == "__main__":
    main()
