# KI-RPA-Pipeline zur automatisierten Ticketvorbefüllung

## Kurzbeschreibung
Dieses Projekt ist ein Prototyp zur automatisierten Vorverarbeitung eingehender E-Mails im Auftragsmanagement. Der Prototyp liest E-Mails aus einem definierten Outlook-Ordner aus, bereitet Betreff und Nachrichtentext für die Klassifikation auf, klassifiziert den Inhalt mit mehreren Transformer-Modellen und erzeugt daraus strukturierte Ticket-JSON-Dateien. Anschließend können die erzeugten Tickets in einer Streamlit-Oberfläche geprüft, bei Bedarf manuell korrigiert und in eine RPA-Inbox zur weiteren Verarbeitung übergeben werden.

Der Prototyp wurde im Rahmen einer Bachelorarbeit entwickelt und verfolgt das Ziel, die manuelle Erstbefüllung von Tickets für die weitere Bearbeitung in Azure DevOps Server zu unterstützen.

## Schnellstart
1. `config.py` an die lokale Umgebung anpassen.
2. Trainierte Modelle in den in `CLASSIFIER_MODELS` hinterlegten Verzeichnissen bereitstellen.
3. Die Pipeline mit `python main.py --mode all` oder direkt über die Streamlit-Oberfläche starten.
4. Die erzeugten Ticketdateien in Streamlit prüfen, bei Bedarf korrigieren und anschließend an die RPA-Inbox übergeben.

## Inhaltsverzeichnis
1. [Schnellstart](#schnellstart)
2. [Funktionsumfang](#funktionsumfang)
3. [Technologiestack](#technologiestack)
4. [Projektstruktur](#projektstruktur)
5. [Voraussetzungen](#voraussetzungen)
6. [Installation](#installation)
7. [Konfiguration](#konfiguration)
8. [Ausführung der Pipeline](#ausführung-der-pipeline)
9. [Streamlit-Oberfläche](#streamlit-oberfläche)
10. [Modelltraining](#modelltraining)
11. [Modellevaluation](#modellevaluation)
12. [Einzelvorhersage testen](#einzelvorhersage-testen)
13. [Ablage und Ausgabedateien](#ablage-und-ausgabedateien)
14. [Bekannte Einschränkungen](#bekannte-einschränkungen)
15. [Weiterentwicklung](#weiterentwicklung)

## Funktionsumfang
Der Prototyp umfasst die folgenden Funktionen:

- Auslesen von E-Mails aus einem konfigurierten Outlook-Postfach beziehungsweise Unterordner.
- Vorverarbeitung der E-Mail-Texte durch Normalisierung von Betreff und Body.
- Entfernung typischer Anrede- und Schlussformeln zur Reduktion von Rauschen.
- Klassifikation in vier Ticketdimensionen:
  - Ticket-Typ
  - Bereich
  - Priorität
  - Schweregrad
- Automatische Vorbefüllung zentraler Ticketfelder.
- Ableitung der Zieliteration anhand eines konfigurierten Release-Kalenders.
- Speicherung der E-Mails, Tickets und Fehlerberichte als JSON-Dateien.
- Nachbearbeitung der Tickets in einer Streamlit-Oberfläche.
- Anzeige von Modellkonfidenzen und alternativen Vorhersagen.
- Protokollierung manueller Änderungen im Ticketdatensatz.
- Übergabe ausgewählter Tickets an eine RPA-Inbox für die weitere automatisierte Verarbeitung.

## Technologiestack
Der Prototyp basiert auf den folgenden Technologien und Bibliotheken:

- Python
- Streamlit
- Pandas
- Altair
- PyTorch
- Hugging Face Transformers
- Datasets
- scikit-learn
- Optuna für optionale Hyperparameter-Suche
- pywin32 für den Outlook-Zugriff unter Windows

## Projektstruktur
Eine mögliche Projektstruktur sieht wie folgt aus:

```text
project-root/
├── config.py
├── main.py
├── outlook_reader.py
├── preprocessing.py
├── storage.py
├── streamlit_ticket_repository.py
├── streamlit_ticket_ui.py
├── classification/
│   ├── train_ticket_classifier.py
│   ├── evaluate_ticket_classifier.py
│   └── predict_ticket_classifier.py
├── data/
│   ├── emails_inbox/
│   ├── tickets/
│   ├── errors/
│   ├── state/
│   └── processed/
└── models/
    ├── ticket-type/
    ├── ticket-area/
    ├── ticket-impact/
    └── ticket-prio/
```

## Voraussetzungen
Für die Ausführung des Prototyps sollten die folgenden Voraussetzungen erfüllt sein:

- Windows-System für den Outlook-Zugriff über COM.
- Installierte Outlook-Desktop-Anwendung mit Zugriff auf das konfigurierte Postfach.
- Python-Umgebung mit allen benötigten Paketen.
- Lokal verfügbare, trainierte Klassifikationsmodelle im Verzeichnis `models/`.
- Schreibzugriff auf die lokalen Datenverzeichnisse sowie auf die konfigurierte RPA-Inbox.

Wichtig: Für den reinen Modus `classify` wird kein Outlook-Zugriff benötigt. Für das Auslesen von E-Mails ist dagegen `pywin32` erforderlich.

## Installation
Nach dem Klonen oder Kopieren des Projekts kann eine virtuelle Umgebung eingerichtet und die benötigte Software installiert werden.

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install streamlit pandas altair torch transformers datasets scikit-learn pywin32 openpyxl optuna
```

Je nach Hardware und Python-Version kann es sinnvoll sein, PyTorch separat und passend zur Zielumgebung zu installieren.

## Konfiguration
Die zentrale Konfiguration erfolgt in `config.py`. Vor der ersten Ausführung sollten insbesondere die folgenden Werte geprüft und an die Zielumgebung angepasst werden:

### Outlook und Mailabruf
- `MAILBOX_SMTP`: Name oder SMTP-Adresse des auszulesenden Postfachs
- `TARGET_FOLDER`: Outlook-Ordner, aus dem die E-Mails gelesen werden
- `MAX_MESSAGES`: maximale Anzahl auszulesender Nachrichten je Lauf
- `UNREAD_ONLY`: steuert, ob nur ungelesene E-Mails verarbeitet werden
- `MARK_AS_READ`: optionales Kennzeichen für den weiteren Umgang mit gelesenen Mails

### Modellpfade
Unter `CLASSIFIER_MODELS` werden die lokalen Modellverzeichnisse für die vier Klassifikatoren hinterlegt:

- `ticket_type`
- `ticket_area`
- `ticket_impact`
- `ticket_priority`

Alle dort referenzierten Modellordner müssen ein lauffähiges Transformers-Modell enthalten.

### Ticketzielstruktur
- `DEFAULT_ENVIRONMENT`: Standardumgebung für die Ticketvorbefüllung
- `ITERATION_ROOT`: Wurzel für die automatische Iterationsableitung
- `RELEASE_CUTOFF_DAYS`: Vorlauf in Tagen zur Release-Zuordnung
- `RELEASE_PRODUCTIVE_DATES`: produktive Releasetermine pro Jahr

### Verzeichnisse
- `DATA_DIR`: lokaler Datenbereich des Prototyps
- `RPA_INBOX_DIR`: Zielordner für JSON-Dateien, die vom RPA-Bot weiterverarbeitet werden

### Feste Auswahlwerte für die Oberfläche
In `config.py` sind außerdem feste Auswahllisten für Priorität, Schweregrad, Umgebung, Ticket-Typ und Bereich hinterlegt. Diese Werte werden in der Streamlit-Oberfläche als Auswahloptionen verwendet.

## Ausführung der Pipeline
Die Pipeline kann über `main.py` in drei Modi ausgeführt werden.

### Gesamte Pipeline ausführen
```bash
python main.py --mode all
```

Dieser Modus:
1. liest neue E-Mails aus Outlook,
2. speichert sie als Inbox-JSON-Dateien,
3. klassifiziert alle noch nicht verarbeiteten E-Mails,
4. erzeugt daraus Ticket-JSON-Dateien.

### Nur E-Mails abrufen
```bash
python main.py --mode fetch
```

Dieser Modus liest nur neue E-Mails aus Outlook und speichert sie im Inbox-Verzeichnis.

### Nur vorhandene Inbox-Dateien klassifizieren
```bash
python main.py --mode classify
```

Dieser Modus eignet sich insbesondere dann, wenn bereits gespeicherte E-Mail-JSON-Dateien erneut oder unabhängig vom Outlook-Zugriff verarbeitet werden sollen.

## Streamlit-Oberfläche
Die Streamlit-Anwendung dient als Prüf- und Nachbearbeitungsoberfläche für die erzeugten Tickets.

Start der Oberfläche:

```bash
streamlit run streamlit_ticket_ui.py
```

Die Oberfläche bietet unter anderem folgende Funktionen:

- Start der Pipeline direkt aus der UI heraus
- Übersicht über alle erzeugten Ticketdateien
- Filter nach Ticket-Typ, Bereich, Priorität und Konfidenz
- Kennzahlen zur Verteilung und zur durchschnittlichen Modellkonfidenz
- Detailansicht eines Tickets mit allen vorausgefüllten Pflichtfeldern
- Anzeige von Modellvorhersagen, Top-Alternativen und Konfidenzen
- manuelle Korrektur einzelner Ticketfelder
- Historisierung manueller Änderungen
- Verschieben freigegebener Tickets in die RPA-Inbox

## Modelltraining
Für das Training neuer Klassifikationsmodelle steht `train_ticket_classifier.py` zur Verfügung.

Das Skript unterstützt zwei Betriebsarten:

1. Training auf Basis **einer einzelnen Datei**, die intern in Train, Validierung und Test aufgeteilt wird.
2. Training auf Basis **vorgefertigter Split-Dateien** für Train, Validierung und Test.

### Beispiel mit einer einzelnen Datei
```bash
python train_ticket_classifier.py \
  --data <PFAD_ZUR_DATEI> \
  --label-col Typ \
  --text-cols Titel Beschreibung \
  --model <BASISMODELL_ODER_HF_MODELLNAME> \
  --output-dir models/ticket-type/<MODELLNAME> \
  --sheet-name Tabelle \
  --use-class-weights
```

### Beispiel mit separaten Splits
```bash
python train_ticket_classifier.py \
  --train-data <TRAIN_DATEI> \
  --val-data <VAL_DATEI> \
  --test-data <TEST_DATEI> \
  --label-col Bereich \
  --text-cols Titel Beschreibung \
  --model <BASISMODELL_ODER_HF_MODELLNAME> \
  --output-dir models/ticket-area/<MODELLNAME>
```

### Wichtige Trainingsoptionen
- `--max-length`: maximale Tokenlänge
- `--epochs`: Anzahl der Trainingsepochen
- `--lr`: Lernrate
- `--train-batch-size`: Batchgröße im Training
- `--eval-batch-size`: Batchgröße in der Evaluation
- `--grad-accum`: Gradient Accumulation Steps
- `--weight-decay`: Weight Decay
- `--seed`: Reproduzierbarkeit
- `--use-class-weights`: gewichtete Verlustfunktion bei Klassenungleichgewicht
- `--do-hpo`: Aktivierung der Hyperparameter-Suche mit Optuna
- `--hpo-trials`: Anzahl der HPO-Durchläufe

Während des Trainings werden unter anderem Early Stopping, Validierung pro Epoche und die Auswahl des besten Modells anhand der Macro-F1 verwendet.

## Modellevaluation
Zur separaten Evaluation eines bereits trainierten Modells kann `evaluate_ticket_classifier.py` genutzt werden.

```bash
python evaluate_ticket_classifier.py \
  --test-data <TEST_DATEI> \
  --label-col Typ \
  --model-dir models/ticket-type/<MODELLNAME> \
  --sheet-name Tabelle
```

Als Ergebnis werden im Modellverzeichnis unter anderem folgende Dateien erzeugt:

- `metrics_test.json`
- `classification_report.json`

## Einzelvorhersage testen
Für schnelle Einzeltests steht das Vorhersageskript zur Verfügung.

```bash
python classification/predict_ticket_classifier.py \
  --model-dir models/ticket-type/<MODELLNAME> \
  --text "Beispieltext für die Klassifikation"
```

Die Ausgabe enthält das vorhergesagte Label, die Softmax-Konfidenz, Wahrscheinlichkeiten und die Top-3-Ergebnisse.

## Ablage und Ausgabedateien
Der Prototyp arbeitet dateibasiert und legt seine Zwischen- und Ergebnisstände in JSON-Dateien ab.

### `data/emails_inbox/`
Hier werden aus Outlook eingelesene E-Mails als strukturierte JSON-Dateien gespeichert.

### `data/tickets/`
Hier werden klassifizierte Tickets als Ticket-JSON-Dateien gespeichert. Diese enthalten unter anderem:

- E-Mail-Metadaten
- bereinigten Klassifikationstext
- vorausgefüllte Ticketfelder
- Modellvorhersagen und Konfidenzen
- technische Metadaten
- optionale Informationen zur manuellen Nachbearbeitung

### `data/errors/`
Hier werden Fehlerberichte im JSON-Format abgelegt, zum Beispiel bei Problemen im Outlook-Zugriff oder während der Klassifikation.

### `data/state/`
Hier werden Statusdateien gepflegt, mit deren Hilfe bereits verarbeitete E-Mails erkannt werden.

### Vereinfachte Struktur einer Ticketdatei
```json
{
  "email": {
    "subject": "...",
    "sender": "...",
    "received_utc": "...",
    "body": "...",
    "body_cleaned": "...",
    "text_for_classification": "..."
  },
  "ticket": {
    "Title": "...",
    "Area": "...",
    "Iteration": "...",
    "Description": "...",
    "Ticket-Type": "...",
    "Environment": "...",
    "Prio": "...",
    "Impact": "..."
  },
  "classification": {
    "ticket_type": { "label": "...", "softmax_confidence": 0.0 },
    "ticket_area": { "label": "...", "softmax_confidence": 0.0 },
    "ticket_priority": { "label": "...", "softmax_confidence": 0.0 },
    "ticket_impact": { "label": "...", "softmax_confidence": 0.0 }
  },
  "meta": {
    "message_id": "...",
    "ticket_created_at_utc": "..."
  }
}
```

## Bekannte Einschränkungen
Der aktuelle Prototyp ist bewusst praxisnah, aber noch nicht als vollständig produktionsreifes System ausgelegt. Zu den wichtigsten Einschränkungen gehören:

- Der Outlook-Zugriff ist an eine Windows-Umgebung mit installiertem Outlook gebunden.
- Die automatische Iterationslogik setzt gepflegte Releasetermine in `config.py` voraus.
- Die Klassifikation funktioniert nur, wenn alle referenzierten Modellordner lokal vorhanden und vollständig sind.
- Die Qualität der Vorhersagen hängt unmittelbar von Trainingsdaten und Modellwahl ab.
- Die erzeugten Tickets sind für eine fachliche Prüfung vorgesehen und sollten vor der endgültigen Ticketanlage validiert werden.
- Der dateibasierte Ansatz eignet sich gut für einen Prototyp, ist aber bei wachsendem Volumen nur begrenzt skalierbar.

## Weiterentwicklung
Mögliche Weiterentwicklungen des Prototyps sind unter anderem:

- direkte technische Anbindung an Azure DevOps Server statt dateibasierter Übergabe
- Ausbau eines Monitoring- und Logging-Konzepts
- Versionierung und zentrale Verwaltung der Modelle
- automatisierte Tests für Pipeline, Vorverarbeitung und UI

---

## Autor und Kontext

Dieses Repository dokumentiert einen im Rahmen einer Bachelorarbeit entwickelten Prototypen zur automatisierten Vorverarbeitung eingehender E-Mails und zur strukturierten Ticketvorbefüllung für Azure DevOps Server.

**Autor:** Alexander Beine
**Institution:** Duale Hochschule Baden-Württemberg Stuttgart (DHBW Stuttgart)  
**Kontext:** Bachelorarbeit im Studiengang Wirtschaftsinformatik