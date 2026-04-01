from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from config import TICKETS_DIR
from streamlit_ticket_repository import (
    build_classification_overview,
    build_editable_ticket,
    collect_options,
    load_ticket_index,
    load_ticket_record_by_id,
    update_ticket_record,
)

OVERVIEW_PAGE = None
DETAIL_PAGE = None


@st.cache_data(show_spinner=False)
def get_ticket_index_cached(signature: tuple[tuple[str, int], ...]) -> list[dict[str, Any]]:
    del signature
    return load_ticket_index(TICKETS_DIR)


def build_inventory_signature() -> tuple[tuple[str, int], ...]:
    directory = Path(TICKETS_DIR)
    directory.mkdir(parents=True, exist_ok=True)

    return tuple(
        sorted((file_path.name, file_path.stat().st_mtime_ns) for file_path in directory.glob("TICKET-*.json"))
    )


def get_ticket_index() -> list[dict[str, Any]]:
    return get_ticket_index_cached(build_inventory_signature())


def format_timestamp(value: str) -> str:
    if not value:
        return "-"

    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return value

    return parsed.tz_convert("Europe/Berlin").strftime("%d.%m.%Y %H:%M")


def format_confidence(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.1%}"


def build_display_dataframe(rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_df = pd.DataFrame(rows)
    if raw_df.empty:
        raw_df = pd.DataFrame(
            columns=[
                "ticket_id",
                "title",
                "sender",
                "received_utc",
                "ticket_type",
                "area",
                "priority",
                "impact",
                "average_confidence",
                "manually_edited",
                "description_preview",
            ]
        )

    display_df = pd.DataFrame(
        {
            "Titel": raw_df["title"],
            "Absender": raw_df["sender"],
            "Empfangen": raw_df["received_utc"].apply(format_timestamp),
            "Typ": raw_df["ticket_type"],
            "Bereich": raw_df["area"],
            "Priorität": raw_df["priority"],
            "Impact": raw_df["impact"],
            "Ø Konfidenz": raw_df["average_confidence"].apply(format_confidence),
            "Manuell geaendert": raw_df["manually_edited"].map({True: "Ja", False: "Nein"}),
            "Beschreibung": raw_df["description_preview"],
        }
    )

    return raw_df.reset_index(drop=True), display_df.reset_index(drop=True)


def apply_filters(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    st.sidebar.header("Filter")
    search_value = st.sidebar.text_input("Suche", placeholder="Titel, Absender oder Beschreibung")

    all_types = sorted({row.get("ticket_type", "") for row in rows if row.get("ticket_type")})
    all_areas = sorted({row.get("area", "") for row in rows if row.get("area")})
    all_priorities = sorted({row.get("priority", "") for row in rows if row.get("priority")})

    selected_types = st.sidebar.multiselect("Ticket-Typ", options=all_types)
    selected_areas = st.sidebar.multiselect("Bereich", options=all_areas)
    selected_priorities = st.sidebar.multiselect("Priorität", options=all_priorities)
    only_manual = st.sidebar.toggle("Nur manuell geaenderte Tickets", value=False)
    min_confidence = st.sidebar.slider("Minimale Ø Konfidenz", 0.0, 1.0, 0.0, 0.05)

    filtered_rows: list[dict[str, Any]] = []
    normalized_search = search_value.strip().lower()

    for row in rows:
        haystack = " ".join(
            [
                str(row.get("title", "")),
                str(row.get("sender", "")),
                str(row.get("area", "")),
                str(row.get("ticket_type", "")),
                str(row.get("description", "")),
            ]
        ).lower()

        if normalized_search and normalized_search not in haystack:
            continue
        if selected_types and row.get("ticket_type") not in selected_types:
            continue
        if selected_areas and row.get("area") not in selected_areas:
            continue
        if selected_priorities and row.get("priority") not in selected_priorities:
            continue
        if only_manual and not row.get("manually_edited"):
            continue

        avg_confidence = row.get("average_confidence")
        if avg_confidence is not None and float(avg_confidence) < min_confidence:
            continue

        filtered_rows.append(row)

    return filtered_rows


def render_overview_page() -> None:
    rows = get_ticket_index()

    st.title("KI Ticket Pilot")
    st.caption(
        "Übersicht über alle klassifizierten Tickets aus dem Postfach inklusive Modellkonfidenzen und manueller Nachbearbeitung."
    )

    if not rows:
        st.info(
            f"Im Verzeichnis {Path(TICKETS_DIR)} wurden noch keine TICKET-JSON-Dateien gefunden. "
        )
        return

    filtered_rows = apply_filters(rows)
    raw_df, display_df = build_display_dataframe(filtered_rows)

    metric_columns = st.columns(4)
    average_confidence = raw_df["average_confidence"].dropna().mean() if not raw_df.empty else None
    low_confidence_count = int((raw_df["average_confidence"].fillna(1.0) < 0.70).sum()) if not raw_df.empty else 0
    manual_count = int(raw_df["manually_edited"].sum()) if not raw_df.empty else 0

    metric_columns[0].metric("Gefilterte Tickets", len(raw_df))
    metric_columns[1].metric("Ø Konfidenz", format_confidence(average_confidence) if average_confidence is not None else "-")
    metric_columns[2].metric("Manuell geändert", manual_count)
    metric_columns[3].metric("Unter 70% Konfidenz", low_confidence_count)

    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.subheader("Verteilung nach Ticket-Typ")
        type_counts = raw_df["ticket_type"].replace("", "Unbekannt").value_counts()
        st.bar_chart(type_counts)

    with chart_right:
        st.subheader("Verteilung nach Bereich")
        area_counts = raw_df["area"].replace("", "Unbekannt").value_counts().head(10)
        st.bar_chart(area_counts)

    st.subheader("Ticketliste")
    event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="ticket_overview_table",
        column_config={
            "Titel": st.column_config.TextColumn(width="large"),
            "Absender": st.column_config.TextColumn(width="medium"),
            "Empfangen": st.column_config.TextColumn(width="small"),
            "Typ": st.column_config.TextColumn(width="medium"),
            "Bereich": st.column_config.TextColumn(width="medium"),
            "Priorität": st.column_config.TextColumn(width="small"),
            "Impact": st.column_config.TextColumn(width="small"),
            "Ø Konfidenz": st.column_config.TextColumn(width="small"),
            "Manuell geändert": st.column_config.TextColumn(width="small"),
            "Beschreibung": st.column_config.TextColumn(width="large"),
        },
    )

    selected_rows = event.selection.rows
    if len(selected_rows) == 1:
        selected_row = raw_df.iloc[selected_rows[0]]
        info_col, action_col = st.columns([4, 1])
        info_col.info(
            f"Ausgewählt: {selected_row['title']} | Typ: {selected_row['ticket_type'] or '-'} | Bereich: {selected_row['area'] or '-'}"
        )
        if action_col.button("Ticket öffnen", use_container_width=True, type="primary"):
            st.session_state["selected_ticket_id"] = selected_row["ticket_id"]
            st.switch_page(DETAIL_PAGE)


def render_detail_page() -> None:
    rows = get_ticket_index()
    flash_message = st.session_state.pop("ticket_flash_message", None)
    selected_ticket_id = st.session_state.get("selected_ticket_id")

    title_col, action_col = st.columns([4, 1])
    title_col.title("Ticketdetail und Nachbearbeitung")
    title_col.caption("Pflichtfelder prüfen, bei Bedarf korrigieren und als JSON-Datei abspeichern.")

    if action_col.button("Zur Übersicht", use_container_width=True):
        st.switch_page(OVERVIEW_PAGE)

    if flash_message:
        st.success(flash_message)

    if not selected_ticket_id:
        st.warning("Es wurde noch kein Ticket aus der Übersicht ausgewählt.")
        return

    loaded = load_ticket_record_by_id(selected_ticket_id, TICKETS_DIR)
    if loaded is None:
        st.error(f"Das Ticket mit der ID {selected_ticket_id} wurde nicht gefunden.")
        return

    source_path, record = loaded
    editable_ticket = build_editable_ticket(record)
    option_map = collect_options(rows)

    email = record.get("email", {})
    meta = record.get("meta", {})
    ticket = record.get("ticket", {})

    meta_columns = st.columns(4)
    meta_columns[0].metric("Message-ID", meta.get("message_id", "-"))
    meta_columns[1].metric("Empfangen", format_timestamp(email.get("received_utc", "")))
    meta_columns[2].metric("Absender", email.get("sender", "-"))
    meta_columns[3].metric("Datei", source_path.name)

    confidence_rows = build_classification_overview(record)
    if confidence_rows:
        confidence_df = pd.DataFrame(confidence_rows)
        score_columns = st.columns(len(confidence_rows))
        for column, confidence_row in zip(score_columns, confidence_rows):
            column.metric(confidence_row["Modell"], format_confidence(float(confidence_row["Konfidenz"])))
    else:
        confidence_df = pd.DataFrame(columns=["Modell", "Vorhersage", "Konfidenz", "Alternative 1", "Alternative 2", "Modellpfad"])

    st.subheader("Vorausgefüllte Pflichtfelder")
    with st.form("ticket_edit_form"):
        first_row = st.columns(2)
        title_value = first_row[0].text_input("Title", value=editable_ticket["Title"])
        area_options = option_map["area"]
        area_value = first_row[1].selectbox(
            "Area",
            options=[editable_ticket["Area"], *[value for value in area_options if value != editable_ticket["Area"]]] or [""],
        )

        second_row = st.columns(2)
        ticket_type_value = second_row[0].selectbox(
            "Ticket-Type",
            options=[editable_ticket["Ticket-Type"], *[value for value in option_map["ticket_type"] if value != editable_ticket["Ticket-Type"]]] or [""],
        )
        iteration_value = second_row[1].text_input("Iteration", value=editable_ticket["Iteration"])

        third_row = st.columns(3)
        environment_value = third_row[0].selectbox(
            "Environment",
            options=[editable_ticket["Environment"], *[value for value in option_map["environment"] if value != editable_ticket["Environment"]]] or [""],
        )
        priority_value = third_row[1].selectbox(
            "Prio",
            options=[editable_ticket["Prio"], *[value for value in option_map["priority"] if value != editable_ticket["Prio"]]] or [""],
        )
        impact_value = third_row[2].selectbox(
            "Impact",
            options=[editable_ticket["Impact"], *[value for value in option_map["impact"] if value != editable_ticket["Impact"]]] or [""],
        )

        description_value = st.text_area("Description", value=editable_ticket["Description"], height=260)

        submitted = st.form_submit_button("Änderungen speichern", type="primary", use_container_width=True)

    if submitted:
        changed_fields = update_ticket_record(
            ticket_id=selected_ticket_id,
            updated_ticket={
                "Title": title_value,
                "Area": area_value,
                "Iteration": iteration_value,
                "Description": description_value,
                "Ticket-Type": ticket_type_value,
                "Environment": environment_value,
                "Prio": priority_value,
                "Impact": impact_value,
            },
            ticket_dir=TICKETS_DIR,
        )
        get_ticket_index_cached.clear()
        if changed_fields:
            field_list = ", ".join(changed_fields.keys())
            st.session_state["ticket_flash_message"] = f"Ticket gespeichert. Gäenderte Felder: {field_list}"
        else:
            st.session_state["ticket_flash_message"] = "Es wurden keine Änderungen erkannt."
        st.rerun()

    expander_left, expander_right = st.columns(2)

    with expander_left:
        with st.expander("Modellvorhersagen und Konfidenzen", expanded=True):
            st.dataframe(
                confidence_df.assign(Konfidenz=confidence_df["Konfidenz"].apply(format_confidence)),
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("Originale Mail", expanded=False):
            st.markdown(f"**Betreff:** {email.get('subject', '-')}")
            st.markdown(f"**Absender:** {email.get('sender', '-')}")
            st.markdown(f"**Empfangen:** {format_timestamp(email.get('received_utc', ''))}")
            st.text_area("Mail-Body", value=email.get("body", ""), height=320, disabled=True)

    with expander_right:
        with st.expander("Bereinigter Text für die Klassifikation", expanded=False):
            st.text_area(
                "Text for classification",
                value=email.get("text_for_classification", ""),
                height=320,
                disabled=True,
            )

        with st.expander("Aktueller Ticketzustand", expanded=False):
            st.json(ticket)

        manual_review = record.get("manual_review", {})
        history = manual_review.get("history", [])
        if history:
            with st.expander("Manuelle Änderungshistorie", expanded=False):
                history_rows = []
                for entry in history:
                    for field_name, payload in entry.get("changed_fields", {}).items():
                        history_rows.append(
                            {
                                "Zeitpunkt": format_timestamp(entry.get("edited_at_utc", "")),
                                "Feld": field_name,
                                "Alt": payload.get("old", ""),
                                "Neu": payload.get("new", ""),
                            }
                        )
                st.dataframe(pd.DataFrame(history_rows), use_container_width=True, hide_index=True)


st.set_page_config(
    page_title="KI Ticket Pilot",
    page_icon=":material/confirmation_number:",
    layout="wide",
    initial_sidebar_state="expanded",
)

OVERVIEW_PAGE = st.Page(render_overview_page, title="Ticketübersicht", icon=":material/dashboard:", default=True)
DETAIL_PAGE = st.Page(
    render_detail_page,
    title="Ticketdetail",
    icon=":material/edit_note:",
    url_path="ticketdetail",
    visibility="hidden",
)

navigation = st.navigation([OVERVIEW_PAGE, DETAIL_PAGE])
navigation.run()
