"""
Dash Analytics Dashboard - Clinical Compliance Analytics + Medical Advisor Chat
Professional dark-themed dashboard with interactive cards, charts, LLM selector, and chatbot.
"""

import json
import os
import sys
import uuid
from pathlib import Path

import dash
from dash import dcc, html, dash_table, Input, Output, State, callback_context
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import requests

sys.path.append(str(Path(__file__).parent.parent))
from config import config

# --- Data Loading ---

API_BASE = os.getenv("API_BASE_URL", config.API_BASE_URL)


def load_batch_results(path="outputs/batch_results.json"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def analyze_all_notes_via_api(api_base=None):
    api_base = api_base or API_BASE
    with open(config.CLINICAL_NOTES_PATH, "r") as f:
        notes = json.load(f)
    results = []
    for note in notes:
        try:
            resp = requests.post(
                f"{api_base}/analyze_note",
                json={"note_id": note["note_id"], "text": note["text"]},
                timeout=120,
            )
            if resp.status_code == 200:
                results.append(resp.json())
            else:
                results.append({"note_id": note["note_id"], "status": "error", "error": resp.text})
        except Exception as e:
            results.append({"note_id": note["note_id"], "status": "error", "error": str(e)})
    return {"total": len(notes), "results": results}


def compute_status(row):
    """Compute display status from compliance data."""
    if row.get("alerts_count", 0) > 0:
        return "CRITICAL"
    score = row.get("compliance_score", 0)
    if score >= 0.8:
        return "Compliant"
    elif score >= 0.5:
        return "Partially Compliant"
    return "Non-Compliant"


def results_to_dataframe(batch_data):
    rows = []
    for r in batch_data.get("results", []):
        row = {
            "note_id": r.get("note_id", ""),
            "status": r.get("status", "error"),
            "diagnosis": r.get("entities", {}).get("diagnosis", "N/A"),
            "medications": ", ".join(r.get("entities", {}).get("medications", [])),
            "compliance_score": r.get("compliance", {}).get("overall_score", 0.0),
            "medication_score": r.get("compliance", {}).get("medication_score", 0.0),
            "contraindication_score": r.get("compliance", {}).get("contraindication_score", 0.0),
            "test_score": r.get("compliance", {}).get("test_score", 0.0),
            "alerts_count": len(r.get("alerts", [])),
            "recommendations_count": len(r.get("recommendations", [])),
            "guideline_disease": r.get("guidelines", {}).get("disease", "N/A"),
            "required_tests": ", ".join(r.get("guidelines", {}).get("required_tests", [])),
            "match_confidence": r.get("guidelines", {}).get("match_confidence", 0.0),
        }
        row["display_status"] = compute_status(row)
        rows.append(row)
    return pd.DataFrame(rows)


# --- Color Palette ---

C = {
    "bg": "#0f172a",
    "card": "#1e293b",
    "accent": "#2563eb",
    "green": "#10b981",
    "red": "#ef4444",
    "yellow": "#f59e0b",
    "purple": "#8b5cf6",
    "text": "#f1f5f9",
    "muted": "#94a3b8",
    "border": "#334155",
    "input_bg": "#0f172a",
    "chat_user": "#1e40af",
    "hover": "#334155",
}

FONT = "'Inter', 'Segoe UI', sans-serif"

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", size=12),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor="#1e293b"),
    yaxis=dict(gridcolor="#1e293b"),
)

# --- App Setup ---

app = dash.Dash(
    __name__,
    title="Clinical AI Advisor",
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

# Custom CSS for dark theme dropdowns and components
app.index_string = '''<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .Select-control { background-color: #1e293b !important; border-color: #334155 !important; color: #f1f5f9 !important; }
            .Select-value-label { color: #f1f5f9 !important; }
            .Select-placeholder { color: #94a3b8 !important; }
            .Select-menu-outer { background-color: #1e293b !important; border-color: #334155 !important; }
            .VirtualizedSelectOption { color: #f1f5f9 !important; background-color: #1e293b !important; }
            .VirtualizedSelectOption:hover { background-color: #334155 !important; }
            .Select-arrow { border-color: #94a3b8 transparent transparent !important; }
            .is-focused .Select-control { border-color: #2563eb !important; box-shadow: none !important; }
            .clickable-card { transition: all 0.2s ease; cursor: pointer; }
            .clickable-card:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
            textarea:focus, input:focus { outline: none; border-color: #2563eb !important; }
            .dash-table-container .dash-filter input { color: #f1f5f9 !important; background-color: #0f172a !important; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>'''

_initial_data = load_batch_results()
_initial_status = ""
if _initial_data:
    _initial_status = f"Loaded {_initial_data.get('total', 0)} results"

# --- Guidelines Sidebar ---


def _render_guidelines_sidebar():
    try:
        with open(config.GUIDELINES_PATH, "r") as f:
            guidelines = json.load(f)
    except Exception:
        return html.P("Guidelines not loaded", style={"color": C["red"], "fontSize": "12px"})
    items = []
    for disease, info in guidelines.items():
        rec = ", ".join(info.get("recommended_drugs", [])) or "-"
        avoid = ", ".join(info.get("avoid_drugs", [])) or "-"
        tests = ", ".join(info.get("required_tests", [])) or "-"
        items.append(
            html.Div(
                style={"padding": "8px", "borderRadius": "6px", "backgroundColor": C["bg"],
                       "marginBottom": "6px", "border": f"1px solid {C['border']}"},
                children=[
                    html.Div(disease.title(), style={"fontWeight": "600", "fontSize": "12px",
                                                      "color": C["accent"], "marginBottom": "3px"}),
                    html.Div(f"Rx: {rec}", style={"fontSize": "11px", "color": C["muted"]}),
                    html.Div(f"Avoid: {avoid}", style={"fontSize": "11px",
                                                        "color": C["red"] if avoid != "-" else C["muted"]}),
                    html.Div(f"Tests: {tests}", style={"fontSize": "11px", "color": C["muted"]}),
                ],
            )
        )
    return html.Div(items)


# --- Helpers ---


def _card(card_id, title, value, color, icon=""):
    """Clickable summary card."""
    return html.Div(
        id=card_id, n_clicks=0, className="clickable-card",
        style={"padding": "20px", "backgroundColor": C["card"], "borderRadius": "12px",
               "borderLeft": f"4px solid {color}", "border": f"1px solid {C['border']}",
               "borderLeftWidth": "4px", "borderLeftColor": color},
        children=[
            html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"}, children=[
                html.Div(title, style={"color": C["muted"], "fontSize": "13px", "fontWeight": "500"}),
                html.Span(icon, style={"fontSize": "18px"}) if icon else html.Div(),
            ]),
            html.Div(id=f"{card_id}-value", children=str(value),
                      style={"color": color, "fontSize": "28px", "fontWeight": "700", "marginTop": "8px"}),
        ],
    )


def _chat_bubble(role, text, guideline_details=None):
    is_user = role == "doctor"
    children = [
        html.Div("You" if is_user else "Clinical AI Advisor",
                  style={"fontSize": "11px", "fontWeight": "600",
                         "color": C["accent"] if not is_user else C["muted"], "marginBottom": "4px"}),
        html.Div(text, style={"whiteSpace": "pre-wrap"}),
    ]
    # Dynamic guideline reference tags for bot responses
    if not is_user and guideline_details:
        tags = []
        for disease, details in guideline_details.items():
            parts = [html.Span(disease.title(), style={"fontWeight": "600", "color": C["accent"]})]
            drugs = ", ".join(details.get("recommended_drugs", []))
            contra = ", ".join(details.get("contraindicated_drugs", []))
            tests = ", ".join(details.get("required_tests", []))
            if drugs:
                parts.append(html.Span(f"  Rx: {drugs}", style={"color": C["green"]}))
            if contra:
                parts.append(html.Span(f"  Avoid: {contra}", style={"color": C["red"]}))
            if tests:
                parts.append(html.Span(f"  Tests: {tests}", style={"color": C["yellow"]}))
            tags.append(html.Div(
                parts,
                style={"padding": "6px 10px", "backgroundColor": C["card"], "borderRadius": "6px",
                       "border": f"1px solid {C['border']}", "fontSize": "11px"},
            ))
        if tags:
            children.append(html.Div(
                [html.Div("Referenced Guidelines:", style={"fontSize": "10px", "color": C["muted"],
                                                            "fontWeight": "600", "marginBottom": "4px"})] + tags,
                style={"marginTop": "10px", "display": "flex", "flexDirection": "column", "gap": "4px"},
            ))

    return html.Div(
        style={"display": "flex", "justifyContent": "flex-end" if is_user else "flex-start"},
        children=[html.Div(
            style={"maxWidth": "80%", "padding": "12px 16px", "borderRadius": "12px",
                   "backgroundColor": C["chat_user"] if is_user else C["bg"],
                   "border": f"1px solid {C['border']}" if not is_user else "none",
                   "fontSize": "14px", "lineHeight": "1.6"},
            children=children,
        )],
    )


# --- Layout ---

app.layout = html.Div(
    style={"fontFamily": FONT, "backgroundColor": C["bg"], "minHeight": "100vh", "color": C["text"]},
    children=[
        # ===== NAVBAR =====
        html.Div(
            style={"background": f"linear-gradient(135deg, {C['card']}, {C['bg']})", "padding": "12px 32px",
                    "display": "flex", "justifyContent": "space-between", "alignItems": "center",
                    "borderBottom": f"1px solid {C['border']}", "position": "sticky", "top": "0", "zIndex": "100"},
            children=[
                # Left: Logo
                html.Div(style={"display": "flex", "alignItems": "center", "gap": "12px"}, children=[
                    html.Div("+", style={
                        "fontSize": "22px", "width": "40px", "height": "40px",
                        "display": "flex", "alignItems": "center", "justifyContent": "center",
                        "backgroundColor": "#1d4ed8", "borderRadius": "10px",
                        "color": "white", "fontWeight": "800", "lineHeight": "1",
                    }),
                    html.Div([
                        html.H1("Clinical AI Advisor", style={
                            "margin": "0", "fontSize": "20px", "fontWeight": "700", "letterSpacing": "-0.5px"}),
                        html.Span("Medical Decision Support", style={
                            "fontSize": "11px", "color": C["muted"], "letterSpacing": "0.5px"}),
                    ]),
                ]),
                # Right: LLM Selector
                html.Div(style={"display": "flex", "alignItems": "center", "gap": "12px"}, children=[
                    html.Span("LLM Provider:", style={"color": C["muted"], "fontSize": "12px", "fontWeight": "500"}),
                    dcc.Dropdown(
                        id="llm-selector",
                        options=[
                            {"label": "Llama 3 (Local)", "value": "ollama-llama3"},
                            {"label": "Mistral (Local)", "value": "ollama-mistral"},
                            {"label": "Meditron (Local)", "value": "ollama-meditron"},
                            {"label": "Gemini (Google)", "value": "gemini"},
                            {"label": "Claude (Anthropic)", "value": "claude"},
                            {"label": "GPT-4 (OpenAI)", "value": "openai"},
                        ],
                        value=config.LLM_PROVIDER,
                        clearable=False,
                        style={"width": "200px", "fontSize": "13px"},
                    ),
                    html.Span(id="provider-status", style={"color": C["green"], "fontSize": "11px", "minWidth": "80px"}),
                ]),
            ],
        ),

        # ===== TABS =====
        html.Div(
            style={"padding": "0 32px", "backgroundColor": C["card"], "borderBottom": f"1px solid {C['border']}"},
            children=[
                dcc.Tabs(id="main-tabs", value="analytics", children=[
                    dcc.Tab(label="Analytics Dashboard", value="analytics",
                            style={"padding": "14px 24px", "backgroundColor": "transparent", "border": "none",
                                   "color": C["muted"], "fontWeight": "500", "fontSize": "14px"},
                            selected_style={"padding": "14px 24px", "backgroundColor": "transparent", "border": "none",
                                            "borderBottom": f"2px solid {C['accent']}", "color": C["accent"],
                                            "fontWeight": "600", "fontSize": "14px"}),
                    dcc.Tab(label="Medical Advisor Chat", value="chatbot",
                            style={"padding": "14px 24px", "backgroundColor": "transparent", "border": "none",
                                   "color": C["muted"], "fontWeight": "500", "fontSize": "14px"},
                            selected_style={"padding": "14px 24px", "backgroundColor": "transparent", "border": "none",
                                            "borderBottom": f"2px solid {C['green']}", "color": C["green"],
                                            "fontWeight": "600", "fontSize": "14px"}),
                ], style={"height": "auto"}),
            ],
        ),

        # ===== ANALYTICS CONTENT (always in DOM, toggled) =====
        html.Div(id="analytics-content", style={"padding": "24px 32px"}, children=[
            # Controls
            html.Div(style={"display": "flex", "gap": "12px", "marginBottom": "24px", "alignItems": "center"}, children=[
                html.Button("Reload Results", id="btn-load", style={
                    "padding": "10px 20px", "cursor": "pointer", "backgroundColor": C["accent"],
                    "color": "white", "border": "none", "borderRadius": "8px", "fontWeight": "500", "fontSize": "14px"}),
                html.Button("Refresh via API", id="btn-refresh", style={
                    "padding": "10px 20px", "cursor": "pointer", "backgroundColor": C["green"],
                    "color": "white", "border": "none", "borderRadius": "8px", "fontWeight": "500", "fontSize": "14px"}),
                html.Div(id="load-status", children=_initial_status,
                          style={"color": C["green"], "fontSize": "14px", "marginLeft": "12px"}),
            ]),

            # Summary Cards (clickable)
            html.Div(id="summary-cards-row", style={
                "display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "16px", "marginBottom": "16px",
            }, children=[
                _card("card-notes", "Notes Analyzed", "--", C["accent"]),
                _card("card-compliance", "Avg Compliance", "--", C["green"]),
                _card("card-alerts", "Contraindication Alerts", "--", C["red"]),
                _card("card-missing", "Missing Tests", "--", C["yellow"]),
            ]),

            # Active filter indicator
            html.Div(id="filter-indicator", style={"display": "none", "marginBottom": "16px"}, children=[
                html.Div(style={"display": "flex", "alignItems": "center", "gap": "10px",
                                "padding": "10px 16px", "backgroundColor": C["card"], "borderRadius": "8px",
                                "border": f"1px solid {C['border']}"}, children=[
                    html.Span(id="filter-text", children="", style={"fontSize": "13px", "color": C["text"]}),
                    html.Button("Clear Filter", id="btn-clear-filter", n_clicks=0, style={
                        "padding": "4px 12px", "cursor": "pointer", "backgroundColor": "transparent",
                        "color": C["muted"], "border": f"1px solid {C['border']}", "borderRadius": "6px",
                        "fontSize": "12px"}),
                ]),
            ]),

            # Charts Row 1
            html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px", "marginBottom": "24px"}, children=[
                html.Div(dcc.Graph(id="compliance-histogram"),
                          style={"backgroundColor": C["card"], "borderRadius": "12px", "padding": "8px",
                                 "border": f"1px solid {C['border']}"}),
                html.Div(dcc.Graph(id="compliance-by-disease"),
                          style={"backgroundColor": C["card"], "borderRadius": "12px", "padding": "8px",
                                 "border": f"1px solid {C['border']}"}),
            ]),
            # Charts Row 2
            html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px", "marginBottom": "24px"}, children=[
                html.Div(dcc.Graph(id="test-completion-chart"),
                          style={"backgroundColor": C["card"], "borderRadius": "12px", "padding": "8px",
                                 "border": f"1px solid {C['border']}"}),
                html.Div(dcc.Graph(id="alerts-chart"),
                          style={"backgroundColor": C["card"], "borderRadius": "12px", "padding": "8px",
                                 "border": f"1px solid {C['border']}"}),
            ]),

            # Table section with status filter
            html.Div(style={"backgroundColor": C["card"], "borderRadius": "12px", "padding": "20px",
                            "border": f"1px solid {C['border']}"}, children=[
                html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
                                "marginBottom": "16px"}, children=[
                    html.H3("Clinical Notes Analysis", style={"color": C["text"], "margin": "0", "fontSize": "18px"}),
                    html.Div(style={"display": "flex", "alignItems": "center", "gap": "10px"}, children=[
                        html.Span("Filter by Status:", style={"color": C["muted"], "fontSize": "12px"}),
                        dcc.Dropdown(
                            id="status-filter",
                            options=[
                                {"label": "All Statuses", "value": "all"},
                                {"label": "Compliant", "value": "Compliant"},
                                {"label": "Partially Compliant", "value": "Partially Compliant"},
                                {"label": "CRITICAL", "value": "CRITICAL"},
                                {"label": "Non-Compliant", "value": "Non-Compliant"},
                            ],
                            value="all",
                            clearable=False,
                            style={"width": "200px", "fontSize": "13px"},
                        ),
                    ]),
                ]),
                html.Div(id="notes-table-container"),
            ]),
            html.Div(id="detail-panel", style={"marginTop": "16px"}),
        ]),

        # ===== CHAT CONTENT (always in DOM, toggled) =====
        html.Div(id="chat-content", style={"display": "none", "padding": "24px 32px"}, children=[
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 320px", "gap": "24px",
                       "height": "calc(100vh - 160px)"},
                children=[
                    # Chat Panel
                    html.Div(style={"display": "flex", "flexDirection": "column", "backgroundColor": C["card"],
                                    "borderRadius": "12px", "border": f"1px solid {C['border']}",
                                    "overflow": "hidden"}, children=[
                        # Header
                        html.Div(style={"padding": "16px 20px", "borderBottom": f"1px solid {C['border']}",
                                        "display": "flex", "justifyContent": "space-between",
                                        "alignItems": "center"}, children=[
                            html.Div([
                                html.H3("Medical Advisor", style={"margin": "0", "fontSize": "16px", "fontWeight": "600"}),
                                html.P("Describe a clinical scenario for guideline-based recommendations",
                                       style={"margin": "2px 0 0 0", "fontSize": "12px", "color": C["muted"]}),
                            ]),
                            html.Button("Clear Chat", id="btn-clear-chat", style={
                                "padding": "6px 14px", "cursor": "pointer", "backgroundColor": "transparent",
                                "color": C["muted"], "border": f"1px solid {C['border']}", "borderRadius": "6px",
                                "fontSize": "12px"}),
                        ]),
                        # Messages
                        html.Div(id="chat-messages", style={
                            "flex": "1", "overflowY": "auto", "padding": "20px",
                            "display": "flex", "flexDirection": "column", "gap": "16px",
                        }, children=[
                            _chat_bubble("advisor",
                                         "Welcome! I'm your Clinical AI Advisor. I can help with:\n\n"
                                         "- Analyzing clinical scenarios against guidelines\n"
                                         "- Recommending appropriate medications\n"
                                         "- Flagging contraindicated drugs\n"
                                         "- Suggesting required diagnostic tests\n\n"
                                         "Describe a patient scenario and I'll provide recommendations."),
                        ]),
                        # Input area
                        html.Div(style={"padding": "16px 20px", "borderTop": f"1px solid {C['border']}"}, children=[
                            html.Div(style={"display": "flex", "gap": "10px", "alignItems": "flex-end"}, children=[
                                dcc.Textarea(
                                    id="chat-input",
                                    placeholder="Describe a clinical scenario for guideline-based advice...",
                                    style={
                                        "flex": "1", "minHeight": "70px", "maxHeight": "150px", "resize": "vertical",
                                        "padding": "14px 16px", "backgroundColor": C["input_bg"],
                                        "border": f"1px solid {C['border']}", "borderRadius": "10px",
                                        "color": C["text"], "fontSize": "14px", "fontFamily": FONT,
                                        "lineHeight": "1.5",
                                    },
                                ),
                                html.Button("Send", id="btn-send-chat", style={
                                    "padding": "14px 28px", "cursor": "pointer",
                                    "backgroundColor": "#1d4ed8",
                                    "color": "white", "border": "none", "borderRadius": "10px",
                                    "fontWeight": "600", "fontSize": "14px", "minHeight": "48px",
                                    "whiteSpace": "nowrap",
                                }),
                            ]),
                        ]),
                    ]),
                    # Sidebar
                    html.Div(style={"backgroundColor": C["card"], "borderRadius": "12px",
                                    "border": f"1px solid {C['border']}", "padding": "16px",
                                    "overflowY": "auto"}, children=[
                        html.H4("Guidelines Reference", style={"margin": "0 0 12px 0", "fontSize": "14px",
                                                                 "fontWeight": "600"}),
                        html.P("Clinical guidelines knowledge base:", style={
                            "fontSize": "12px", "color": C["muted"], "margin": "0 0 12px 0"}),
                        _render_guidelines_sidebar(),
                        html.Hr(style={"borderColor": C["border"], "margin": "16px 0"}),
                        html.H4("Quick Prompts", style={"margin": "0 0 8px 0", "fontSize": "14px",
                                                         "fontWeight": "600"}),
                        html.Div(style={"display": "flex", "flexDirection": "column", "gap": "6px"}, children=[
                            html.Div(p, style={"padding": "8px", "backgroundColor": C["bg"],
                                               "border": f"1px solid {C['border']}", "borderRadius": "6px",
                                               "color": C["muted"], "fontSize": "11px", "cursor": "pointer"})
                            for p in [
                                "45yo male with fever, cough. Diagnosed pneumonia. On amoxicillin.",
                                "30yo female with dysuria. UTI suspected. Considering ciprofloxacin.",
                                "60yo male with chest pain. MI suspected. What tests and drugs?",
                                "Child with ear pain and fever. Recommended treatment?",
                                "Patient with type 2 diabetes. Medication and tests?",
                            ]
                        ]),
                    ]),
                ],
            ),
        ]),

        # ===== STORES =====
        dcc.Store(id="results-store", data=_initial_data),
        dcc.Store(id="chat-session-id", data=f"session_{uuid.uuid4().hex[:12]}"),
        dcc.Store(id="chat-history-store", data=[]),
        dcc.Store(id="active-filter", data="all"),
        # Auto-refresh: check for new batch results every 15 seconds
        dcc.Interval(id="auto-refresh-interval", interval=15_000, n_intervals=0),
    ],
)


# ===== CALLBACKS =====


# --- Tab Switching ---
@app.callback(
    [Output("analytics-content", "style"), Output("chat-content", "style")],
    Input("main-tabs", "value"),
)
def toggle_tabs(tab):
    show = {"display": "block", "padding": "24px 32px"}
    hide = {"display": "none"}
    if tab == "chatbot":
        return hide, show
    return show, hide


# --- Data Loading ---
_last_batch_mtime = 0  # Track file modification time for auto-refresh


@app.callback(
    [Output("results-store", "data"), Output("load-status", "children")],
    [Input("btn-load", "n_clicks"), Input("btn-refresh", "n_clicks"),
     Input("auto-refresh-interval", "n_intervals")],
    State("results-store", "data"),
    prevent_initial_call=True,
)
def load_data(load_clicks, refresh_clicks, n_intervals, current_data):
    global _last_batch_mtime
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update

    trigger = ctx.triggered[0]["prop_id"]

    # Auto-refresh: only reload if file has changed
    if "auto-refresh-interval" in trigger:
        batch_path = "outputs/batch_results.json"
        try:
            mtime = os.path.getmtime(batch_path)
            if mtime > _last_batch_mtime:
                _last_batch_mtime = mtime
                data = load_batch_results(batch_path)
                if data and data.get("results"):
                    total = data.get("total", 0)
                    success = data.get("success", sum(1 for r in data.get("results", []) if r.get("status") != "error"))
                    return data, f"Auto-loaded {success}/{total} results"
            return dash.no_update, dash.no_update
        except (FileNotFoundError, OSError):
            return dash.no_update, dash.no_update

    if "btn-load" in trigger:
        data = load_batch_results()
        if data:
            return data, f"Loaded {data.get('total', 0)} results"
        return None, "No results found. Run batch analysis first."
    elif "btn-refresh" in trigger:
        try:
            data = analyze_all_notes_via_api()
            with open("outputs/batch_results.json", "w") as f:
                json.dump(data, f, indent=2)
            return data, f"Analyzed {data.get('total', 0)} notes via API"
        except Exception as e:
            return None, f"API error: {str(e)}"
    return dash.no_update, dash.no_update


# --- Card Values ---
@app.callback(
    [Output("card-notes-value", "children"), Output("card-compliance-value", "children"),
     Output("card-alerts-value", "children"), Output("card-missing-value", "children")],
    Input("results-store", "data"),
)
def update_card_values(data):
    if not data or not data.get("results"):
        return "--", "--", "--", "--"
    df = results_to_dataframe(data)
    ok = df[df["status"] != "error"]
    if ok.empty:
        return "0", "--", "0", "0"
    avg = ok["compliance_score"].mean()
    return (
        str(len(ok)),
        f"{avg:.0%}",
        str(int((ok["alerts_count"] > 0).sum())),
        str(int((ok["test_score"] < 1.0).sum())),
    )


# --- Card Click Filter ---
@app.callback(
    [Output("active-filter", "data"), Output("filter-indicator", "style"), Output("filter-text", "children")],
    [Input("card-alerts", "n_clicks"), Input("card-missing", "n_clicks"), Input("btn-clear-filter", "n_clicks")],
    prevent_initial_call=True,
)
def update_card_filter(alert_clicks, missing_clicks, clear_clicks):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update
    trigger = ctx.triggered[0]["prop_id"]
    show = {"display": "block", "marginBottom": "16px"}
    hide = {"display": "none", "marginBottom": "16px"}

    if "card-alerts" in trigger:
        return "alerts", show, "Showing: Notes with Contraindication Alerts"
    elif "card-missing" in trigger:
        return "missing_tests", show, "Showing: Notes with Missing Tests"
    elif "btn-clear-filter" in trigger:
        return "all", hide, ""
    return "all", hide, ""


# --- Charts + Table ---
@app.callback(
    [Output("compliance-histogram", "figure"), Output("compliance-by-disease", "figure"),
     Output("test-completion-chart", "figure"), Output("alerts-chart", "figure"),
     Output("notes-table-container", "children")],
    [Input("results-store", "data"), Input("active-filter", "data"), Input("status-filter", "value")],
)
def update_charts_and_table(data, active_filter, status_filter):
    empty = go.Figure()
    empty.update_layout(
        title="No data available",
        annotations=[{"text": "Click 'Reload Results' or 'Refresh via API'", "showarrow": False,
                       "font": {"size": 14, "color": "#94a3b8"}}],
        **CHART_LAYOUT,
    )
    no_data = (empty, empty, empty, empty, html.P("No data loaded.", style={"color": C["muted"]}))

    if not data or not data.get("results"):
        return no_data

    df = results_to_dataframe(data)
    ok = df[df["status"] != "error"].copy()
    if ok.empty:
        return no_data

    # Apply card filter
    filtered = ok.copy()
    if active_filter == "alerts":
        filtered = filtered[filtered["alerts_count"] > 0]
    elif active_filter == "missing_tests":
        filtered = filtered[filtered["test_score"] < 1.0]

    # Apply status filter
    if status_filter and status_filter != "all":
        filtered = filtered[filtered["display_status"] == status_filter]

    if filtered.empty:
        empty_msg = go.Figure()
        empty_msg.update_layout(
            title="No matching results",
            annotations=[{"text": "Try changing the filter", "showarrow": False,
                           "font": {"size": 14, "color": "#94a3b8"}}],
            **CHART_LAYOUT,
        )
        return (empty_msg, empty_msg, empty_msg, empty_msg,
                html.P("No notes match the current filter.", style={"color": C["muted"]}))

    # --- Chart 1: Compliance Histogram ---
    fig1 = px.histogram(filtered, x="compliance_score", nbins=10, title="Compliance Score Distribution",
                        labels={"compliance_score": "Score"}, color_discrete_sequence=[C["accent"]])
    fig1.update_layout(**CHART_LAYOUT, bargap=0.1)

    # --- Chart 2: Compliance by Disease ---
    da = filtered.groupby("guideline_disease")["compliance_score"].agg(["mean", "count"]).reset_index()
    da.columns = ["Disease", "Avg Score", "Count"]
    da = da.sort_values("Avg Score")
    fig2 = px.bar(da, x="Disease", y="Avg Score", title="Compliance by Disease", color="Avg Score",
                  color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"], text="Count")
    fig2.update_layout(**CHART_LAYOUT, coloraxis_showscale=False)

    # --- Chart 3: Test Completion Rate (FIXED) ---
    # Show average test completion rate per disease instead of total vs missing
    td = filtered[filtered["required_tests"] != ""].copy()
    if not td.empty:
        tc = td.groupby("guideline_disease").agg(
            avg_completion=("test_score", "mean"),
            note_count=("note_id", "count"),
            req_tests=("required_tests", "first"),
        ).reset_index()
        tc["num_required"] = tc["req_tests"].apply(lambda x: len(x.split(", ")) if x else 0)
        tc["avg_ordered"] = tc["avg_completion"] * tc["num_required"]
        tc["avg_missing"] = tc["num_required"] - tc["avg_ordered"]

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=tc["guideline_disease"], y=tc["avg_ordered"],
            name="Avg Tests Ordered", marker_color=C["accent"],
        ))
        fig3.add_trace(go.Bar(
            x=tc["guideline_disease"], y=tc["avg_missing"],
            name="Avg Tests Missing", marker_color=C["red"],
        ))
        fig3.update_layout(
            title="Required Tests: Avg Ordered vs Missing per Note",
            barmode="stack", **CHART_LAYOUT,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
    else:
        fig3 = go.Figure()
        fig3.update_layout(title="No test data", **CHART_LAYOUT)

    # --- Chart 4: Alerts by Disease ---
    ad = filtered.groupby("guideline_disease")["alerts_count"].sum().reset_index()
    ad.columns = ["Disease", "Alerts"]
    ad = ad[ad["Alerts"] > 0].sort_values("Alerts", ascending=False)
    if not ad.empty:
        fig4 = px.bar(ad, x="Disease", y="Alerts", title="Contraindication Alerts by Disease",
                      color="Alerts", color_continuous_scale=["#f59e0b", "#ef4444"])
        fig4.update_layout(**CHART_LAYOUT, coloraxis_showscale=False)
    else:
        fig4 = go.Figure()
        fig4.update_layout(
            title="No Alerts",
            annotations=[{"text": "No contraindication alerts found", "showarrow": False,
                           "font": {"size": 14, "color": "#10b981"}}],
            **CHART_LAYOUT,
        )

    # --- Table ---
    tdf = filtered[["note_id", "diagnosis", "medications", "compliance_score",
                     "display_status", "alerts_count"]].copy()
    tdf["compliance_score"] = tdf["compliance_score"].apply(lambda x: f"{x:.0%}")
    tdf = tdf.rename(columns={"display_status": "status"})
    table = dash_table.DataTable(
        id="notes-table",
        columns=[{"name": c.replace("_", " ").title(), "id": c} for c in tdf.columns],
        data=tdf.to_dict("records"),
        sort_action="native", filter_action="native", page_size=15,
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "12px", "backgroundColor": C["card"],
                     "color": C["text"], "border": f"1px solid {C['border']}", "fontSize": "13px"},
        style_header={"backgroundColor": C["bg"], "color": C["text"], "fontWeight": "600",
                       "border": f"1px solid {C['border']}"},
        style_data_conditional=[
            {"if": {"filter_query": "{status} = 'CRITICAL'"}, "backgroundColor": "rgba(239,68,68,0.15)",
             "color": "#fca5a5"},
            {"if": {"filter_query": "{status} = 'Compliant'"}, "backgroundColor": "rgba(16,185,129,0.1)",
             "color": "#6ee7b7"},
            {"if": {"filter_query": "{status} = 'Partially Compliant'"}, "backgroundColor": "rgba(245,158,11,0.1)",
             "color": "#fcd34d"},
        ],
        row_selectable="single",
    )
    return fig1, fig2, fig3, fig4, table


# --- Detail Panel ---
@app.callback(
    Output("detail-panel", "children"),
    Input("notes-table", "selected_rows"),
    [State("results-store", "data"), State("active-filter", "data"), State("status-filter", "value")],
)
def show_detail(selected_rows, data, active_filter, status_filter):
    if not selected_rows or not data:
        return html.Div()

    df = results_to_dataframe(data)
    ok = df[df["status"] != "error"].copy()

    # Apply same filters as table
    if active_filter == "alerts":
        ok = ok[ok["alerts_count"] > 0]
    elif active_filter == "missing_tests":
        ok = ok[ok["test_score"] < 1.0]
    if status_filter and status_filter != "all":
        ok = ok[ok["display_status"] == status_filter]

    idx = selected_rows[0]
    if idx >= len(ok):
        return html.Div()

    note_id = ok.iloc[idx]["note_id"]
    # Find the full result object
    r = None
    for result in data.get("results", []):
        if result.get("note_id") == note_id and result.get("status") not in ("error",):
            r = result
            break
    if not r:
        return html.Div()

    ent = r.get("entities", {})
    gui = r.get("guidelines", {})
    comp = r.get("compliance", {})
    expl = r.get("explanation", {})
    score = comp.get("overall_score", 0)

    return html.Div(
        style={"backgroundColor": C["card"], "padding": "24px", "borderRadius": "12px",
               "border": f"1px solid {C['border']}"},
        children=[
            html.H3(f"Detail: {r.get('note_id', '')}", style={"color": C["accent"], "marginTop": "0"}),
            html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"}, children=[
                html.Div([
                    html.H4("Entities", style={"fontSize": "14px"}),
                    html.Pre(json.dumps(ent, indent=2), style={
                        "backgroundColor": C["bg"], "padding": "12px", "borderRadius": "8px",
                        "fontSize": "12px", "color": C["muted"], "overflow": "auto", "maxHeight": "200px"}),
                ]),
                html.Div([
                    html.H4("Guidelines", style={"fontSize": "14px"}),
                    html.Pre(json.dumps(gui, indent=2), style={
                        "backgroundColor": C["bg"], "padding": "12px", "borderRadius": "8px",
                        "fontSize": "12px", "color": C["muted"], "overflow": "auto", "maxHeight": "200px"}),
                ]),
            ]),
            html.H4(f"Compliance: {score:.0%}",
                     style={"color": C["green"] if score >= 0.7 else C["red"], "fontSize": "16px"}),
            html.Ul([html.Li(b.get("reason", ""), style={"color": C["muted"], "fontSize": "13px"})
                      for b in comp.get("breakdown", [])]),
            html.Div(
                [html.H4("Alerts", style={"color": C["red"], "fontSize": "14px"})] +
                [html.Li(a, style={"color": C["red"], "fontSize": "13px"}) for a in r.get("alerts", [])]
            ) if r.get("alerts") else html.Div(),
            html.Div(
                [html.H4("Recommendations", style={"color": C["green"], "fontSize": "14px"})] +
                [html.Li(rec, style={"color": C["muted"], "fontSize": "13px"}) for rec in r.get("recommendations", [])]
            ) if r.get("recommendations") else html.Div(),
            html.H4("Explanation", style={"fontSize": "14px"}),
            html.Pre(json.dumps(expl.get("rationale", {}), indent=2), style={
                "backgroundColor": C["bg"], "padding": "12px", "borderRadius": "8px",
                "fontSize": "12px", "color": C["muted"], "maxHeight": "300px", "overflow": "auto"}),
        ],
    )


# --- Chat ---
@app.callback(
    [Output("chat-messages", "children"), Output("chat-input", "value"), Output("chat-history-store", "data")],
    [Input("btn-send-chat", "n_clicks"), Input("btn-clear-chat", "n_clicks")],
    [State("chat-input", "value"), State("chat-session-id", "data"), State("chat-history-store", "data")],
    prevent_initial_call=True,
)
def handle_chat(send_clicks, clear_clicks, input_value, session_id, history):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update
    trigger = ctx.triggered[0]["prop_id"]

    if "btn-clear-chat" in trigger:
        return [_chat_bubble("advisor", "Chat cleared. Describe a clinical scenario to get started.")], "", []

    if not input_value or not input_value.strip():
        return dash.no_update, dash.no_update, dash.no_update

    message = input_value.strip()
    messages = [_chat_bubble("advisor", "Welcome! Describe a patient scenario for guideline-based recommendations.")]
    for msg in (history or []):
        messages.append(_chat_bubble(msg["role"], msg["content"], msg.get("guideline_details")))
    messages.append(_chat_bubble("doctor", message))

    # Call API
    bot_response = "Error connecting to API."
    guideline_details = {}
    try:
        resp = requests.post(f"{API_BASE}/chat", json={"message": message, "session_id": session_id}, timeout=120)
        if resp.status_code == 200:
            resp_data = resp.json()
            bot_response = resp_data.get("response", "No response.")
            guideline_details = resp_data.get("guideline_details", {})
        else:
            bot_response = f"Error: API returned {resp.status_code}"
    except Exception as e:
        bot_response = f"Connection error: {e}. Make sure the API is running."

    messages.append(_chat_bubble("advisor", bot_response, guideline_details))
    new_history = list(history or [])
    new_history.append({"role": "doctor", "content": message})
    new_history.append({"role": "advisor", "content": bot_response, "guideline_details": guideline_details})
    return messages, "", new_history


# --- Provider Switching ---
@app.callback(
    Output("provider-status", "children"),
    Input("llm-selector", "value"),
    prevent_initial_call=True,
)
def switch_provider(provider):
    try:
        resp = requests.post(f"{API_BASE}/set_provider", json={"provider": provider}, timeout=30)
        if resp.status_code == 200:
            return f"Active: {provider}"
        return f"Error switching"
    except Exception:
        return "API unavailable"


# --- Main ---

def main():
    print(f"\nStarting Clinical AI Dashboard on port {config.DASHBOARD_PORT}...")
    print(f"Dashboard: http://localhost:{config.DASHBOARD_PORT}")
    print(f"API: {API_BASE}")
    app.run(host=config.DASHBOARD_HOST, port=config.DASHBOARD_PORT, debug=config.DASHBOARD_DEBUG)


if __name__ == "__main__":
    main()
