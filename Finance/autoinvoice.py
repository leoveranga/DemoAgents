import json
import os
import urllib.error
import urllib.request
from datetime import date, timedelta
from html import escape
from io import BytesIO

import pandas as pd
import streamlit as st


DEFAULT_PROVIDER = "Ollama"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OLLAMA_MODEL = "llama3.2:1b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
REASONING_MODEL_MARKERS = (
    "deepseek-r1",
    "gemma4",
    "gpt-oss",
    "qwen3",
    "qwen3.5",
    "reason",
    "thinking",
    "think",
)

DEFAULT_CLIENT = {
    "Client Name": "Acme Operations LLC",
    "Contact Email": "ap@acme-ops.example",
    "Billing Address": "415 Market Street, Suite 900, Philadelphia, PA 19106",
    "Invoice Number": "INV-2026-0005",
    "Issue Date": date.today(),
    "Due Days": 14,
    "Tax Rate": 6.0,
    "Discount": 0.0,
}

DEFAULT_ITEMS = [
    {"Description": "Consulting - Project A", "Quantity": 10.0, "Unit": "hours", "Rate": 150.0},
    {"Description": "Design Work", "Quantity": 5.0, "Unit": "hours", "Rate": 120.0},
    {"Description": "Training Session", "Quantity": 3.0, "Unit": "hours", "Rate": 200.0},
]


def default_invoice_items():
    return pd.DataFrame(DEFAULT_ITEMS)


def clean_invoice_items(items):
    df = items.copy()
    expected_columns = ["Description", "Quantity", "Unit", "Rate"]
    for column in expected_columns:
        if column not in df.columns:
            df[column] = "" if column in ["Description", "Unit"] else 0.0

    df = df[expected_columns]
    df["Description"] = df["Description"].fillna("").astype(str).str.strip()
    df["Unit"] = df["Unit"].fillna("units").astype(str).str.strip()
    df["Unit"] = df["Unit"].where(df["Unit"] != "", "units")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0.0)
    df["Rate"] = pd.to_numeric(df["Rate"], errors="coerce").fillna(0.0)
    df = df[df["Description"] != ""].reset_index(drop=True)
    df["Line Total"] = (df["Quantity"] * df["Rate"]).round(2)
    return df


def calculate_totals(items, tax_rate, discount):
    subtotal = float(items["Line Total"].sum()) if not items.empty else 0.0
    discount = max(float(discount), 0.0)
    taxable_amount = max(subtotal - discount, 0.0)
    tax = taxable_amount * (float(tax_rate) / 100)
    total = taxable_amount + tax
    return {
        "subtotal": subtotal,
        "discount": discount,
        "tax": tax,
        "total": total,
    }


def build_prompt(client, items, totals):
    due_date = client["Issue Date"] + timedelta(days=int(client["Due Days"]))
    invoice_table = items.to_string(index=False)
    return f"""
You are an invoicing assistant AI. Generate a professional invoice narrative from the billing data below.

Client:
- Name: {client["Client Name"]}
- Contact Email: {client["Contact Email"]}
- Billing Address: {client["Billing Address"]}
- Invoice Number: {client["Invoice Number"]}
- Issue Date: {client["Issue Date"]}
- Due Date: {due_date}

Invoice Items:
{invoice_table}

Totals:
- Subtotal: ${totals["subtotal"]:,.2f}
- Discount: ${totals["discount"]:,.2f}
- Tax: ${totals["tax"]:,.2f}
- Total Due: ${totals["total"]:,.2f}

Return:
1. A short summary of services rendered.
2. A clean line-item explanation.
3. Payment terms and a polite closing message.
Keep the response concise and ready to paste into an invoice.
""".strip()


def local_invoice_summary(client, items, totals):
    due_date = client["Issue Date"] + timedelta(days=int(client["Due Days"]))
    if items.empty:
        line_summary = "No billable line items were provided."
    else:
        lines = [
            (
                f"- {row['Description']}: {row['Quantity']:g} {row['Unit']} "
                f"at ${row['Rate']:,.2f}, totaling ${row['Line Total']:,.2f}."
            )
            for _, row in items.iterrows()
        ]
        line_summary = "\n".join(lines)

    return f"""
### Summary
Invoice {client["Invoice Number"]} bills {client["Client Name"]} for the services listed below. The invoice subtotal is ${totals["subtotal"]:,.2f}, with a ${totals["discount"]:,.2f} discount and ${totals["tax"]:,.2f} in tax.

### Line Items
{line_summary}

### Payment Terms
Total amount due is ${totals["total"]:,.2f}. Please remit payment by {due_date.isoformat()}. Thank you for your business.
""".strip()


def openai_invoice_summary(client, items, totals, model):
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("OPENAI_API_KEY is not set. Using local invoice text instead.")
        return None

    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=model, temperature=0.3)
        response = llm.invoke(build_prompt(client, items, totals))
        return response.content
    except Exception as exc:
        st.warning(f"OpenAI response unavailable, using local invoice text instead. Details: {exc}")
        return None


def is_reasoning_model(model):
    normalized = (model or "").lower()
    return any(marker in normalized for marker in REASONING_MODEL_MARKERS)


def render_ollama_thinking_control(model):
    if is_reasoning_model(model):
        disable_thinking = st.checkbox(
            "Turn off thinking mode",
            value=True,
            help=(
                "For supported Ollama reasoning models, sends `think: false` "
                "to reduce latency and token usage."
            ),
        )
        st.session_state["ollama_think"] = False if disable_thinking else None
        if disable_thinking:
            st.caption("Thinking mode will be disabled for this Ollama request.")
    else:
        st.session_state["ollama_think"] = None


@st.cache_data(ttl=15)
def get_ollama_models(base_url):
    tags_url = f"{base_url.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return [], str(exc)

    models = sorted(
        model["name"]
        for model in payload.get("models", [])
        if model.get("name")
    )
    return models, None


def ollama_invoice_summary(client, items, totals, base_url, model):
    if not model:
        st.warning("No Ollama model selected. Using local invoice text instead.")
        return None

    generate_url = f"{base_url.rstrip('/')}/api/generate"
    payload =     {
        "model": model,
        "prompt": build_prompt(client, items, totals),
        "stream": False,
        "options": {"temperature": 0.3},
    }
    think = st.session_state.get("ollama_think")
    if think is not None:
        payload["think"] = think

    request_body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        generate_url,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload.get("response", "").strip() or None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        st.warning(f"Ollama response unavailable, using local invoice text instead. Details: {exc}")
        return None


def generate_invoice_summary(client, items, totals, provider, openai_model, ollama_base_url, ollama_model):
    if provider == "OpenAI":
        return openai_invoice_summary(client, items, totals, openai_model)
    if provider == "Ollama":
        return ollama_invoice_summary(client, items, totals, ollama_base_url, ollama_model)
    return None


def build_invoice_text(client, items, totals, summary):
    due_date = client["Issue Date"] + timedelta(days=int(client["Due Days"]))
    lines = [
        "INVOICE",
        f"Invoice Number: {client['Invoice Number']}",
        f"Issue Date: {client['Issue Date']}",
        f"Due Date: {due_date}",
        "",
        "Bill To:",
        client["Client Name"],
        client["Billing Address"],
        client["Contact Email"],
        "",
        "Line Items:",
    ]

    for _, row in items.iterrows():
        lines.append(
            (
                f"- {row['Description']}: {row['Quantity']:g} {row['Unit']} x "
                f"${row['Rate']:,.2f} = ${row['Line Total']:,.2f}"
            )
        )

    lines.extend(
        [
            "",
            f"Subtotal: ${totals['subtotal']:,.2f}",
            f"Discount: ${totals['discount']:,.2f}",
            f"Tax: ${totals['tax']:,.2f}",
            f"Total Due: ${totals['total']:,.2f}",
            "",
            "Invoice Summary:",
            summary,
        ]
    )
    return "\n".join(lines)


def build_invoice_html(client, items, totals, summary):
    due_date = client["Issue Date"] + timedelta(days=int(client["Due Days"]))
    rows = "\n".join(
        (
            "<tr>"
            f"<td>{escape(str(row['Description']))}</td>"
            f"<td>{row['Quantity']:g}</td>"
            f"<td>{escape(str(row['Unit']))}</td>"
            f"<td>${row['Rate']:,.2f}</td>"
            f"<td>${row['Line Total']:,.2f}</td>"
            "</tr>"
        )
        for _, row in items.iterrows()
    )
    safe_summary = escape(summary).replace("\n", "<br>")
    client_name = escape(client["Client Name"])
    billing_address = escape(client["Billing Address"])
    contact_email = escape(client["Contact Email"])
    invoice_number = escape(client["Invoice Number"])
    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{invoice_number}</title>
  <style>
    body {{ font-family: Arial, sans-serif; color: #1f2933; margin: 40px; }}
    h1 {{ margin-bottom: 0; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 24px; }}
    th, td {{ border-bottom: 1px solid #d9e2ec; padding: 10px; text-align: left; }}
    th {{ background: #f0f4f8; }}
    .total {{ font-size: 20px; font-weight: 700; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-top: 24px; }}
  </style>
</head>
<body>
  <h1>INVOICE</h1>
  <p>{invoice_number}</p>
  <div class="grid">
    <div>
      <strong>Bill To</strong><br>
      {client_name}<br>
      {billing_address}<br>
      {contact_email}
    </div>
    <div>
      <strong>Issue Date:</strong> {client["Issue Date"]}<br>
      <strong>Due Date:</strong> {due_date}<br>
      <strong>Tax Rate:</strong> {float(client["Tax Rate"]):g}%
    </div>
  </div>
  <table>
    <thead>
      <tr><th>Description</th><th>Qty</th><th>Unit</th><th>Rate</th><th>Total</th></tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
  <p><strong>Subtotal:</strong> ${totals["subtotal"]:,.2f}</p>
  <p><strong>Discount:</strong> ${totals["discount"]:,.2f}</p>
  <p><strong>Tax:</strong> ${totals["tax"]:,.2f}</p>
  <p class="total">Total Due: ${totals["total"]:,.2f}</p>
  <h2>Summary</h2>
  <p>{safe_summary}</p>
</body>
</html>
""".strip()


def try_build_pdf(invoice_text):
    try:
        from fpdf import FPDF
    except ImportError:
        return None

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=11)
    for line in invoice_text.splitlines():
        safe_line = line.encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 8, safe_line)

    output = pdf.output(dest="S")
    if isinstance(output, str):
        return output.encode("latin-1")
    return bytes(output)


def render_provider_controls():
    providers = ["OpenAI", "Ollama", "Local fallback"]
    default_index = providers.index(DEFAULT_PROVIDER)
    provider = st.selectbox("AI Provider", providers, index=default_index)

    openai_model = DEFAULT_OPENAI_MODEL
    ollama_base_url = DEFAULT_OLLAMA_BASE_URL
    ollama_model = DEFAULT_OLLAMA_MODEL

    if provider == "OpenAI":
        openai_model = st.text_input("OpenAI Model", value=openai_model)
        if not os.getenv("OPENAI_API_KEY"):
            st.info("Set OPENAI_API_KEY to use OpenAI. The app will fall back locally until then.")
    elif provider == "Ollama":
        ollama_base_url = st.text_input("Ollama URL", value=ollama_base_url)
        if st.button("Refresh Ollama Models"):
            get_ollama_models.clear()

        ollama_models, ollama_error = get_ollama_models(ollama_base_url)
        if ollama_error:
            st.warning(f"Could not list Ollama models from {ollama_base_url}: {ollama_error}")
            ollama_model = st.text_input("Ollama Model", value=ollama_model)
        elif ollama_models:
            selectable_models = [DEFAULT_OLLAMA_MODEL] + [
                model for model in ollama_models if model != DEFAULT_OLLAMA_MODEL
            ]
            ollama_model = st.selectbox(
                "Available Ollama Models",
                selectable_models,
                index=0,
            )
            st.caption(f"{len(ollama_models)} model(s) available.")
        else:
            st.info(f"No Ollama models found. Pull the default with `ollama pull {DEFAULT_OLLAMA_MODEL}`.")
            ollama_model = st.text_input("Ollama Model", value=ollama_model)

    if provider == "Ollama":
        render_ollama_thinking_control(ollama_model)
    else:
        st.session_state["ollama_think"] = None

    return provider, openai_model, ollama_base_url, ollama_model


def render_sidebar():
    with st.sidebar:
        st.header("Invoice Settings")
        provider, openai_model, ollama_base_url, ollama_model = render_provider_controls()

        st.divider()
        client = {
            "Client Name": st.text_input("Client Name", value=DEFAULT_CLIENT["Client Name"]),
            "Contact Email": st.text_input("Contact Email", value=DEFAULT_CLIENT["Contact Email"]),
            "Billing Address": st.text_area("Billing Address", value=DEFAULT_CLIENT["Billing Address"], height=80),
            "Invoice Number": st.text_input("Invoice Number", value=DEFAULT_CLIENT["Invoice Number"]),
            "Issue Date": st.date_input("Issue Date", value=DEFAULT_CLIENT["Issue Date"]),
            "Due Days": st.number_input("Payment Due In Days", min_value=0, max_value=120, value=DEFAULT_CLIENT["Due Days"]),
            "Tax Rate": st.number_input(
                "Tax Rate (%)",
                min_value=0.0,
                max_value=30.0,
                value=DEFAULT_CLIENT["Tax Rate"],
                step=0.25,
            ),
            "Discount": st.number_input(
                "Discount ($)",
                min_value=0.0,
                max_value=100000.0,
                value=DEFAULT_CLIENT["Discount"],
                step=25.0,
            ),
        }

    return client, provider, openai_model, ollama_base_url, ollama_model


def render_metrics(totals):
    metric_columns = st.columns(4)
    metric_columns[0].metric("Subtotal", f"${totals['subtotal']:,.2f}")
    metric_columns[1].metric("Discount", f"${totals['discount']:,.2f}")
    metric_columns[2].metric("Tax", f"${totals['tax']:,.2f}")
    metric_columns[3].metric("Total Due", f"${totals['total']:,.2f}")


def main():
    st.set_page_config(page_title="Automated Invoicing Agent", page_icon=":receipt:", layout="wide")
    st.title("Automated Invoicing Agent")

    client, provider, openai_model, ollama_base_url, ollama_model = render_sidebar()

    st.subheader("Invoice Items")
    edited_items = st.data_editor(
        default_invoice_items(),
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Description": st.column_config.TextColumn("Description", required=True),
            "Quantity": st.column_config.NumberColumn("Quantity", min_value=0.0, step=0.5, format="%.2f"),
            "Unit": st.column_config.TextColumn("Unit"),
            "Rate": st.column_config.NumberColumn("Rate", min_value=0.0, step=10.0, format="$%.2f"),
        },
    )

    invoice_items = clean_invoice_items(edited_items)
    totals = calculate_totals(invoice_items, client["Tax Rate"], client["Discount"])
    render_metrics(totals)

    if st.button("Generate Invoice", type="primary"):
        if invoice_items.empty:
            st.error("Add at least one invoice line item before generating the invoice.")
            return

        summary = generate_invoice_summary(
            client,
            invoice_items,
            totals,
            provider,
            openai_model,
            ollama_base_url,
            ollama_model,
        )
        if not summary:
            summary = local_invoice_summary(client, invoice_items, totals)

        st.session_state["invoice_items"] = invoice_items
        st.session_state["invoice_totals"] = totals
        st.session_state["invoice_summary"] = summary
        st.session_state["invoice_client"] = client

    if "invoice_summary" not in st.session_state:
        st.info("Review the line items and generate an invoice when ready.")
        return

    invoice_items = st.session_state["invoice_items"]
    totals = st.session_state["invoice_totals"]
    summary = st.session_state["invoice_summary"]
    client = st.session_state["invoice_client"]

    st.subheader("Final Invoice")
    st.dataframe(
        invoice_items,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rate": st.column_config.NumberColumn("Rate", format="$%.2f"),
            "Line Total": st.column_config.NumberColumn("Line Total", format="$%.2f"),
        },
    )

    st.markdown(summary)

    invoice_text = build_invoice_text(client, invoice_items, totals, summary)
    invoice_html = build_invoice_html(client, invoice_items, totals, summary)
    pdf_bytes = try_build_pdf(invoice_text)

    download_columns = st.columns(3)
    download_columns[0].download_button(
        "Download TXT",
        data=invoice_text,
        file_name=f"{client['Invoice Number']}.txt",
        mime="text/plain",
    )
    download_columns[1].download_button(
        "Download HTML",
        data=invoice_html,
        file_name=f"{client['Invoice Number']}.html",
        mime="text/html",
    )
    if pdf_bytes:
        download_columns[2].download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name=f"{client['Invoice Number']}.pdf",
            mime="application/pdf",
        )
    else:
        download_columns[2].caption("Install `fpdf` to enable PDF export.")


if __name__ == "__main__":
    main()
