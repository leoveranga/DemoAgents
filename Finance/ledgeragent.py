import json
import os
import urllib.error
import urllib.request
from datetime import datetime
from io import StringIO

import pandas as pd
import streamlit as st


DEFAULT_LEDGER = [
    {
        "Date": "2025-07-01",
        "Description": "Customer Payment",
        "Account": "Accounts Receivable",
        "Debit": 0.00,
        "Credit": 500.00,
    },
    {
        "Date": "2025-07-01",
        "Description": "Revenue Recorded",
        "Account": "Revenue",
        "Debit": 500.00,
        "Credit": 0.00,
    },
    {
        "Date": "2025-07-02",
        "Description": "Office Supplies",
        "Account": "Office Supplies",
        "Debit": 100.00,
        "Credit": 0.00,
    },
    {
        "Date": "2025-07-02",
        "Description": "Cash Paid",
        "Account": "Cash",
        "Debit": 100.00,
        "Credit": 0.00,
    },
    {
        "Date": "2025-07-03",
        "Description": "Consulting Income",
        "Account": "Revenue",
        "Debit": 0.00,
        "Credit": 1500.00,
    },
]


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


def get_sample_ledger():
    return pd.DataFrame(DEFAULT_LEDGER)


def normalize_ledger(ledger):
    df = ledger.copy()
    expected_columns = ["Date", "Description", "Account", "Debit", "Credit"]
    for column in expected_columns:
        if column not in df.columns:
            df[column] = "" if column not in ["Debit", "Credit"] else 0.0

    df = df[expected_columns]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    df["Description"] = df["Description"].fillna("").astype(str).str.strip()
    df["Account"] = df["Account"].fillna("").astype(str).str.strip()
    df["Debit"] = pd.to_numeric(df["Debit"], errors="coerce").fillna(0.0)
    df["Credit"] = pd.to_numeric(df["Credit"], errors="coerce").fillna(0.0)
    return df


def ledger_metrics(ledger):
    total_debits = float(ledger["Debit"].sum())
    total_credits = float(ledger["Credit"].sum())
    difference = total_debits - total_credits
    return {
        "total_debits": total_debits,
        "total_credits": total_credits,
        "difference": difference,
        "is_balanced": abs(difference) < 0.005,
        "entry_count": len(ledger),
    }


def find_potential_issues(ledger):
    issues = []

    for index, row in ledger.iterrows():
        row_number = index + 1
        debit = float(row["Debit"])
        credit = float(row["Credit"])
        description = row["Description"] or "No description"

        if debit < 0 or credit < 0:
            issues.append(f"Row {row_number} ({description}) has a negative debit or credit value.")
        if debit > 0 and credit > 0:
            issues.append(f"Row {row_number} ({description}) has both debit and credit populated.")
        if debit == 0 and credit == 0:
            issues.append(f"Row {row_number} ({description}) has no debit or credit amount.")
        if not row["Account"]:
            issues.append(f"Row {row_number} ({description}) is missing an account.")
        if pd.isna(row["Date"]):
            issues.append(f"Row {row_number} ({description}) has an invalid or missing date.")

    return issues


def build_prompt(ledger, user_question):
    metrics = ledger_metrics(ledger)
    issues = find_potential_issues(ledger)
    issue_text = "\n".join(f"- {issue}" for issue in issues) or "- No row-level validation issues found."
    ledger_table = ledger.to_string(index=False)

    return f"""
You are a financial ledger analyst AI. Analyze the transactions below.

Ledger:
{ledger_table}

Computed totals:
- Total debits: ${metrics['total_debits']:,.2f}
- Total credits: ${metrics['total_credits']:,.2f}
- Difference, debits minus credits: ${metrics['difference']:,.2f}
- Balanced: {metrics['is_balanced']}

Initial validation findings:
{issue_text}

Tasks:
1. State whether debits equal credits.
2. Highlight imbalances, missing fields, unusual rows, or possible accounting errors.
3. Provide reconciliation notes for the finance team.
4. Answer the user's question if one is provided.

User question: {user_question or 'No specific question'}

Respond clearly and concisely. Do not invent transactions that are not present in the ledger.
""".strip()


def fallback_analysis(ledger, user_question):
    metrics = ledger_metrics(ledger)
    issues = find_potential_issues(ledger)
    by_account = (
        ledger.groupby("Account", dropna=False)[["Debit", "Credit"]]
        .sum()
        .reset_index()
        .sort_values("Account")
    )

    if metrics["is_balanced"]:
        status = (
            f"Debits and credits are balanced at ${metrics['total_debits']:,.2f}. "
            "The ledger can move to normal review, assuming source documents match the entries."
        )
    else:
        direction = "debits exceed credits" if metrics["difference"] > 0 else "credits exceed debits"
        status = (
            f"Debits and credits are not balanced. Total debits are ${metrics['total_debits']:,.2f}, "
            f"total credits are ${metrics['total_credits']:,.2f}, and {direction} by "
            f"${abs(metrics['difference']):,.2f}."
        )

    if issues:
        issue_lines = "\n".join(f"- {issue}" for issue in issues)
    else:
        issue_lines = "- No row-level validation issues were found."

    account_lines = "\n".join(
        f"- {row['Account'] or 'Unassigned'}: debits ${row['Debit']:,.2f}, credits ${row['Credit']:,.2f}"
        for _, row in by_account.iterrows()
    )

    recommendation = (
        "Review the source documents for entries driving the imbalance, confirm whether any offsetting "
        "entries are missing, and post an adjusting entry only after the business event is verified."
    )
    if metrics["is_balanced"] and not issues:
        recommendation = "Archive the reconciliation support and continue with normal close procedures."

    question_context = ""
    if user_question:
        question_context = f"\n\n### Response to Question\nFor '{user_question}', use the totals and issue list above as the starting point for reconciliation."

    return f"""
### Ledger Status
{status}

### Potential Issues
{issue_lines}

### Account Summary
{account_lines}

### Reconciliation Notes
{recommendation}{question_context}
""".strip()


def openai_analysis(ledger, user_question, model):
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("OPENAI_API_KEY is not set. Using local ledger logic instead.")
        return None

    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=model, temperature=0.2)
        response = llm.invoke(build_prompt(ledger, user_question))
        return response.content
    except Exception as exc:
        st.warning(f"OpenAI response unavailable, using local ledger logic instead. Details: {exc}")
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


def ollama_analysis(ledger, user_question, base_url, model):
    if not model:
        st.warning("No Ollama model selected. Using local ledger logic instead.")
        return None

    generate_url = f"{base_url.rstrip('/')}/api/generate"
    payload =     {
        "model": model,
        "prompt": build_prompt(ledger, user_question),
        "stream": False,
        "options": {"temperature": 0.2},
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
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload.get("response")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        st.warning(f"Ollama response unavailable, using local ledger logic instead. Details: {exc}")
        return None


def generate_analysis(ledger, user_question, provider, openai_model, ollama_base_url, ollama_model):
    if provider == "OpenAI":
        return openai_analysis(ledger, user_question, openai_model)
    if provider == "Ollama":
        return ollama_analysis(ledger, user_question, ollama_base_url, ollama_model)
    return None


def parse_uploaded_ledger(uploaded_file):
    if uploaded_file is None:
        return None, None

    try:
        if uploaded_file.name.lower().endswith(".csv"):
            return pd.read_csv(uploaded_file), None
        if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file), None
        return None, "Upload a CSV or Excel file."
    except Exception as exc:
        return None, str(exc)


def render_ledger_editor(initial_ledger):
    edited = st.data_editor(
        initial_ledger,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Date": st.column_config.DateColumn("Date"),
            "Debit": st.column_config.NumberColumn("Debit", format="$%.2f", min_value=0.0),
            "Credit": st.column_config.NumberColumn("Credit", format="$%.2f", min_value=0.0),
        },
        key="ledger_editor",
    )
    return normalize_ledger(edited)


def render_metrics(metrics):
    cols = st.columns(4)
    cols[0].metric("Debits", f"${metrics['total_debits']:,.2f}")
    cols[1].metric("Credits", f"${metrics['total_credits']:,.2f}")
    cols[2].metric("Difference", f"${metrics['difference']:,.2f}")
    cols[3].metric("Balanced", "Yes" if metrics["is_balanced"] else "No")


def initialize_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Review or edit the ledger, then ask for a reconciliation summary.",
            }
        ]
    if "ledger_source" not in st.session_state:
        st.session_state.ledger_source = get_sample_ledger()


def main():
    st.set_page_config(
        page_title="Ledger Analysis Agent",
        page_icon="$",
        layout="wide",
    )
    initialize_state()

    st.title("Ledger Analysis Agent")

    with st.sidebar:
        st.header("Ledger Source")
        uploaded_file = st.file_uploader("Upload Ledger", type=["csv", "xlsx", "xls"])
        uploaded_ledger, upload_error = parse_uploaded_ledger(uploaded_file)
        if upload_error:
            st.warning(upload_error)
        if uploaded_ledger is not None and st.button("Use Uploaded Ledger"):
            st.session_state.ledger_source = uploaded_ledger
            st.rerun()
        if st.button("Reset Sample Ledger"):
            st.session_state.ledger_source = get_sample_ledger()
            st.rerun()

        st.subheader("AI Provider")
        providers = ["OpenAI", "Ollama", "Local fallback"]
        provider = st.radio(
            "Provider",
            providers,
            index=providers.index(DEFAULT_PROVIDER),
            horizontal=True,
        )
        openai_model = DEFAULT_OPENAI_MODEL
        ollama_base_url = DEFAULT_OLLAMA_BASE_URL
        ollama_model = None

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
                ollama_model = st.text_input("Ollama Model", value=DEFAULT_OLLAMA_MODEL)
            elif ollama_models:
                default_index = (
                    ollama_models.index(DEFAULT_OLLAMA_MODEL)
                    if DEFAULT_OLLAMA_MODEL in ollama_models
                    else 0
                )
                ollama_model = st.selectbox(
                    "Available Ollama Models",
                    ollama_models,
                    index=default_index,
                )
                st.caption(f"{len(ollama_models)} model(s) available.")
            else:
                st.info(f"No Ollama models found. Pull the default with `ollama pull {DEFAULT_OLLAMA_MODEL}`.")
                ollama_model = st.text_input("Ollama Model", value=DEFAULT_OLLAMA_MODEL)

        if provider == "Ollama":
            render_ollama_thinking_control(ollama_model)
        else:
            st.session_state["ollama_think"] = None

    ledger = render_ledger_editor(normalize_ledger(st.session_state.ledger_source))
    metrics = ledger_metrics(ledger)
    render_metrics(metrics)

    left, right = st.columns([1, 1.35])

    with left:
        st.subheader("Validation")
        issues = find_potential_issues(ledger)
        if issues:
            for issue in issues:
                st.warning(issue)
        else:
            st.success("No row-level validation issues found.")

        st.subheader("Account Totals")
        account_totals = (
            ledger.groupby("Account", dropna=False)[["Debit", "Credit"]]
            .sum()
            .reset_index()
            .sort_values("Account")
        )
        st.dataframe(
            account_totals,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Debit": st.column_config.NumberColumn("Debit", format="$%.2f"),
                "Credit": st.column_config.NumberColumn("Credit", format="$%.2f"),
            },
        )

    with right:
        st.subheader("Ledger Chat")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask about imbalances, entries, or reconciliation steps"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing ledger..."):
                    response = generate_analysis(
                        ledger,
                        prompt,
                        provider,
                        openai_model,
                        ollama_base_url,
                        ollama_model,
                    )
                    if response is None:
                        response = fallback_analysis(ledger, prompt)
                    st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

        if st.button("Generate Reconciliation Summary", type="primary"):
            response = generate_analysis(
                ledger,
                "",
                provider,
                openai_model,
                ollama_base_url,
                ollama_model,
            )
            if response is None:
                response = fallback_analysis(ledger, "")
            timestamp = datetime.now().strftime("%I:%M %p")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"_{timestamp}_\n\n{response}"}
            )
            st.rerun()

    csv_buffer = StringIO()
    ledger.to_csv(csv_buffer, index=False)
    st.download_button(
        "Download Current Ledger CSV",
        data=csv_buffer.getvalue(),
        file_name="ledger_review.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
