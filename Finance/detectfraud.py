import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timedelta

import numpy as np
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

TRANSACTION_TYPES = ["Payment", "Refund", "Transfer", "Withdrawal"]
ACCOUNTS = ["A101", "A102", "A103", "A104"]
CHANNELS = ["ACH", "Card", "Wire", "ATM", "Online"]
MERCHANTS = ["Northwind Office", "Contoso Travel", "Fabrikam Supply", "Tailspin Retail", "Adventure Works"]


def generate_transactions(n=100, seed=42, anomaly_count=5):
    rng = np.random.default_rng(seed)
    amounts = rng.normal(loc=200, scale=50, size=n).clip(min=5)

    anomaly_count = min(max(int(anomaly_count), 0), n)
    anomaly_indices = rng.choice(n, size=anomaly_count, replace=False) if anomaly_count else []
    if anomaly_count:
        amounts[anomaly_indices] *= rng.integers(5, 10, size=anomaly_count)

    start_time = datetime(2026, 5, 1, 8, 0)
    timestamps = [
        start_time + timedelta(minutes=int(rng.integers(8, 7200)))
        for _ in range(n)
    ]

    transactions = pd.DataFrame(
        {
            "Transaction ID": [f"TX-{i + 1:04d}" for i in range(n)],
            "Timestamp": timestamps,
            "Amount": amounts.round(2),
            "Type": rng.choice(TRANSACTION_TYPES, size=n, p=[0.45, 0.12, 0.25, 0.18]),
            "Account": rng.choice(ACCOUNTS, size=n),
            "Channel": rng.choice(CHANNELS, size=n, p=[0.28, 0.34, 0.12, 0.12, 0.14]),
            "Merchant": rng.choice(MERCHANTS, size=n),
        }
    )
    transactions = transactions.sort_values("Timestamp").reset_index(drop=True)
    return transactions


def flag_anomalies(df, z_threshold=3.0, high_amount_threshold=1000.0):
    reviewed = df.copy()
    mean = reviewed["Amount"].mean()
    std = reviewed["Amount"].std(ddof=0)
    reviewed["Z-Score"] = 0.0 if std == 0 else ((reviewed["Amount"] - mean) / std).round(2)

    reviewed["Is Z-Score Outlier"] = reviewed["Z-Score"].abs() > z_threshold
    reviewed["Is High Amount"] = reviewed["Amount"] >= high_amount_threshold
    reviewed["Is High Risk Channel"] = reviewed["Channel"].isin(["Wire", "ATM"]) & (
        reviewed["Amount"] >= high_amount_threshold * 0.6
    )
    reviewed["Is High Risk Type"] = reviewed["Type"].isin(["Transfer", "Withdrawal"]) & (
        reviewed["Amount"] >= high_amount_threshold * 0.5
    )

    reason_columns = [
        ("Is Z-Score Outlier", f"z-score above {z_threshold:g}"),
        ("Is High Amount", f"amount at or above ${high_amount_threshold:,.0f}"),
        ("Is High Risk Channel", "large wire or ATM activity"),
        ("Is High Risk Type", "large transfer or withdrawal"),
    ]

    reasons = []
    for _, row in reviewed.iterrows():
        row_reasons = [label for column, label in reason_columns if row[column]]
        reasons.append("; ".join(row_reasons) if row_reasons else "No fraud indicator")

    reviewed["Reason"] = reasons
    reviewed["Flagged"] = reviewed["Reason"] != "No fraud indicator"
    reviewed["Risk Score"] = (
        reviewed[["Is Z-Score Outlier", "Is High Amount", "Is High Risk Channel", "Is High Risk Type"]]
        .astype(int)
        .mul([45, 25, 15, 15])
        .sum(axis=1)
    )
    return reviewed


def build_prompt(row):
    return f"""
You are a forensic accountant AI. A financial transaction has been flagged for possible fraud:

Transaction ID: {row["Transaction ID"]}
Timestamp: {row["Timestamp"]}
Amount: ${row["Amount"]:,.2f}
Type: {row["Type"]}
Account: {row["Account"]}
Channel: {row["Channel"]}
Merchant: {row["Merchant"]}
Z-Score: {row["Z-Score"]}
Risk Score: {row["Risk Score"]}
Detection Reason: {row["Reason"]}

Explain why this transaction might be suspicious and what actions a finance team should take.
Keep the explanation concise and avoid stating that fraud is proven.
""".strip()


def local_explanation(row):
    amount = float(row["Amount"])
    reason = row["Reason"]
    next_actions = (
        "Verify source documentation, confirm the recipient or merchant, compare the activity "
        "with the account's normal pattern, and hold release or reimbursement until the review is complete."
    )
    return f"""
### Why It Was Flagged
Transaction {row["Transaction ID"]} is a {row["Type"].lower()} for ${amount:,.2f}. It was flagged because {reason}. Its z-score is {row["Z-Score"]}, which compares the transaction amount with the simulated population.

### Recommended Actions
{next_actions}
""".strip()


def openai_explanation(row, model):
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("OPENAI_API_KEY is not set. Using local fraud explanation instead.")
        return None

    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=model, temperature=0.2)
        response = llm.invoke(build_prompt(row))
        return response.content
    except Exception as exc:
        st.warning(f"OpenAI response unavailable, using local fraud explanation instead. Details: {exc}")
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


def ollama_explanation(row, base_url, model):
    if not model:
        st.warning("No Ollama model selected. Using local fraud explanation instead.")
        return None

    generate_url = f"{base_url.rstrip('/')}/api/generate"
    payload =     {
        "model": model,
        "prompt": build_prompt(row),
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
        st.warning(f"Ollama response unavailable, using local fraud explanation instead. Details: {exc}")
        return None


def generate_explanation(row, provider, openai_model, ollama_base_url, ollama_model):
    if provider == "OpenAI":
        return openai_explanation(row, openai_model)
    if provider == "Ollama":
        return ollama_explanation(row, ollama_base_url, ollama_model)
    return None


def render_provider_controls():
    st.header("AI Provider")
    provider = st.radio(
        "Provider",
        ["OpenAI", "Ollama", "Local fallback"],
        index=1,
        horizontal=True,
    )

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
            st.info("No Ollama models found. Pull the default with `ollama pull llama3.2:1b`.")
            ollama_model = st.text_input("Ollama Model", value=ollama_model)

    if provider == "Ollama":
        render_ollama_thinking_control(ollama_model)
    else:
        st.session_state["ollama_think"] = None

    return provider, openai_model, ollama_base_url, ollama_model


def render_transaction_table(df):
    display_df = df.copy()
    display_df["Timestamp"] = pd.to_datetime(display_df["Timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
            "Z-Score": st.column_config.NumberColumn("Z-Score", format="%.2f"),
            "Risk Score": st.column_config.ProgressColumn(
                "Risk Score",
                min_value=0,
                max_value=100,
                format="%d",
            ),
        },
    )


def main():
    st.set_page_config(
        page_title="Fraud Detection Agent",
        page_icon="$",
        layout="wide",
    )
    st.title("Fraud Detection Agent")

    with st.sidebar:
        render_provider = render_provider_controls()
        provider, openai_model, ollama_base_url, ollama_model = render_provider

        st.header("Simulation")
        transaction_count = st.slider("Transactions", min_value=25, max_value=500, value=100, step=25)
        anomaly_count = st.slider("Injected Anomalies", min_value=0, max_value=25, value=5)
        seed = st.number_input("Random Seed", min_value=1, max_value=9999, value=42, step=1)

        st.header("Detection Rules")
        z_threshold = st.slider("Z-Score Threshold", min_value=1.0, max_value=5.0, value=3.0, step=0.25)
        high_amount_threshold = st.number_input(
            "High Amount Threshold",
            min_value=100.0,
            value=1000.0,
            step=50.0,
            format="%.2f",
        )

    transactions = generate_transactions(
        n=transaction_count,
        seed=int(seed),
        anomaly_count=anomaly_count,
    )
    reviewed = flag_anomalies(
        transactions,
        z_threshold=z_threshold,
        high_amount_threshold=high_amount_threshold,
    )
    flagged = reviewed[reviewed["Flagged"]].sort_values(
        ["Risk Score", "Amount"],
        ascending=[False, False],
    )

    total_amount = float(reviewed["Amount"].sum())
    flagged_amount = float(flagged["Amount"].sum()) if not flagged.empty else 0.0
    flagged_rate = len(flagged) / len(reviewed) if len(reviewed) else 0

    metrics = st.columns(4)
    metrics[0].metric("Transactions", f"{len(reviewed):,}")
    metrics[1].metric("Flagged", f"{len(flagged):,}", f"{flagged_rate:.1%}")
    metrics[2].metric("Total Amount", f"${total_amount:,.2f}")
    metrics[3].metric("Flagged Amount", f"${flagged_amount:,.2f}")

    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("All Transactions")
        render_transaction_table(reviewed)

    with right:
        st.subheader("Flagged Transactions")
        if flagged.empty:
            st.info("No transactions are currently flagged by the configured rules.")
        else:
            render_transaction_table(
                flagged[
                    [
                        "Transaction ID",
                        "Timestamp",
                        "Amount",
                        "Type",
                        "Account",
                        "Channel",
                        "Z-Score",
                        "Risk Score",
                        "Reason",
                    ]
                ]
            )

    st.subheader("AI Explanation")
    if flagged.empty:
        st.caption("Adjust the simulation or detection rules to generate flagged transactions.")
    else:
        selected_id = st.selectbox(
            "Choose a flagged Transaction ID",
            flagged["Transaction ID"].tolist(),
        )
        selected_row = flagged[flagged["Transaction ID"] == selected_id].iloc[0]

        if st.button("Explain Selected Transaction", type="primary"):
            with st.spinner("Generating fraud review explanation..."):
                explanation = generate_explanation(
                    selected_row,
                    provider,
                    openai_model,
                    ollama_base_url,
                    ollama_model,
                )
                if explanation is None:
                    explanation = local_explanation(selected_row)
                st.session_state.fraud_explanation = explanation
                st.session_state.fraud_explanation_id = selected_id
                st.session_state.fraud_explanation_time = datetime.now().strftime("%I:%M %p")

        if st.session_state.get("fraud_explanation"):
            explanation_id = st.session_state.get("fraud_explanation_id")
            timestamp = st.session_state.get("fraud_explanation_time")
            if timestamp and explanation_id:
                st.caption(f"Last explanation: {explanation_id} at {timestamp}")
            st.markdown(st.session_state.fraud_explanation)
        else:
            st.info("Select a flagged transaction and generate an explanation.")

    st.caption("This lab app uses simulated data and rule-based flags for educational fraud review workflows.")


if __name__ == "__main__":
    main()
