import json
import os
import urllib.error
import urllib.request
from datetime import datetime

import pandas as pd
import streamlit as st


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


DEFAULT_EXPENSES = [
    {
        "Date": "2025-07-01",
        "Employee": "Jordan Lee",
        "Description": "Flight to NYC",
        "Category": "Travel",
        "Amount": 450.00,
        "Receipt": "Yes",
    },
    {
        "Date": "2025-07-02",
        "Employee": "Jordan Lee",
        "Description": "Hotel Stay",
        "Category": "Lodging",
        "Amount": 600.00,
        "Receipt": "Yes",
    },
    {
        "Date": "2025-07-03",
        "Employee": "Jordan Lee",
        "Description": "Team Dinner",
        "Category": "Meals",
        "Amount": 180.00,
        "Receipt": "No",
    },
    {
        "Date": "2025-07-03",
        "Employee": "Jordan Lee",
        "Description": "Taxi",
        "Category": "Transport",
        "Amount": 45.00,
        "Receipt": "Yes",
    },
    {
        "Date": "2025-07-04",
        "Employee": "Jordan Lee",
        "Description": "Conference Fee",
        "Category": "Training",
        "Amount": 1200.00,
        "Receipt": "Yes",
    },
]


POLICY_LIMITS = {
    "Travel": 500.00,
    "Meals": 100.00,
    "Training": 1000.00,
}


CATEGORY_KEYWORDS = {
    "Travel": ["flight", "airfare", "plane", "train"],
    "Lodging": ["hotel", "lodging", "inn"],
    "Meals": ["dinner", "lunch", "breakfast", "meal", "restaurant"],
    "Transport": ["taxi", "uber", "lyft", "cab", "parking"],
    "Training": ["conference", "course", "training", "certification"],
}


def categorize_description(description):
    text = str(description).lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return category
    return "Other"


def normalize_expenses(expenses):
    df = expenses.copy()
    expected_columns = ["Date", "Employee", "Description", "Category", "Amount", "Receipt"]
    for column in expected_columns:
        if column not in df.columns:
            df[column] = ""

    df = df[expected_columns]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    df["Employee"] = df["Employee"].fillna("").astype(str).str.strip()
    df["Description"] = df["Description"].fillna("").astype(str).str.strip()
    df["Category"] = df["Category"].fillna("").astype(str).str.strip()
    df["Category"] = df.apply(
        lambda row: row["Category"] or categorize_description(row["Description"]),
        axis=1,
    )
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
    df["Receipt"] = df["Receipt"].fillna("No").astype(str).str.strip()
    df["Receipt"] = df["Receipt"].where(df["Receipt"].isin(["Yes", "No"]), "No")
    return df


def review_expenses(expenses):
    df = normalize_expenses(expenses)
    review_rows = []

    for _, row in df.iterrows():
        category = row["Category"]
        amount = float(row["Amount"])
        limit = POLICY_LIMITS.get(category)
        issues = []

        if limit is not None and amount > limit:
            issues.append(f"exceeds ${limit:,.0f} {category.lower()} limit")
        if row["Receipt"] != "Yes":
            issues.append("missing receipt")
        if amount <= 0:
            issues.append("amount must be greater than zero")
        if not row["Description"]:
            issues.append("missing description")

        status = "Flagged" if issues else "Approved"
        review_rows.append(
            {
                "Date": row["Date"],
                "Employee": row["Employee"] or "Unassigned",
                "Description": row["Description"],
                "Category": category,
                "Amount": amount,
                "Policy Limit": limit,
                "Receipt": row["Receipt"],
                "Status": status,
                "Reason": "; ".join(issues) if issues else "Within policy",
            }
        )

    return pd.DataFrame(review_rows)


def build_prompt(review_df):
    expense_table = review_df.to_string(index=False)
    return f"""
You are an AI expense compliance auditor.

Below is a list of expenses submitted by an employee:

{expense_table}

Policy limits:
- Travel > $500 should be flagged.
- Meals > $100 should be flagged.
- Training > $1000 should be flagged.
- Missing receipts should be reviewed.

Tasks:
1. Identify expenses that should be reviewed or flagged.
2. Explain the policy reason for each flagged item.
3. Recommend next actions for the finance team.
4. Provide a concise reimbursement summary.
""".strip()


def fallback_summary(review_df):
    flagged = review_df[review_df["Status"] == "Flagged"]
    approved = review_df[review_df["Status"] == "Approved"]
    total = review_df["Amount"].sum()
    flagged_total = flagged["Amount"].sum()

    if flagged.empty:
        flagged_section = "No submitted expenses require review based on the configured policy limits and receipt checks."
        recommendations = "Approve the report after normal finance validation."
    else:
        lines = [
            f"- {row['Description'] or 'Unnamed expense'} (${row['Amount']:,.2f}): {row['Reason']}."
            for _, row in flagged.iterrows()
        ]
        flagged_section = "\n".join(lines)
        recommendations = (
            "Request missing receipts, validate business purpose for flagged items, "
            "and approve only the portion that meets policy or has a documented exception."
        )

    return f"""
### Compliance Summary
Reviewed {len(review_df)} expenses totaling ${total:,.2f}. {len(approved)} item(s) are within policy and {len(flagged)} item(s) require finance review. Flagged spend totals ${flagged_total:,.2f}.

### Flagged Items
{flagged_section}

### Recommended Actions
{recommendations}
""".strip()


def openai_summary(review_df, model):
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("OPENAI_API_KEY is not set. Using local compliance logic instead.")
        return None

    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=model, temperature=0.2)
        response = llm.invoke(build_prompt(review_df))
        return response.content
    except Exception as exc:
        st.warning(f"OpenAI response unavailable, using local compliance logic instead. Details: {exc}")
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


def ollama_summary(review_df, base_url, model):
    if not model:
        st.warning("No Ollama model selected. Using local compliance logic instead.")
        return None

    generate_url = f"{base_url.rstrip('/')}/api/generate"
    payload =     {
        "model": model,
        "prompt": build_prompt(review_df),
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
        st.warning(f"Ollama response unavailable, using local compliance logic instead. Details: {exc}")
        return None


def generate_summary(review_df, provider, openai_model, ollama_base_url, ollama_model):
    if provider == "OpenAI":
        return openai_summary(review_df, openai_model)
    if provider == "Ollama":
        return ollama_summary(review_df, ollama_base_url, ollama_model)
    return None


def default_expense_frame():
    df = pd.DataFrame(DEFAULT_EXPENSES)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    return df


def render_review_table(review_df):
    st.dataframe(
        review_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
            "Policy Limit": st.column_config.NumberColumn("Policy Limit", format="$%.2f"),
        },
    )


def main():
    st.set_page_config(
        page_title="Expense Management Agent",
        page_icon="$",
        layout="wide",
    )

    st.title("Expense Management Agent")

    with st.sidebar:
        st.header("AI Provider")
        provider = st.radio(
            "Provider",
            ["OpenAI", "Ollama", "Local fallback"],
            index=1,
            horizontal=True,
        )

        openai_model = "gpt-4o-mini"
        ollama_base_url = "http://localhost:11434"
        ollama_model = "llama3.2:1b"

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
                default_index = (
                    ollama_models.index(ollama_model)
                    if ollama_model in ollama_models
                    else 0
                )
                ollama_model = st.selectbox(
                    "Available Ollama Models",
                    ollama_models,
                    index=default_index,
                )
                st.caption(f"{len(ollama_models)} model(s) available.")
            else:
                st.info("No Ollama models found. Pull one with `ollama pull <model>`.")
                ollama_model = st.text_input("Ollama Model", value=ollama_model)

        if provider == "Ollama":
            render_ollama_thinking_control(ollama_model)
        else:
            st.session_state["ollama_think"] = None

        st.header("Policy Limits")
        for category, limit in POLICY_LIMITS.items():
            st.metric(category, f"${limit:,.0f}")

    st.subheader("Expense Report")
    edited_expenses = st.data_editor(
        default_expense_frame(),
        hide_index=True,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Date": st.column_config.DateColumn("Date"),
            "Category": st.column_config.SelectboxColumn(
                "Category",
                options=["Travel", "Lodging", "Meals", "Transport", "Training", "Other"],
            ),
            "Amount": st.column_config.NumberColumn("Amount", min_value=0.0, format="$%.2f"),
            "Receipt": st.column_config.SelectboxColumn("Receipt", options=["Yes", "No"]),
        },
    )

    review_df = review_expenses(edited_expenses)
    flagged_count = int((review_df["Status"] == "Flagged").sum())
    total_amount = float(review_df["Amount"].sum())
    flagged_amount = float(review_df.loc[review_df["Status"] == "Flagged", "Amount"].sum())
    reimbursable_amount = total_amount - flagged_amount

    metrics = st.columns(4)
    metrics[0].metric("Submitted Spend", f"${total_amount:,.2f}")
    metrics[1].metric("Flagged Items", flagged_count)
    metrics[2].metric("Flagged Spend", f"${flagged_amount:,.2f}")
    metrics[3].metric("Ready To Reimburse", f"${reimbursable_amount:,.2f}")

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Policy Review")
        render_review_table(review_df)

    with right:
        st.subheader("Compliance Summary")
        if st.button("Audit Expenses", type="primary"):
            with st.spinner("Reviewing expense policy compliance..."):
                summary = generate_summary(
                    review_df,
                    provider,
                    openai_model,
                    ollama_base_url,
                    ollama_model,
                )
                if summary is None:
                    summary = fallback_summary(review_df)
                st.session_state.expense_summary = summary
                st.session_state.expense_summary_time = datetime.now().strftime("%I:%M %p")

        summary = st.session_state.get("expense_summary")
        if summary:
            timestamp = st.session_state.get("expense_summary_time")
            if timestamp:
                st.caption(f"Last audit: {timestamp}")
            st.markdown(summary)
        else:
            st.info("Run an audit to generate the finance compliance summary.")


if __name__ == "__main__":
    main()
