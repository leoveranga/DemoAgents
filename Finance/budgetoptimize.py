import json
import os
import urllib.error
import urllib.request
from datetime import datetime

import pandas as pd
import streamlit as st


DEFAULT_BUDGET_ROWS = [
    {"Department": "Marketing", "Allocated Budget": 80000.0, "Actual Spending": 95000.0},
    {"Department": "Sales", "Allocated Budget": 60000.0, "Actual Spending": 58000.0},
    {"Department": "Engineering", "Allocated Budget": 150000.0, "Actual Spending": 140000.0},
    {"Department": "HR", "Allocated Budget": 40000.0, "Actual Spending": 30000.0},
    {"Department": "IT", "Allocated Budget": 50000.0, "Actual Spending": 60000.0},
    {"Department": "Operations", "Allocated Budget": 70000.0, "Actual Spending": 85000.0},
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


def apply_dark_theme():
    st.markdown(
        """
        <style>
            :root {
                color-scheme: dark;
            }

            .stApp {
                background: #0f1117;
                color: #f4f6fb;
            }

            [data-testid="stSidebar"],
            [data-testid="stHeader"],
            [data-testid="stToolbar"] {
                background: #111827;
            }

            [data-testid="stMetric"],
            [data-testid="stDataFrame"],
            [data-testid="stExpander"] {
                background-color: #171c26;
                border: 1px solid #263041;
                border-radius: 8px;
                padding: 0.75rem;
            }

            div[data-testid="stMarkdownContainer"] p,
            div[data-testid="stMarkdownContainer"] li {
                color: #e5e7eb;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def default_budget_frame():
    return pd.DataFrame(DEFAULT_BUDGET_ROWS)


def normalize_budget_data(budget_data):
    df = budget_data.copy()
    expected_columns = ["Department", "Allocated Budget", "Actual Spending"]
    for column in expected_columns:
        if column not in df.columns:
            df[column] = "" if column == "Department" else 0.0

    df = df[expected_columns]
    df["Department"] = df["Department"].fillna("").astype(str).str.strip()
    df["Department"] = df["Department"].where(df["Department"] != "", "Unassigned")
    df["Allocated Budget"] = pd.to_numeric(df["Allocated Budget"], errors="coerce").fillna(0.0)
    df["Actual Spending"] = pd.to_numeric(df["Actual Spending"], errors="coerce").fillna(0.0)
    df = df[df["Department"].str.len() > 0].reset_index(drop=True)

    df["Variance"] = df["Actual Spending"] - df["Allocated Budget"]
    df["Variance %"] = df.apply(
        lambda row: row["Variance"] / row["Allocated Budget"]
        if row["Allocated Budget"]
        else 0.0,
        axis=1,
    )
    df["Variance Percent"] = df["Variance %"] * 100
    df["Status"] = "On Budget"
    df.loc[df["Variance"] > 0, "Status"] = "Overspent"
    df.loc[df["Variance"] < 0, "Status"] = "Underutilized"
    return df


def budget_summary(analysis_df):
    total_allocated = float(analysis_df["Allocated Budget"].sum())
    total_spent = float(analysis_df["Actual Spending"].sum())
    variance = total_spent - total_allocated
    overspent = analysis_df[analysis_df["Variance"] > 0]
    underutilized = analysis_df[analysis_df["Variance"] < 0]
    return {
        "total_allocated": total_allocated,
        "total_spent": total_spent,
        "variance": variance,
        "overspent_count": len(overspent),
        "underutilized_count": len(underutilized),
        "overspent_total": float(overspent["Variance"].sum()),
        "underutilized_total": abs(float(underutilized["Variance"].sum())),
    }


def build_prompt(analysis_df, priorities):
    budget_table = analysis_df.to_string(index=False)
    priority_text = priorities.strip() or "Maintain operational continuity while improving cost efficiency."
    return f"""
You are a financial planning assistant.

Given the following department-level budget data:

{budget_table}

Strategic priorities:
{priority_text}

Tasks:
1. Identify departments that overspent or underspent.
2. Suggest budget reallocations for next quarter.
3. Recommend cost-saving strategies.
4. Call out any spending pattern that needs management review.

Present the answer in concise bullet points with dollar amounts where useful.
""".strip()


def fallback_recommendations(analysis_df, priorities):
    summary = budget_summary(analysis_df)
    overspent = analysis_df[analysis_df["Variance"] > 0].sort_values("Variance", ascending=False)
    underutilized = analysis_df[analysis_df["Variance"] < 0].sort_values("Variance")

    if overspent.empty:
        overspent_lines = "- No departments are currently above allocated budget."
    else:
        overspent_lines = "\n".join(
            f"- {row['Department']} overspent by ${row['Variance']:,.0f} "
            f"({row['Variance %']:.1%} above budget)."
            for _, row in overspent.iterrows()
        )

    if underutilized.empty:
        underutilized_lines = "- No departments are currently underutilizing budget."
    else:
        underutilized_lines = "\n".join(
            f"- {row['Department']} is under budget by ${abs(row['Variance']):,.0f} "
            f"({abs(row['Variance %']):.1%} below budget)."
            for _, row in underutilized.iterrows()
        )

    available_reallocation = min(summary["overspent_total"], summary["underutilized_total"])
    if available_reallocation:
        reallocation_line = (
            f"Consider reallocating up to ${available_reallocation:,.0f} from departments with "
            "confirmed underutilization toward verified overspend drivers, while preserving funds "
            "for strategic priorities."
        )
    elif summary["variance"] > 0:
        reallocation_line = (
            f"The organization is ${summary['variance']:,.0f} over budget overall. Treat this as "
            "a cost-containment issue before increasing future allocations."
        )
    else:
        reallocation_line = (
            f"The organization is ${abs(summary['variance']):,.0f} under budget overall. Reserve "
            "the surplus for priority initiatives or risk buffers."
        )

    priority_text = priorities.strip() or "current strategic priorities"
    return f"""
### Overspending
{overspent_lines}

### Underutilization
{underutilized_lines}

### Reallocation Recommendation
- {reallocation_line}
- Review the largest positive variances first and require owner-level explanations before permanently raising next-quarter budgets.
- Align any surplus with {priority_text}.

### Cost-Saving Measures
- Rebid vendor contracts, subscriptions, and project services in departments exceeding budget.
- Move recurring discretionary spend into monthly approval thresholds.
- Use zero-based budgeting for departments with repeated variance above 10%.
""".strip()


def openai_recommendations(analysis_df, priorities, model):
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("OPENAI_API_KEY is not set. Using local budget logic instead.")
        return None

    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=model, temperature=0.3)
        response = llm.invoke(build_prompt(analysis_df, priorities))
        return response.content
    except Exception as exc:
        st.warning(f"OpenAI response unavailable, using local budget logic instead. Details: {exc}")
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


def ollama_recommendations(analysis_df, priorities, base_url, model):
    if not model:
        st.warning("No Ollama model selected. Using local budget logic instead.")
        return None

    generate_url = f"{base_url.rstrip('/')}/api/generate"
    payload =     {
        "model": model,
        "prompt": build_prompt(analysis_df, priorities),
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
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload.get("response")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        st.warning(f"Ollama response unavailable, using local budget logic instead. Details: {exc}")
        return None


def generate_recommendations(analysis_df, priorities, provider, openai_model, ollama_base_url, ollama_model):
    if provider == "OpenAI":
        return openai_recommendations(analysis_df, priorities, openai_model)
    if provider == "Ollama":
        return ollama_recommendations(analysis_df, priorities, ollama_base_url, ollama_model)
    return None


def render_budget_table(analysis_df):
    st.dataframe(
        analysis_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Allocated Budget": st.column_config.NumberColumn(
                "Allocated Budget",
                format="$%.0f",
            ),
            "Actual Spending": st.column_config.NumberColumn(
                "Actual Spending",
                format="$%.0f",
            ),
            "Variance": st.column_config.NumberColumn(
                "Variance",
                format="$%.0f",
            ),
            "Variance %": None,
            "Variance Percent": st.column_config.NumberColumn(
                "Variance %",
                format="%.1f%%",
            ),
        },
    )


def render_budget_chart(analysis_df):
    chart_df = analysis_df.set_index("Department")[["Allocated Budget", "Actual Spending"]]
    st.bar_chart(chart_df, use_container_width=True)


def main():
    st.set_page_config(
        page_title="Budget Optimization Agent",
        page_icon="$",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_dark_theme()

    st.title("Budget Optimization Agent")

    with st.sidebar:
        st.header("AI Provider")
        provider = st.radio(
            "Provider",
            ["OpenAI", "Ollama", "Local fallback"],
            index=["OpenAI", "Ollama", "Local fallback"].index(DEFAULT_PROVIDER),
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
                st.info("No Ollama models found. Pull one with `ollama pull llama3.2:1b`.")
                ollama_model = st.text_input("Ollama Model", value=ollama_model)

        if provider == "Ollama":
            render_ollama_thinking_control(ollama_model)
        else:
            st.session_state["ollama_think"] = None

        st.header("Strategic Priorities")
        priorities = st.text_area(
            "Priorities",
            value="Protect customer-facing operations, fund growth projects, and reduce recurring vendor spend.",
            height=120,
        )

    st.subheader("Department Budget Data")
    edited_budget = st.data_editor(
        default_budget_frame(),
        hide_index=True,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Department": st.column_config.TextColumn("Department", required=True),
            "Allocated Budget": st.column_config.NumberColumn(
                "Allocated Budget",
                min_value=0.0,
                step=1000.0,
                format="$%.0f",
            ),
            "Actual Spending": st.column_config.NumberColumn(
                "Actual Spending",
                min_value=0.0,
                step=1000.0,
                format="$%.0f",
            ),
        },
    )

    analysis_df = normalize_budget_data(edited_budget)
    summary = budget_summary(analysis_df)

    metrics = st.columns(4)
    metrics[0].metric("Allocated Budget", f"${summary['total_allocated']:,.0f}")
    metrics[1].metric("Actual Spending", f"${summary['total_spent']:,.0f}")
    metrics[2].metric("Net Variance", f"${summary['variance']:,.0f}")
    metrics[3].metric("Departments Over Budget", summary["overspent_count"])

    left, right = st.columns([1.15, 1])

    with left:
        st.subheader("Budget Performance")
        render_budget_chart(analysis_df)
        st.subheader("Variance Review")
        render_budget_table(analysis_df)

    with right:
        st.subheader("AI Recommendations")
        if st.button("Analyze Budget", type="primary"):
            with st.spinner("Reviewing spending patterns..."):
                recommendations = generate_recommendations(
                    analysis_df,
                    priorities,
                    provider,
                    openai_model,
                    ollama_base_url,
                    ollama_model,
                )
                if recommendations is None:
                    recommendations = fallback_recommendations(analysis_df, priorities)
                st.session_state.budget_recommendations = recommendations
                st.session_state.budget_recommendations_time = datetime.now().strftime("%I:%M %p")

        recommendations = st.session_state.get("budget_recommendations")
        if recommendations:
            timestamp = st.session_state.get("budget_recommendations_time")
            if timestamp:
                st.caption(f"Last analysis: {timestamp}")
            st.markdown(recommendations)
        else:
            st.info("Run the analysis to generate budget reallocation and cost-saving guidance.")


if __name__ == "__main__":
    main()
