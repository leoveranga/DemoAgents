import json
import os
import urllib.error
import urllib.request
from datetime import datetime

import pandas as pd
import streamlit as st


DEFAULT_PROFILE = {
    "income": 70000,
    "monthly_expenses": {
        "housing": 1500,
        "food": 600,
        "transport": 300,
        "entertainment": 200,
        "others": 150,
    },
    "financial_goals": ["save for a house", "retire at 60"],
    "risk_tolerance": "moderate",
}


RISK_ALLOCATIONS = {
    "conservative": "30% stock index funds, 60% high-quality bonds, and 10% cash reserves",
    "moderate": "60% stock index funds, 35% bonds, and 5% cash reserves",
    "aggressive": "80% diversified stock index funds, 15% bonds, and 5% cash reserves",
}

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


def build_profile(income, expenses, goals, risk_tolerance):
    clean_goals = [goal.strip() for goal in goals.split(",") if goal.strip()]
    return {
        "income": int(income),
        "monthly_expenses": {
            key: float(value) for key, value in expenses.items()
        },
        "financial_goals": clean_goals or DEFAULT_PROFILE["financial_goals"],
        "risk_tolerance": risk_tolerance,
    }


def monthly_summary(profile):
    monthly_income = profile["income"] / 12
    expenses = sum(profile["monthly_expenses"].values())
    surplus = monthly_income - expenses
    savings_rate = surplus / monthly_income if monthly_income else 0
    emergency_fund = expenses * 6
    return {
        "monthly_income": monthly_income,
        "monthly_expenses": expenses,
        "monthly_surplus": surplus,
        "savings_rate": savings_rate,
        "emergency_fund": emergency_fund,
    }


def fallback_advice(profile, user_question):
    summary = monthly_summary(profile)
    risk = profile["risk_tolerance"].lower()
    allocation = RISK_ALLOCATIONS.get(risk, RISK_ALLOCATIONS["moderate"])
    suggested_savings = max(summary["monthly_surplus"] * 0.65, 0)
    goals = ", ".join(profile["financial_goals"])

    if summary["monthly_surplus"] <= 0:
        savings_line = (
            "Your expenses meet or exceed estimated take-home capacity before taxes, "
            "so the first priority is reducing recurring spending or increasing income."
        )
    else:
        savings_line = (
            f"Target about ${suggested_savings:,.0f} per month toward goals, while "
            f"keeping roughly ${summary['monthly_surplus'] - suggested_savings:,.0f} "
            "available for taxes, irregular bills, and flexibility."
        )

    question_context = ""
    if user_question:
        question_context = f"\n\nRegarding your question, '{user_question}', prioritize actions that support {goals} without weakening emergency savings."

    return f"""
### Financial Status
Annual income is ${profile['income']:,.0f}, with estimated monthly income of ${summary['monthly_income']:,.0f} and tracked monthly expenses of ${summary['monthly_expenses']:,.0f}. Estimated monthly surplus is ${summary['monthly_surplus']:,.0f}, which implies a savings capacity near {summary['savings_rate']:.0%} before tax adjustments.

### Monthly Savings Goal
{savings_line} Build or maintain an emergency fund of about ${summary['emergency_fund']:,.0f}, equal to six months of current expenses.

### Investment Strategy
For a {risk} risk profile, consider a diversified allocation such as {allocation}. Use automated monthly contributions, keep short-term house savings in cash or high-yield savings, and reserve investment risk for longer-term goals like retirement.

### Next Steps
Review housing, food, and discretionary spending monthly. Tie each dollar of surplus to one of these goals: {goals}.{question_context}
""".strip()


def build_prompt(profile, user_question):
    return f"""
You are a financial advisor AI. Given the user's financial profile below, provide personalized suggestions:
- Summarize their financial status
- Suggest a monthly savings goal
- Recommend an investment strategy based on their risk profile
- Answer the user's question if one is provided

User Profile:
Income: ${profile['income']:,.0f}
Expenses: {profile['monthly_expenses']}
Goals: {', '.join(profile['financial_goals'])}
Risk: {profile['risk_tolerance']}
User Question: {user_question or 'No specific question'}

Respond clearly and concisely. Avoid guarantees and remind the user this is educational guidance.
""".strip()


def openai_advice(profile, user_question, model):
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("OPENAI_API_KEY is not set. Using local advisory logic instead.")
        return None

    prompt = build_prompt(profile, user_question)

    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=model, temperature=0.4)
        response = llm.invoke(prompt)
        return response.content
    except Exception as exc:
        st.warning(f"OpenAI response unavailable, using local advisory logic instead. Details: {exc}")
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


def ollama_advice(profile, user_question, base_url, model):
    if not model:
        st.warning("No Ollama model selected. Using local advisory logic instead.")
        return None

    prompt = build_prompt(profile, user_question)
    generate_url = f"{base_url.rstrip('/')}/api/generate"
    payload =     {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.4},
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
        st.warning(f"Ollama response unavailable, using local advisory logic instead. Details: {exc}")
        return None


def generate_advice(profile, user_question, provider, openai_model, ollama_base_url, ollama_model):
    if provider == "OpenAI":
        return openai_advice(profile, user_question, openai_model)
    if provider == "Ollama":
        return ollama_advice(profile, user_question, ollama_base_url, ollama_model)
    return None


def render_profile_table(profile):
    rows = [
        {"Category": category.title(), "Monthly Amount": amount}
        for category, amount in profile["monthly_expenses"].items()
    ]
    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Monthly Amount": st.column_config.NumberColumn(
                "Monthly Amount",
                format="$%.0f",
            )
        },
    )


def initialize_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Enter or adjust the financial profile, then ask for an advisory summary.",
            }
        ]


def main():
    st.set_page_config(
        page_title="Individualized Financial Advisory Agent",
        page_icon="$",
        layout="wide",
    )
    initialize_state()

    st.title("Individualized Financial Advisory Agent")

    with st.sidebar:
        st.header("Financial Profile")
        income = st.number_input(
            "Annual Income",
            min_value=0,
            value=DEFAULT_PROFILE["income"],
            step=1000,
            format="%d",
        )

        st.subheader("Monthly Expenses")
        expenses = {}
        for category, default in DEFAULT_PROFILE["monthly_expenses"].items():
            expenses[category] = st.number_input(
                category.title(),
                min_value=0,
                value=int(default),
                step=50,
                format="%d",
            )

        goals = st.text_input(
            "Financial Goals",
            value=", ".join(DEFAULT_PROFILE["financial_goals"]),
        )
        risk_tolerance = st.selectbox(
            "Risk Tolerance",
            ["conservative", "moderate", "aggressive"],
            index=1,
        )
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
                st.info("No Ollama models found. Pull one with `ollama pull <model>`.")

        if provider == "Ollama":
            render_ollama_thinking_control(ollama_model)
        else:
            st.session_state["ollama_think"] = None

    profile = build_profile(income, expenses, goals, risk_tolerance)
    summary = monthly_summary(profile)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Monthly Income", f"${summary['monthly_income']:,.0f}")
    metric_cols[1].metric("Monthly Expenses", f"${summary['monthly_expenses']:,.0f}")
    metric_cols[2].metric("Monthly Surplus", f"${summary['monthly_surplus']:,.0f}")
    metric_cols[3].metric("Savings Capacity", f"{summary['savings_rate']:.0%}")

    left, right = st.columns([1, 1.35])

    with left:
        st.subheader("Expense Snapshot")
        render_profile_table(profile)
        st.caption("This app provides educational guidance, not professional financial advice.")

    with right:
        st.subheader("Advisor Chat")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask for a savings, budget, or investment recommendation"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Preparing financial recommendation..."):
                    response = generate_advice(
                        profile,
                        prompt,
                        provider,
                        openai_model,
                        ollama_base_url,
                        ollama_model,
                    )
                    if response is None:
                        response = fallback_advice(profile, prompt)
                    st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

        if st.button("Generate Advisory Summary", type="primary"):
            response = generate_advice(
                profile,
                "",
                provider,
                openai_model,
                ollama_base_url,
                ollama_model,
            )
            if response is None:
                response = fallback_advice(profile, "")
            timestamp = datetime.now().strftime("%I:%M %p")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"_{timestamp}_\n\n{response}"}
            )
            st.rerun()


if __name__ == "__main__":
    main()
