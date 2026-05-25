import json
import os
import urllib.error
import urllib.request
from datetime import datetime

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


def get_revenue_data(start_month, months, starting_revenue, monthly_growth, volatility):
    dates = pd.date_range(start=start_month, periods=months, freq="ME")
    rows = []

    for index, month in enumerate(dates):
        seasonal_adjustment = 0
        if month.month in (11, 12):
            seasonal_adjustment = starting_revenue * 0.08
        elif month.month in (1, 2, 7):
            seasonal_adjustment = -starting_revenue * 0.04

        deterministic_noise = ((index * 37) % 11 - 5) * volatility
        revenue = starting_revenue + (index * monthly_growth) + seasonal_adjustment + deterministic_noise
        rows.append({"Month": month, "Revenue": max(round(revenue, 2), 0)})

    return pd.DataFrame(rows)


def calculate_local_forecast(df, forecast_months):
    if df.empty:
        return pd.DataFrame(columns=["Month", "Forecast Revenue"])

    monthly_deltas = df["Revenue"].diff().dropna()
    average_growth = monthly_deltas.mean() if not monthly_deltas.empty else 0
    recent_growth = monthly_deltas.tail(3).mean() if len(monthly_deltas) >= 3 else average_growth
    blended_growth = (average_growth * 0.55) + (recent_growth * 0.45)

    last_month = df["Month"].max()
    last_revenue = df["Revenue"].iloc[-1]
    future_months = pd.date_range(
        start=last_month + pd.offsets.MonthEnd(1),
        periods=forecast_months,
        freq="ME",
    )

    forecast_rows = []
    for step, month in enumerate(future_months, start=1):
        seasonal_adjustment = 0
        if month.month in (11, 12):
            seasonal_adjustment = last_revenue * 0.04
        elif month.month in (1, 2, 7):
            seasonal_adjustment = -last_revenue * 0.025

        forecast_rows.append(
            {
                "Month": month,
                "Forecast Revenue": max(round(last_revenue + (blended_growth * step) + seasonal_adjustment, 2), 0),
            }
        )

    return pd.DataFrame(forecast_rows)


def build_prompt(history_df, forecast_df, business_context):
    revenue_table = history_df.copy()
    revenue_table["Month"] = revenue_table["Month"].dt.strftime("%Y-%m")
    revenue_table["Revenue"] = revenue_table["Revenue"].map(lambda value: f"${value:,.0f}")

    forecast_table = forecast_df.copy()
    forecast_table["Month"] = forecast_table["Month"].dt.strftime("%Y-%m")
    forecast_table["Forecast Revenue"] = forecast_table["Forecast Revenue"].map(lambda value: f"${value:,.0f}")

    return f"""
You are a financial forecasting analyst. Review the monthly revenue data and draft a business-friendly forecast.

Historical revenue:
{revenue_table.to_string(index=False)}

Baseline statistical forecast:
{forecast_table.to_string(index=False)}

Business context:
{business_context or "No additional context provided."}

Tasks:
1. Identify growth patterns, seasonality, or anomalies.
2. Forecast the next {len(forecast_df)} month(s), using the baseline forecast as a guide.
3. Provide practical recommendations for leadership.

Keep the response concise, use clear headings, and avoid guarantees.
""".strip()


def local_forecast_summary(history_df, forecast_df, business_context):
    monthly_deltas = history_df["Revenue"].diff().dropna()
    average_growth = monthly_deltas.mean() if not monthly_deltas.empty else 0
    growth_rate = (
        (history_df["Revenue"].iloc[-1] - history_df["Revenue"].iloc[0])
        / history_df["Revenue"].iloc[0]
        if history_df["Revenue"].iloc[0]
        else 0
    )
    forecast_values = ", ".join(
        f"{row['Month'].strftime('%b %Y')}: ${row['Forecast Revenue']:,.0f}"
        for _, row in forecast_df.iterrows()
    )

    context_line = ""
    if business_context:
        context_line = f"\n\n### Context Note\nThe forecast should be read alongside this business context: {business_context}"

    return f"""
### Trend Read
Revenue moved from ${history_df['Revenue'].iloc[0]:,.0f} to ${history_df['Revenue'].iloc[-1]:,.0f}, a total change of {growth_rate:.1%}. Average month-over-month movement is about ${average_growth:,.0f}.

### Forecast
Projected revenue for the next {len(forecast_df)} month(s): {forecast_values}.

### Recommendations
Use the forecast to plan staffing, cash reserves, and campaign timing. Watch months with softer seasonal behavior before adding fixed costs, and compare actuals against this baseline every month so the forecast can be refreshed quickly.{context_line}
""".strip()


def build_combined_chart(history_df, forecast_df):
    actual = history_df.rename(columns={"Revenue": "Actual Revenue"})[["Month", "Actual Revenue"]]
    forecast = forecast_df.rename(columns={"Forecast Revenue": "Forecast Revenue"})[
        ["Month", "Forecast Revenue"]
    ]
    return pd.merge(actual, forecast, how="outer", on="Month").sort_values("Month")


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


def openai_forecast(prompt, model):
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("OPENAI_API_KEY is not set. Using local forecast summary instead.")
        return None

    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=model, temperature=0.3)
        response = llm.invoke(prompt)
        return response.content
    except Exception as exc:
        st.warning(f"OpenAI response unavailable, using local forecast summary instead. Details: {exc}")
        return None


def ollama_forecast(prompt, base_url, model):
    if not model:
        st.warning("No Ollama model selected. Using local forecast summary instead.")
        return None

    generate_url = f"{base_url.rstrip('/')}/api/generate"
    payload =     {
        "model": model,
        "prompt": prompt,
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
        st.warning(f"Ollama response unavailable, using local forecast summary instead. Details: {exc}")
        return None


def generate_forecast_summary(
    history_df,
    forecast_df,
    business_context,
    provider,
    openai_model,
    ollama_base_url,
    ollama_model,
):
    prompt = build_prompt(history_df, forecast_df, business_context)
    if provider == "OpenAI":
        return openai_forecast(prompt, openai_model)
    if provider == "Ollama":
        return ollama_forecast(prompt, ollama_base_url, ollama_model)
    return None


def render_provider_controls():
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
            ollama_model = st.text_input("Ollama Model", value=DEFAULT_OLLAMA_MODEL)
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
            ollama_model = st.text_input("Ollama Model", value=DEFAULT_OLLAMA_MODEL)

    if provider == "Ollama":
        render_ollama_thinking_control(ollama_model)
    else:
        st.session_state["ollama_think"] = None

    return provider, openai_model, ollama_base_url, ollama_model


def main():
    st.set_page_config(
        page_title="Advanced Prediction Agent",
        page_icon="$",
        layout="wide",
    )
    st.title("Advanced Prediction Agent")

    with st.sidebar:
        st.header("Revenue Inputs")
        start_month = st.date_input("Start Month", value=datetime(2024, 1, 1))
        historical_months = st.slider("Historical Months", min_value=6, max_value=36, value=18)
        forecast_months = st.slider("Forecast Months", min_value=1, max_value=12, value=3)
        starting_revenue = st.number_input(
            "Starting Monthly Revenue",
            min_value=0,
            value=10000,
            step=500,
            format="%d",
        )
        monthly_growth = st.number_input(
            "Expected Monthly Growth",
            min_value=-10000,
            value=300,
            step=50,
            format="%d",
        )
        volatility = st.number_input(
            "Mock Volatility",
            min_value=0,
            value=175,
            step=25,
            format="%d",
        )
        business_context = st.text_area(
            "Business Context",
            value="Revenue is from a subscription business with occasional year-end demand lift.",
            height=120,
        )
        provider, openai_model, ollama_base_url, ollama_model = render_provider_controls()

    history_df = get_revenue_data(
        start_month=start_month,
        months=historical_months,
        starting_revenue=starting_revenue,
        monthly_growth=monthly_growth,
        volatility=volatility,
    )
    forecast_df = calculate_local_forecast(history_df, forecast_months)
    chart_df = build_combined_chart(history_df, forecast_df).set_index("Month")

    latest_revenue = history_df["Revenue"].iloc[-1]
    projected_revenue = forecast_df["Forecast Revenue"].iloc[-1]
    projected_change = (
        (projected_revenue - latest_revenue) / latest_revenue
        if latest_revenue
        else 0
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Latest Revenue", f"${latest_revenue:,.0f}")
    metric_cols[1].metric("Final Forecast", f"${projected_revenue:,.0f}", f"{projected_change:.1%}")
    metric_cols[2].metric("Historical Months", f"{historical_months}")
    metric_cols[3].metric("Forecast Months", f"{forecast_months}")

    chart_col, table_col = st.columns([1.35, 1])
    with chart_col:
        st.subheader("Revenue Trend")
        st.line_chart(chart_df, use_container_width=True)

    with table_col:
        st.subheader("Forecast Values")
        display_forecast = forecast_df.copy()
        display_forecast["Month"] = display_forecast["Month"].dt.strftime("%b %Y")
        st.dataframe(
            display_forecast,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Forecast Revenue": st.column_config.NumberColumn(
                    "Forecast Revenue",
                    format="$%.0f",
                )
            },
        )

    st.subheader("AI Forecast Summary")
    if st.button("Run Forecast Analysis", type="primary"):
        with st.spinner("Generating forecast analysis..."):
            summary = generate_forecast_summary(
                history_df,
                forecast_df,
                business_context,
                provider,
                openai_model,
                ollama_base_url,
                ollama_model,
            )
            if summary is None:
                summary = local_forecast_summary(history_df, forecast_df, business_context)
            st.session_state.forecast_summary = summary

    if "forecast_summary" not in st.session_state:
        st.session_state.forecast_summary = local_forecast_summary(
            history_df,
            forecast_df,
            business_context,
        )
    st.markdown(st.session_state.forecast_summary)
    st.caption("Forecasts are planning estimates for educational use, not financial guarantees.")


if __name__ == "__main__":
    main()
