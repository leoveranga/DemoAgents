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

DEFAULT_PROFILE = {
    "entity_name": "Contoso Manufacturing",
    "sector": "Industrial Manufacturing",
    "decision": "Extend a $2.5M revolving credit facility",
    "debt_to_equity": 2.8,
    "liquidity_ratio": 0.95,
    "revenue_growth": -5.0,
    "net_margin": 3.2,
    "credit_score": 620,
    "market_volatility": 8,
    "credit_exposure": 2_500_000,
    "external_rating": "BB",
    "market_trend": "Weakening demand and elevated input costs",
}

RATING_RISK = {
    "AAA": 0,
    "AA": 5,
    "A": 10,
    "BBB": 20,
    "BB": 35,
    "B": 50,
    "CCC": 70,
    "Unrated": 55,
}


def risk_band(score):
    if score >= 65:
        return "High"
    if score >= 35:
        return "Medium"
    return "Low"


def score_financial_profile(profile):
    debt_score = min(max((profile["debt_to_equity"] / 3.5) * 100, 0), 100)
    liquidity_score = min(max(((1.5 - profile["liquidity_ratio"]) / 1.5) * 100, 0), 100)
    growth_score = min(max(((10 - profile["revenue_growth"]) / 25) * 100, 0), 100)
    margin_score = min(max(((15 - profile["net_margin"]) / 20) * 100, 0), 100)
    credit_score = min(max(((760 - profile["credit_score"]) / 360) * 100, 0), 100)
    volatility_score = min(max((profile["market_volatility"] / 10) * 100, 0), 100)
    rating_score = RATING_RISK.get(profile["external_rating"], RATING_RISK["Unrated"])

    components = {
        "Debt leverage": (debt_score, 0.20),
        "Liquidity pressure": (liquidity_score, 0.20),
        "Revenue momentum": (growth_score, 0.15),
        "Profitability": (margin_score, 0.15),
        "Credit quality": (credit_score, 0.15),
        "Market volatility": (volatility_score, 0.10),
        "External rating": (rating_score, 0.05),
    }
    total_score = sum(score * weight for score, weight in components.values())
    return round(total_score), components


def build_financial_profile(
    entity_name,
    sector,
    decision,
    debt_to_equity,
    liquidity_ratio,
    revenue_growth,
    net_margin,
    credit_score,
    market_volatility,
    credit_exposure,
    external_rating,
    market_trend,
):
    return {
        "entity_name": entity_name.strip() or DEFAULT_PROFILE["entity_name"],
        "sector": sector.strip() or DEFAULT_PROFILE["sector"],
        "decision": decision.strip() or DEFAULT_PROFILE["decision"],
        "debt_to_equity": float(debt_to_equity),
        "liquidity_ratio": float(liquidity_ratio),
        "revenue_growth": float(revenue_growth),
        "net_margin": float(net_margin),
        "credit_score": int(credit_score),
        "market_volatility": int(market_volatility),
        "credit_exposure": float(credit_exposure),
        "external_rating": external_rating,
        "market_trend": market_trend.strip() or DEFAULT_PROFILE["market_trend"],
    }


def profile_to_dataframe(profile):
    return pd.DataFrame(
        [
            {"Metric": "Debt-to-Equity Ratio", "Value": profile["debt_to_equity"]},
            {"Metric": "Liquidity Ratio", "Value": profile["liquidity_ratio"]},
            {"Metric": "Revenue Growth (YoY %)", "Value": profile["revenue_growth"]},
            {"Metric": "Net Margin (%)", "Value": profile["net_margin"]},
            {"Metric": "Credit Score", "Value": profile["credit_score"]},
            {"Metric": "Market Volatility (1-10)", "Value": profile["market_volatility"]},
            {"Metric": "Credit Exposure", "Value": profile["credit_exposure"]},
            {"Metric": "External Rating", "Value": profile["external_rating"]},
            {"Metric": "Market Trend", "Value": profile["market_trend"]},
        ]
    )


def component_dataframe(components):
    rows = []
    for name, (score, weight) in components.items():
        rows.append(
            {
                "Risk Driver": name,
                "Component Score": round(score),
                "Weight": weight,
                "Weighted Contribution": round(score * weight, 1),
            }
        )
    return pd.DataFrame(rows)


def local_assessment(profile):
    score, components = score_financial_profile(profile)
    band = risk_band(score)
    ranked = sorted(
        components.items(),
        key=lambda item: item[1][0] * item[1][1],
        reverse=True,
    )
    top_drivers = [name for name, _ in ranked[:3]]

    recommendations = []
    if profile["liquidity_ratio"] < 1:
        recommendations.append("Improve short-term liquidity before increasing exposure.")
    if profile["debt_to_equity"] > 2:
        recommendations.append("Limit new borrowing or require a deleveraging plan.")
    if profile["revenue_growth"] < 0:
        recommendations.append("Validate revenue recovery assumptions and monitor customer concentration.")
    if profile["market_volatility"] >= 7:
        recommendations.append("Use tighter covenants, collateral, or staged approvals while volatility remains elevated.")
    if profile["credit_score"] < 660:
        recommendations.append("Request updated credit documentation and strengthen repayment controls.")
    if not recommendations:
        recommendations.append("Maintain routine monitoring and reassess if market or credit conditions deteriorate.")

    driver_text = ", ".join(top_drivers).lower()
    recommendation_text = "\n".join(f"- {item}" for item in recommendations)

    return f"""
### Risk Level: {band.upper()}

The profile scores {score}/100. The largest modeled drivers are {driver_text}, which indicates that the proposed decision should be reviewed with clear limits and monitoring.

### Rationale
- Entity: {profile['entity_name']} in {profile['sector']}.
- Decision: {profile['decision']}.
- Credit exposure: ${profile['credit_exposure']:,.0f}.
- External rating: {profile['external_rating']}; market context: {profile['market_trend']}.

### Recommendations
{recommendation_text}
""".strip()


def build_prompt(profile, score, components):
    metrics_table = profile_to_dataframe(profile).to_string(index=False)
    component_table = component_dataframe(components).to_string(index=False)
    band = risk_band(score)

    return f"""
You are a financial risk analyst AI.

Assess the risk of this decision, transaction, or entity using the financial metrics and model scores below.

Entity: {profile['entity_name']}
Sector: {profile['sector']}
Decision or transaction: {profile['decision']}
Modeled risk score: {score}/100
Modeled risk band: {band}

Financial metrics:
{metrics_table}

Risk driver model:
{component_table}

Please:
1. Classify the risk level as Low, Medium, or High.
2. Justify the assessment using the most important metrics.
3. Recommend concrete actions to reduce or monitor risk.

Keep the response concise, practical, and educational. Avoid guarantees.
""".strip()


def openai_assessment(profile, model):
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("OPENAI_API_KEY is not set. Using local risk assessment instead.")
        return None

    score, components = score_financial_profile(profile)
    prompt = build_prompt(profile, score, components)

    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=model, temperature=0.3)
        response = llm.invoke(prompt)
        return response.content
    except Exception as exc:
        st.warning(f"OpenAI response unavailable, using local risk assessment instead. Details: {exc}")
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


def ollama_assessment(profile, base_url, model):
    if not model:
        st.warning("No Ollama model selected. Using local risk assessment instead.")
        return None

    score, components = score_financial_profile(profile)
    generate_url = f"{base_url.rstrip('/')}/api/generate"
    payload =     {
        "model": model,
        "prompt": build_prompt(profile, score, components),
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
        st.warning(f"Ollama response unavailable, using local risk assessment instead. Details: {exc}")
        return None


def generate_assessment(profile, provider, openai_model, ollama_base_url, ollama_model):
    if provider == "OpenAI":
        return openai_assessment(profile, openai_model)
    if provider == "Ollama":
        return ollama_assessment(profile, ollama_base_url, ollama_model)
    return None


def render_provider_controls():
    st.header("AI Provider")
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


def render_profile_table(profile):
    display_df = profile_to_dataframe(profile)
    st.dataframe(display_df, hide_index=True, use_container_width=True)


def render_component_table(components):
    display_df = component_dataframe(components)
    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Component Score": st.column_config.ProgressColumn(
                "Component Score",
                min_value=0,
                max_value=100,
                format="%d",
            ),
            "Weight": st.column_config.NumberColumn("Weight", format="%.0%%"),
            "Weighted Contribution": st.column_config.NumberColumn(
                "Weighted Contribution",
                format="%.1f",
            ),
        },
    )


def main():
    st.set_page_config(
        page_title="Financial Risk Assessment Agent",
        page_icon="$",
        layout="wide",
    )
    st.title("Financial Risk Assessment Agent")

    with st.sidebar:
        provider, openai_model, ollama_base_url, ollama_model = render_provider_controls()

        st.header("Risk Profile")
        entity_name = st.text_input("Entity Name", value=DEFAULT_PROFILE["entity_name"])
        sector = st.text_input("Sector", value=DEFAULT_PROFILE["sector"])
        decision = st.text_area("Decision or Transaction", value=DEFAULT_PROFILE["decision"], height=80)
        credit_exposure = st.number_input(
            "Credit Exposure",
            min_value=0,
            value=DEFAULT_PROFILE["credit_exposure"],
            step=50_000,
            format="%d",
        )
        external_rating = st.selectbox(
            "External Rating",
            list(RATING_RISK.keys()),
            index=list(RATING_RISK.keys()).index(DEFAULT_PROFILE["external_rating"]),
        )
        market_trend = st.text_area("Market Trend", value=DEFAULT_PROFILE["market_trend"], height=80)

        st.header("Financial Metrics")
        debt_to_equity = st.number_input(
            "Debt-to-Equity Ratio",
            min_value=0.0,
            max_value=10.0,
            value=DEFAULT_PROFILE["debt_to_equity"],
            step=0.1,
        )
        liquidity_ratio = st.number_input(
            "Liquidity Ratio",
            min_value=0.0,
            max_value=5.0,
            value=DEFAULT_PROFILE["liquidity_ratio"],
            step=0.05,
        )
        revenue_growth = st.slider(
            "Revenue Growth YoY (%)",
            min_value=-30.0,
            max_value=40.0,
            value=DEFAULT_PROFILE["revenue_growth"],
            step=0.5,
        )
        net_margin = st.slider(
            "Net Margin (%)",
            min_value=-25.0,
            max_value=40.0,
            value=DEFAULT_PROFILE["net_margin"],
            step=0.5,
        )
        credit_score = st.slider(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=DEFAULT_PROFILE["credit_score"],
            step=5,
        )
        market_volatility = st.slider(
            "Market Volatility (1-10)",
            min_value=1,
            max_value=10,
            value=DEFAULT_PROFILE["market_volatility"],
            step=1,
        )

    profile = build_financial_profile(
        entity_name,
        sector,
        decision,
        debt_to_equity,
        liquidity_ratio,
        revenue_growth,
        net_margin,
        credit_score,
        market_volatility,
        credit_exposure,
        external_rating,
        market_trend,
    )
    score, components = score_financial_profile(profile)
    band = risk_band(score)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Risk Level", band)
    metric_cols[1].metric("Risk Score", f"{score}/100")
    metric_cols[2].metric("Credit Exposure", f"${profile['credit_exposure']:,.0f}")
    metric_cols[3].metric("External Rating", profile["external_rating"])

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Financial Profile")
        render_profile_table(profile)

    with right:
        st.subheader("Risk Driver Scores")
        render_component_table(components)

    st.subheader("AI Risk Assessment")
    if st.button("Run Risk Analysis", type="primary"):
        with st.spinner("Generating risk assessment..."):
            assessment = generate_assessment(
                profile,
                provider,
                openai_model,
                ollama_base_url,
                ollama_model,
            )
            if assessment is None:
                assessment = local_assessment(profile)
            st.session_state.risk_assessment = assessment
            st.session_state.risk_assessment_time = datetime.now().strftime("%I:%M %p")

    if st.session_state.get("risk_assessment"):
        st.caption(f"Last assessment generated at {st.session_state.risk_assessment_time}")
        st.markdown(st.session_state.risk_assessment)
    else:
        st.info("Run the analysis to classify the profile and generate recommendations.")

    st.caption("This lab app uses simulated inputs and educational analysis, not professional financial advice.")


if __name__ == "__main__":
    main()
