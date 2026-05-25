import json
import os
import urllib.error
import urllib.request
from datetime import datetime

import altair as alt
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

DEFAULT_PORTFOLIO = [
    {
        "Asset": "US Stocks",
        "Allocation (%)": 40.0,
        "Annual Return (%)": 8.0,
        "Risk Score (1-10)": 6.0,
    },
    {
        "Asset": "International Stocks",
        "Allocation (%)": 25.0,
        "Annual Return (%)": 6.5,
        "Risk Score (1-10)": 7.0,
    },
    {
        "Asset": "Bonds",
        "Allocation (%)": 20.0,
        "Annual Return (%)": 3.5,
        "Risk Score (1-10)": 2.0,
    },
    {
        "Asset": "Real Estate",
        "Allocation (%)": 10.0,
        "Annual Return (%)": 5.0,
        "Risk Score (1-10)": 5.0,
    },
    {
        "Asset": "Crypto",
        "Allocation (%)": 5.0,
        "Annual Return (%)": 20.0,
        "Risk Score (1-10)": 9.0,
    },
]

TARGET_ALLOCATIONS = {
    "Conservative": {"growth": 35, "defensive": 55, "alternatives": 10},
    "Balanced": {"growth": 60, "defensive": 30, "alternatives": 10},
    "Growth": {"growth": 75, "defensive": 15, "alternatives": 10},
    "Aggressive": {"growth": 85, "defensive": 5, "alternatives": 10},
}


def apply_dark_theme():
    st.markdown(
        """
        <style>
        :root {
            color-scheme: dark;
        }
        .stApp {
            background: #0b1018;
            color: #eef3f8;
        }
        [data-testid="stSidebar"] {
            background: #111827;
            border-right: 1px solid #263244;
        }
        [data-testid="stMetric"],
        [data-testid="stExpander"],
        div[data-testid="stDataFrame"],
        div[data-testid="stDataEditor"] {
            background: #111827;
            border: 1px solid #263244;
            border-radius: 8px;
            padding: 0.4rem;
        }
        h1, h2, h3, h4, h5, h6,
        .stMarkdown, .stCaption, label, p, span {
            color: #eef3f8;
        }
        .stButton > button {
            border-radius: 7px;
            border: 1px solid #3b82f6;
            background: #1d4ed8;
            color: #ffffff;
        }
        .stButton > button:hover {
            border-color: #60a5fa;
            background: #2563eb;
            color: #ffffff;
        }
        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div,
        div[data-baseweb="select"] > div {
            background: #0f172a;
            border-color: #334155;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_portfolio():
    return pd.DataFrame(DEFAULT_PORTFOLIO)


def sanitize_portfolio(df):
    portfolio = df.copy()
    for column in ["Allocation (%)", "Annual Return (%)", "Risk Score (1-10)"]:
        portfolio[column] = pd.to_numeric(portfolio[column], errors="coerce").fillna(0)
    portfolio["Asset"] = portfolio["Asset"].fillna("").astype(str).str.strip()
    portfolio = portfolio[portfolio["Asset"] != ""]
    portfolio["Allocation (%)"] = portfolio["Allocation (%)"].clip(lower=0, upper=100)
    portfolio["Risk Score (1-10)"] = portfolio["Risk Score (1-10)"].clip(lower=1, upper=10)
    return portfolio.reset_index(drop=True)


def classify_asset(asset_name):
    name = asset_name.lower()
    if any(token in name for token in ["bond", "treasury", "cash", "money market", "cd"]):
        return "defensive"
    if any(token in name for token in ["real estate", "reit", "crypto", "commodity", "gold"]):
        return "alternatives"
    return "growth"


def portfolio_metrics(df, portfolio_value):
    total_allocation = float(df["Allocation (%)"].sum())
    if total_allocation <= 0:
        return {
            "total_allocation": 0.0,
            "expected_return": 0.0,
            "weighted_risk": 0.0,
            "projected_gain": 0.0,
            "high_risk_allocation": 0.0,
            "largest_position": "None",
            "largest_position_allocation": 0.0,
            "categories": pd.DataFrame(columns=["Category", "Allocation (%)"]),
        }

    weights = df["Allocation (%)"] / total_allocation
    expected_return = float((weights * df["Annual Return (%)"]).sum())
    weighted_risk = float((weights * df["Risk Score (1-10)"]).sum())
    high_risk_allocation = float(df.loc[df["Risk Score (1-10)"] >= 8, "Allocation (%)"].sum())
    largest_row = df.loc[df["Allocation (%)"].idxmax()]

    categorized = df.copy()
    categorized["Category"] = categorized["Asset"].apply(classify_asset)
    categories = (
        categorized.groupby("Category", as_index=False)["Allocation (%)"]
        .sum()
        .sort_values("Allocation (%)", ascending=False)
    )

    return {
        "total_allocation": total_allocation,
        "expected_return": expected_return,
        "weighted_risk": weighted_risk,
        "projected_gain": portfolio_value * (expected_return / 100),
        "high_risk_allocation": high_risk_allocation,
        "largest_position": str(largest_row["Asset"]),
        "largest_position_allocation": float(largest_row["Allocation (%)"]),
        "categories": categories,
    }


def risk_band(weighted_risk):
    if weighted_risk >= 7:
        return "High"
    if weighted_risk >= 4.5:
        return "Moderate"
    return "Low"


def local_portfolio_guidance(df, metrics, risk_profile, objective, constraints):
    band = risk_band(metrics["weighted_risk"])
    category_allocations = {
        row["Category"]: row["Allocation (%)"]
        for _, row in metrics["categories"].iterrows()
    }
    target = TARGET_ALLOCATIONS[risk_profile]
    high_risk_assets = df[df["Risk Score (1-10)"] >= 8].sort_values(
        "Allocation (%)",
        ascending=False,
    )
    low_return_assets = df[df["Annual Return (%)"] <= 4].sort_values(
        "Allocation (%)",
        ascending=False,
    )

    recommendations = []
    high_risk_allocation = metrics["high_risk_allocation"]
    if high_risk_allocation > 10 and risk_profile in ["Conservative", "Balanced"]:
        recommendations.append(
            f"Reduce assets scoring 8 or higher in risk from {high_risk_allocation:.1f}% toward a single-digit allocation."
        )
    if metrics["largest_position_allocation"] > 35:
        recommendations.append(
            f"Trim concentration in {metrics['largest_position']} or rebalance around it to avoid one position dominating outcomes."
        )
    defensive_gap = target["defensive"] - category_allocations.get("defensive", 0)
    if defensive_gap > 10:
        recommendations.append(
            f"Add roughly {defensive_gap:.0f}% to defensive assets such as bonds, cash, or short-duration fixed income."
        )
    growth_gap = category_allocations.get("growth", 0) - target["growth"]
    if growth_gap > 15:
        recommendations.append(
            f"Growth exposure is about {growth_gap:.0f}% above the {risk_profile.lower()} target; consider shifting part of it to defensive assets."
        )
    if not recommendations:
        recommendations.append(
            "Keep the current strategic mix, then rebalance quarterly when any sleeve drifts more than 5 percentage points from target."
        )

    high_risk_text = "None above the threshold"
    if not high_risk_assets.empty:
        high_risk_text = ", ".join(high_risk_assets["Asset"].head(3).tolist())

    low_return_text = "None material"
    if not low_return_assets.empty:
        low_return_text = ", ".join(low_return_assets["Asset"].head(3).tolist())

    recommendation_text = "\n".join(f"- {item}" for item in recommendations)
    constraints_text = constraints.strip() or "No special constraints provided."

    return f"""
### Portfolio Health: {band} Risk

The portfolio is allocated {metrics['total_allocation']:.1f}% across tracked assets, with an estimated weighted return of {metrics['expected_return']:.1f}% and a weighted risk score of {metrics['weighted_risk']:.1f}/10. The largest position is {metrics['largest_position']} at {metrics['largest_position_allocation']:.1f}%.

### Risk and Return Notes
- High-risk exposure: {metrics['high_risk_allocation']:.1f}%.
- Highest-risk assets: {high_risk_text}.
- Lower-return holdings to review: {low_return_text}.
- Investor profile: {risk_profile}; objective: {objective}.
- Constraints: {constraints_text}

### Rebalancing Strategy
{recommendation_text}

### Next Quarter
Review drift, rebalance with new contributions first, and avoid forced sales unless risk exposure is outside the target range. This is educational analysis, not personalized investment advice.
""".strip()


def build_prompt(df, metrics, risk_profile, objective, constraints, portfolio_value):
    portfolio_table = df.to_string(index=False)
    category_table = metrics["categories"].to_string(index=False)
    target = TARGET_ALLOCATIONS[risk_profile]

    return f"""
You are an AI investment advisor.

Analyze the simulated investment portfolio below and provide educational guidance.

Portfolio value: ${portfolio_value:,.0f}
Risk profile: {risk_profile}
Objective: {objective}
Constraints: {constraints or "No special constraints provided"}

Portfolio holdings:
{portfolio_table}

Category allocation:
{category_table}

Computed metrics:
- Total allocation: {metrics['total_allocation']:.1f}%
- Weighted expected annual return: {metrics['expected_return']:.2f}%
- Weighted risk score: {metrics['weighted_risk']:.2f}/10
- High-risk allocation, risk score >= 8: {metrics['high_risk_allocation']:.1f}%
- Largest position: {metrics['largest_position']} at {metrics['largest_position_allocation']:.1f}%

Target category guide for this risk profile:
- Growth: {target['growth']}%
- Defensive: {target['defensive']}%
- Alternatives: {target['alternatives']}%

Please:
1. Evaluate the risk versus return balance.
2. Identify concentration or diversification concerns.
3. Recommend concrete rebalancing steps for the next quarter.
4. Keep the response concise and professional.
5. State that the response is educational and not personalized investment advice.
""".strip()


def openai_guidance(df, metrics, risk_profile, objective, constraints, portfolio_value, model):
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("OPENAI_API_KEY is not set. Using local portfolio guidance instead.")
        return None

    prompt = build_prompt(df, metrics, risk_profile, objective, constraints, portfolio_value)

    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=model, temperature=0.3)
        response = llm.invoke(prompt)
        return response.content
    except Exception as exc:
        st.warning(f"OpenAI response unavailable, using local portfolio guidance instead. Details: {exc}")
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


def ollama_guidance(
    df,
    metrics,
    risk_profile,
    objective,
    constraints,
    portfolio_value,
    base_url,
    model,
):
    if not model:
        st.warning("No Ollama model selected. Using local portfolio guidance instead.")
        return None

    generate_url = f"{base_url.rstrip('/')}/api/generate"
    payload =     {
        "model": model,
        "prompt": build_prompt(
            df,
            metrics,
            risk_profile,
            objective,
            constraints,
            portfolio_value,
        ),
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
        st.warning(f"Ollama response unavailable, using local portfolio guidance instead. Details: {exc}")
        return None


def generate_guidance(
    df,
    metrics,
    risk_profile,
    objective,
    constraints,
    portfolio_value,
    provider,
    openai_model,
    ollama_base_url,
    ollama_model,
):
    if provider == "OpenAI":
        return openai_guidance(
            df,
            metrics,
            risk_profile,
            objective,
            constraints,
            portfolio_value,
            openai_model,
        )
    if provider == "Ollama":
        return ollama_guidance(
            df,
            metrics,
            risk_profile,
            objective,
            constraints,
            portfolio_value,
            ollama_base_url,
            ollama_model,
        )
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


def render_portfolio_editor(df):
    edited_df = st.data_editor(
        df,
        hide_index=True,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "Asset": st.column_config.TextColumn("Asset", required=True),
            "Allocation (%)": st.column_config.NumberColumn(
                "Allocation (%)",
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                format="%.1f",
            ),
            "Annual Return (%)": st.column_config.NumberColumn(
                "Annual Return (%)",
                min_value=-50.0,
                max_value=100.0,
                step=0.5,
                format="%.1f",
            ),
            "Risk Score (1-10)": st.column_config.NumberColumn(
                "Risk Score (1-10)",
                min_value=1.0,
                max_value=10.0,
                step=0.5,
                format="%.1f",
            ),
        },
    )
    return sanitize_portfolio(edited_df)


def render_allocation_chart(df):
    chart = (
        alt.Chart(df)
        .mark_arc(innerRadius=65)
        .encode(
            theta=alt.Theta("Allocation (%):Q", stack=True),
            color=alt.Color("Asset:N", legend=alt.Legend(orient="bottom")),
            tooltip=[
                alt.Tooltip("Asset:N"),
                alt.Tooltip("Allocation (%):Q", format=".1f"),
                alt.Tooltip("Annual Return (%):Q", format=".1f"),
                alt.Tooltip("Risk Score (1-10):Q", format=".1f"),
            ],
        )
        .properties(height=330)
    )
    st.altair_chart(chart, use_container_width=True)


def render_return_risk_chart(df):
    chart = (
        alt.Chart(df)
        .mark_circle(size=280, opacity=0.85)
        .encode(
            x=alt.X("Risk Score (1-10):Q", scale=alt.Scale(domain=[1, 10])),
            y=alt.Y("Annual Return (%):Q"),
            size=alt.Size("Allocation (%):Q", legend=alt.Legend(title="Allocation")),
            color=alt.Color("Asset:N", legend=None),
            tooltip=[
                alt.Tooltip("Asset:N"),
                alt.Tooltip("Allocation (%):Q", format=".1f"),
                alt.Tooltip("Annual Return (%):Q", format=".1f"),
                alt.Tooltip("Risk Score (1-10):Q", format=".1f"),
            ],
        )
        .properties(height=330)
    )
    st.altair_chart(chart, use_container_width=True)


def initialize_state():
    if "portfolio_df" not in st.session_state:
        st.session_state.portfolio_df = load_portfolio()


def main():
    st.set_page_config(
        page_title="AI Portfolio Manager",
        page_icon="$",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_dark_theme()
    initialize_state()

    st.title("AI Portfolio Manager")

    with st.sidebar:
        provider, openai_model, ollama_base_url, ollama_model = render_provider_controls()

        st.header("Portfolio Settings")
        portfolio_value = st.number_input(
            "Portfolio Value",
            min_value=0,
            value=250_000,
            step=10_000,
            format="%d",
        )
        risk_profile = st.selectbox(
            "Investor Risk Profile",
            list(TARGET_ALLOCATIONS.keys()),
            index=list(TARGET_ALLOCATIONS.keys()).index("Balanced"),
        )
        objective = st.text_input(
            "Primary Objective",
            value="Balanced long-term growth with controlled volatility",
        )
        constraints = st.text_area(
            "Constraints",
            value="Keep crypto exposure modest and preserve quarterly liquidity.",
            height=90,
        )

        if st.button("Reset Portfolio"):
            st.session_state.portfolio_df = load_portfolio()
            st.session_state.pop("portfolio_guidance", None)
            st.rerun()

    st.subheader("Portfolio Inputs")
    portfolio_df = render_portfolio_editor(st.session_state.portfolio_df)
    st.session_state.portfolio_df = portfolio_df

    if portfolio_df.empty:
        st.warning("Add at least one asset to analyze the portfolio.")
        return

    metrics = portfolio_metrics(portfolio_df, portfolio_value)

    if abs(metrics["total_allocation"] - 100) > 0.1:
        st.warning(
            f"Current allocation totals {metrics['total_allocation']:.1f}%. "
            "The analysis normalizes weights for risk and return estimates."
        )

    metric_cols = st.columns(5)
    metric_cols[0].metric("Total Allocation", f"{metrics['total_allocation']:.1f}%")
    metric_cols[1].metric("Expected Return", f"{metrics['expected_return']:.1f}%")
    metric_cols[2].metric("Weighted Risk", f"{metrics['weighted_risk']:.1f}/10")
    metric_cols[3].metric("Projected Annual Gain", f"${metrics['projected_gain']:,.0f}")
    metric_cols[4].metric("High-Risk Exposure", f"{metrics['high_risk_allocation']:.1f}%")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Allocation Distribution")
        render_allocation_chart(portfolio_df)

    with right:
        st.subheader("Risk and Return Map")
        render_return_risk_chart(portfolio_df)

    st.subheader("Category View")
    st.dataframe(
        metrics["categories"],
        hide_index=True,
        use_container_width=True,
        column_config={
            "Allocation (%)": st.column_config.ProgressColumn(
                "Allocation (%)",
                min_value=0,
                max_value=100,
                format="%.1f%%",
            )
        },
    )

    st.subheader("AI Investment Guidance")
    if st.button("Analyze My Portfolio", type="primary"):
        with st.spinner("Generating portfolio guidance..."):
            guidance = generate_guidance(
                portfolio_df,
                metrics,
                risk_profile,
                objective,
                constraints,
                portfolio_value,
                provider,
                openai_model,
                ollama_base_url,
                ollama_model,
            )
            if guidance is None:
                guidance = local_portfolio_guidance(
                    portfolio_df,
                    metrics,
                    risk_profile,
                    objective,
                    constraints,
                )
            st.session_state.portfolio_guidance = guidance
            st.session_state.portfolio_guidance_time = datetime.now().strftime("%I:%M %p")

    if st.session_state.get("portfolio_guidance"):
        st.caption(f"Last analysis generated at {st.session_state.portfolio_guidance_time}")
        st.markdown(st.session_state.portfolio_guidance)
    else:
        st.info("Run the analysis to generate portfolio health and rebalancing guidance.")

    st.caption("This app uses simulated inputs and educational analysis, not professional financial advice.")


if __name__ == "__main__":
    main()
