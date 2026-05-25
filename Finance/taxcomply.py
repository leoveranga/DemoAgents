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

DEFAULT_RECORDS = [
    {"Category": "Revenue", "Amount": 120000.0},
    {"Category": "COGS", "Amount": 40000.0},
    {"Category": "Operating Expenses", "Amount": 25000.0},
    {"Category": "VAT Collected", "Amount": 8000.0},
    {"Category": "VAT Paid", "Amount": 3500.0},
    {"Category": "Payroll Tax", "Amount": 7000.0},
    {"Category": "Corporate Income Tax Paid", "Amount": 5000.0},
]

DEFAULT_ASSUMPTIONS = {
    "entity_name": "Contoso Manufacturing",
    "jurisdiction": "US multi-state / VAT simulation",
    "vat_rate": 10.0,
    "corporate_tax_rate": 21.0,
    "payroll_tax_rate": 7.65,
    "payroll_taxable_wages": 90000.0,
    "revenue_registration_threshold": 100000.0,
    "materiality_threshold": 1000.0,
    "notes": "Quarterly review using simulated accounting records.",
}


def apply_dark_theme():
    st.markdown(
        """
        <style>
        :root {
            color-scheme: dark;
        }

        .stApp {
            background: #0e1117;
            color: #f3f6fb;
        }

        [data-testid="stSidebar"] {
            background: #151923;
        }

        [data-testid="stMetric"],
        div[data-testid="stAlert"],
        div[data-testid="stDataFrame"] {
            background-color: #151923;
            border: 1px solid #2a3140;
            border-radius: 8px;
        }

        h1, h2, h3, label, p, span {
            color: #f3f6fb;
        }

        .stButton > button {
            border-radius: 8px;
            border: 1px solid #3b4355;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def default_records_df():
    return pd.DataFrame(DEFAULT_RECORDS)


def clean_records(records_df):
    df = records_df.copy()
    if "Category" not in df.columns or "Amount" not in df.columns:
        return default_records_df()

    df = df[["Category", "Amount"]]
    df["Category"] = df["Category"].astype(str).str.strip()
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
    df = df[df["Category"] != ""]
    return df.reset_index(drop=True)


def get_amount(records_df, category):
    match = records_df.loc[
        records_df["Category"].str.lower() == category.lower(),
        "Amount",
    ]
    if match.empty:
        return 0.0
    return float(match.iloc[0])


def build_assumptions(
    entity_name,
    jurisdiction,
    vat_rate,
    corporate_tax_rate,
    payroll_tax_rate,
    payroll_taxable_wages,
    revenue_registration_threshold,
    materiality_threshold,
    notes,
):
    return {
        "entity_name": entity_name.strip() or DEFAULT_ASSUMPTIONS["entity_name"],
        "jurisdiction": jurisdiction.strip() or DEFAULT_ASSUMPTIONS["jurisdiction"],
        "vat_rate": float(vat_rate),
        "corporate_tax_rate": float(corporate_tax_rate),
        "payroll_tax_rate": float(payroll_tax_rate),
        "payroll_taxable_wages": float(payroll_taxable_wages),
        "revenue_registration_threshold": float(revenue_registration_threshold),
        "materiality_threshold": float(materiality_threshold),
        "notes": notes.strip() or DEFAULT_ASSUMPTIONS["notes"],
    }


def analyze_records(records_df, assumptions):
    revenue = get_amount(records_df, "Revenue")
    cogs = get_amount(records_df, "COGS")
    operating_expenses = get_amount(records_df, "Operating Expenses")
    vat_collected = get_amount(records_df, "VAT Collected")
    vat_paid = get_amount(records_df, "VAT Paid")
    payroll_tax = get_amount(records_df, "Payroll Tax")
    corporate_tax_paid = get_amount(records_df, "Corporate Income Tax Paid")

    net_profit = revenue - cogs - operating_expenses
    expected_vat_collected = revenue * assumptions["vat_rate"] / 100
    net_vat_due = vat_collected - vat_paid
    expected_corporate_tax = max(net_profit, 0) * assumptions["corporate_tax_rate"] / 100
    corporate_tax_gap = expected_corporate_tax - corporate_tax_paid
    expected_payroll_tax = assumptions["payroll_taxable_wages"] * assumptions["payroll_tax_rate"] / 100
    payroll_tax_gap = expected_payroll_tax - payroll_tax

    issues = []
    materiality = assumptions["materiality_threshold"]

    if revenue >= assumptions["revenue_registration_threshold"] and vat_collected <= 0:
        issues.append(("High", "Revenue exceeds the registration threshold but no VAT was collected."))
    if abs(expected_vat_collected - vat_collected) > materiality:
        issues.append(
            (
                "Medium",
                f"VAT collected differs from the configured {assumptions['vat_rate']:.2f}% rate by ${expected_vat_collected - vat_collected:,.0f}.",
            )
        )
    if net_vat_due > materiality:
        issues.append(("Medium", f"Net VAT payable is ${net_vat_due:,.0f}; confirm it was remitted on time."))
    if corporate_tax_gap > materiality:
        issues.append(
            (
                "High",
                f"Corporate income tax paid is ${corporate_tax_gap:,.0f} below the estimate from net profit.",
            )
        )
    if payroll_tax_gap > materiality:
        issues.append(("Medium", f"Payroll tax appears underpaid by about ${payroll_tax_gap:,.0f}."))
    if not issues:
        issues.append(("Low", "No material exception was identified under the configured assumptions."))

    high_count = sum(1 for severity, _ in issues if severity == "High")
    medium_count = sum(1 for severity, _ in issues if severity == "Medium")
    if high_count:
        risk_level = "High"
    elif medium_count:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    metrics = {
        "revenue": revenue,
        "cogs": cogs,
        "operating_expenses": operating_expenses,
        "net_profit": net_profit,
        "vat_collected": vat_collected,
        "vat_paid": vat_paid,
        "expected_vat_collected": expected_vat_collected,
        "net_vat_due": net_vat_due,
        "corporate_tax_paid": corporate_tax_paid,
        "expected_corporate_tax": expected_corporate_tax,
        "corporate_tax_gap": corporate_tax_gap,
        "payroll_tax": payroll_tax,
        "expected_payroll_tax": expected_payroll_tax,
        "payroll_tax_gap": payroll_tax_gap,
        "risk_level": risk_level,
        "issue_count": len(issues),
    }
    return metrics, issues


def compliance_summary_df(metrics):
    rows = [
        {"Check": "Net Profit", "Amount": metrics["net_profit"], "Status": "Calculated"},
        {"Check": "Expected VAT Collected", "Amount": metrics["expected_vat_collected"], "Status": "Benchmark"},
        {"Check": "Net VAT Due", "Amount": metrics["net_vat_due"], "Status": "Review"},
        {"Check": "Expected Corporate Tax", "Amount": metrics["expected_corporate_tax"], "Status": "Benchmark"},
        {"Check": "Corporate Tax Gap", "Amount": metrics["corporate_tax_gap"], "Status": "Review"},
        {"Check": "Expected Payroll Tax", "Amount": metrics["expected_payroll_tax"], "Status": "Benchmark"},
        {"Check": "Payroll Tax Gap", "Amount": metrics["payroll_tax_gap"], "Status": "Review"},
    ]
    return pd.DataFrame(rows)


def local_audit_report(records_df, assumptions, metrics, issues):
    issue_lines = "\n".join(f"- {severity}: {message}" for severity, message in issues)
    recommendations = []

    if metrics["net_vat_due"] > assumptions["materiality_threshold"]:
        recommendations.append("Reconcile VAT returns to collections and input credits, then confirm remittance timing.")
    if metrics["corporate_tax_gap"] > assumptions["materiality_threshold"]:
        recommendations.append("Recalculate estimated corporate income tax and prepare an adjusting payment or accrual.")
    if metrics["payroll_tax_gap"] > assumptions["materiality_threshold"]:
        recommendations.append("Validate taxable wage base, payroll filings, and employer tax deposits.")
    if metrics["vat_collected"] <= 0 and metrics["revenue"] >= assumptions["revenue_registration_threshold"]:
        recommendations.append("Review tax registration obligations before the next filing period.")
    if not recommendations:
        recommendations.append("Maintain supporting schedules and continue routine filing review.")

    recommendation_text = "\n".join(f"- {item}" for item in recommendations)

    return f"""
### Audit Risk: {metrics['risk_level'].upper()}

{assumptions['entity_name']} reports revenue of ${metrics['revenue']:,.0f}, net profit of ${metrics['net_profit']:,.0f}, VAT collected of ${metrics['vat_collected']:,.0f}, and corporate income tax paid of ${metrics['corporate_tax_paid']:,.0f}.

### Findings
{issue_lines}

### Recommended Corrections
{recommendation_text}

### Audit Readiness
Keep source invoices, VAT remittance confirmations, payroll tax deposit records, and corporate tax workpapers tied to the accounting ledger. This review is based on simulated records and configured assumptions for {assumptions['jurisdiction']}.
""".strip()


def build_prompt(records_df, assumptions, metrics, issues):
    findings = "\n".join(f"- {severity}: {message}" for severity, message in issues)
    summary_table = compliance_summary_df(metrics).to_string(index=False)

    return f"""
You are a tax compliance AI auditor.

Prepare an audit-ready tax compliance summary for this company.

Entity: {assumptions['entity_name']}
Jurisdiction context: {assumptions['jurisdiction']}
Review notes: {assumptions['notes']}

Financial records:
{records_df.to_string(index=False)}

Configured assumptions:
- VAT / sales tax rate: {assumptions['vat_rate']:.2f}%
- Corporate income tax rate: {assumptions['corporate_tax_rate']:.2f}%
- Payroll tax rate: {assumptions['payroll_tax_rate']:.2f}%
- Payroll taxable wages: ${assumptions['payroll_taxable_wages']:,.0f}
- Tax registration revenue threshold: ${assumptions['revenue_registration_threshold']:,.0f}
- Materiality threshold: ${assumptions['materiality_threshold']:,.0f}

Calculated compliance checks:
{summary_table}

Preliminary findings:
{findings}

Please:
1. Identify discrepancies in VAT or sales tax reporting.
2. Assess whether corporate income tax paid is reasonable based on net profit.
3. Flag payroll or registration risks.
4. Recommend corrective actions.

Use a concise, audit-ready tone. Do not provide legal advice.
""".strip()


def openai_audit(records_df, assumptions, metrics, issues, model):
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("OPENAI_API_KEY is not set. Using local tax audit logic instead.")
        return None

    prompt = build_prompt(records_df, assumptions, metrics, issues)

    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=model, temperature=0.2)
        response = llm.invoke(prompt)
        return response.content
    except Exception as exc:
        st.warning(f"OpenAI response unavailable, using local tax audit logic instead. Details: {exc}")
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


def ollama_audit(records_df, assumptions, metrics, issues, base_url, model):
    if not model:
        st.warning("No Ollama model selected. Using local tax audit logic instead.")
        return None

    generate_url = f"{base_url.rstrip('/')}/api/generate"
    payload =     {
        "model": model,
        "prompt": build_prompt(records_df, assumptions, metrics, issues),
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
        st.warning(f"Ollama response unavailable, using local tax audit logic instead. Details: {exc}")
        return None


def generate_audit(records_df, assumptions, provider, openai_model, ollama_base_url, ollama_model):
    metrics, issues = analyze_records(records_df, assumptions)
    if provider == "OpenAI":
        report = openai_audit(records_df, assumptions, metrics, issues, openai_model)
    elif provider == "Ollama":
        report = ollama_audit(records_df, assumptions, metrics, issues, ollama_base_url, ollama_model)
    else:
        report = None

    if report is None:
        report = local_audit_report(records_df, assumptions, metrics, issues)
    return metrics, issues, report


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


def render_records_editor():
    st.subheader("Financial Records")
    edited = st.data_editor(
        default_records_df(),
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        column_config={
            "Category": st.column_config.TextColumn("Category", required=True),
            "Amount": st.column_config.NumberColumn("Amount", min_value=0.0, format="$%.2f"),
        },
    )
    return clean_records(edited)


def render_summary_table(metrics):
    st.dataframe(
        compliance_summary_df(metrics),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Amount": st.column_config.NumberColumn("Amount", format="$%.0f"),
        },
    )


def render_findings(issues):
    findings_df = pd.DataFrame(
        [{"Severity": severity, "Finding": message} for severity, message in issues]
    )
    st.dataframe(findings_df, hide_index=True, use_container_width=True)


def main():
    st.set_page_config(
        page_title="Tax Compliance Agent",
        page_icon="$",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_dark_theme()
    st.title("Tax Compliance Agent")

    with st.sidebar:
        provider, openai_model, ollama_base_url, ollama_model = render_provider_controls()

        st.header("Audit Assumptions")
        entity_name = st.text_input("Entity Name", value=DEFAULT_ASSUMPTIONS["entity_name"])
        jurisdiction = st.text_input("Jurisdiction", value=DEFAULT_ASSUMPTIONS["jurisdiction"])
        vat_rate = st.number_input(
            "VAT / Sales Tax Rate (%)",
            min_value=0.0,
            max_value=30.0,
            value=DEFAULT_ASSUMPTIONS["vat_rate"],
            step=0.25,
        )
        corporate_tax_rate = st.number_input(
            "Corporate Income Tax Rate (%)",
            min_value=0.0,
            max_value=50.0,
            value=DEFAULT_ASSUMPTIONS["corporate_tax_rate"],
            step=0.25,
        )
        payroll_tax_rate = st.number_input(
            "Payroll Tax Rate (%)",
            min_value=0.0,
            max_value=25.0,
            value=DEFAULT_ASSUMPTIONS["payroll_tax_rate"],
            step=0.05,
        )
        payroll_taxable_wages = st.number_input(
            "Payroll Taxable Wages",
            min_value=0.0,
            value=DEFAULT_ASSUMPTIONS["payroll_taxable_wages"],
            step=1000.0,
            format="%.0f",
        )
        revenue_registration_threshold = st.number_input(
            "Revenue Registration Threshold",
            min_value=0.0,
            value=DEFAULT_ASSUMPTIONS["revenue_registration_threshold"],
            step=5000.0,
            format="%.0f",
        )
        materiality_threshold = st.number_input(
            "Materiality Threshold",
            min_value=0.0,
            value=DEFAULT_ASSUMPTIONS["materiality_threshold"],
            step=250.0,
            format="%.0f",
        )
        notes = st.text_area("Review Notes", value=DEFAULT_ASSUMPTIONS["notes"], height=90)

    assumptions = build_assumptions(
        entity_name,
        jurisdiction,
        vat_rate,
        corporate_tax_rate,
        payroll_tax_rate,
        payroll_taxable_wages,
        revenue_registration_threshold,
        materiality_threshold,
        notes,
    )

    records_df = render_records_editor()
    metrics, issues = analyze_records(records_df, assumptions)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Audit Risk", metrics["risk_level"])
    metric_cols[1].metric("Net Profit", f"${metrics['net_profit']:,.0f}")
    metric_cols[2].metric("Net VAT Due", f"${metrics['net_vat_due']:,.0f}")
    metric_cols[3].metric("Corporate Tax Gap", f"${metrics['corporate_tax_gap']:,.0f}")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Compliance Checks")
        render_summary_table(metrics)

    with right:
        st.subheader("Preliminary Findings")
        render_findings(issues)

    st.subheader("AI Tax Compliance Report")
    if st.button("Audit Tax Records", type="primary"):
        with st.spinner("Generating tax compliance report..."):
            metrics, issues, report = generate_audit(
                records_df,
                assumptions,
                provider,
                openai_model,
                ollama_base_url,
                ollama_model,
            )
            st.session_state.tax_audit_report = report
            st.session_state.tax_audit_time = datetime.now().strftime("%I:%M %p")

    if st.session_state.get("tax_audit_report"):
        st.caption(f"Last audit generated at {st.session_state.tax_audit_time}")
        st.markdown(st.session_state.tax_audit_report)
    else:
        st.info("Run the audit to generate a concise report with findings and corrective actions.")

    st.caption("This lab app uses simulated records and educational analysis, not professional tax or legal advice.")


if __name__ == "__main__":
    main()
