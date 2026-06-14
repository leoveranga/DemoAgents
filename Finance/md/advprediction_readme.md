# Advanced Prediction Agent README

`advprediction.py` is a Streamlit finance demo that generates monthly revenue history, calculates a local baseline forecast, and optionally asks an LLM to explain the forecast in business terms.

The app is designed for planning demonstrations and educational use. It should not be treated as professional financial advice or a guaranteed forecast.

## File

| Item | Value |
| --- | --- |
| App file | `advprediction.py` |
| App title | `Advanced Prediction Agent` |
| Streamlit page title | `Advanced Prediction Agent` |
| Layout | `wide` |
| Main workflow | Generate revenue history, calculate a baseline forecast, chart actuals and forecast values, then produce a forecast summary. |

## Run Command

From the `AgentCollections` repository root:

```powershell
streamlit run .\Finance\advprediction.py
```

Or through the shared virtual environment:

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\Finance\advprediction.py
```

## Required Python Packages

The app imports:

| Package | Purpose |
| --- | --- |
| `streamlit` | Web UI, sidebar controls, metrics, chart, table, session state, cache. |
| `pandas` | Revenue history and forecast DataFrame construction. |
| `langchain_openai` | OpenAI chat model integration when the OpenAI provider is selected. |
| Standard library: `json`, `os`, `urllib`, `datetime` | Provider payload handling, environment variable lookup, Ollama HTTP calls, default dates. |

`langchain_openai` is only required when using the OpenAI provider.

## Default LLM Settings

| Setting | Default |
| --- | --- |
| Default provider | `Ollama` |
| Default OpenAI model | `gpt-4o-mini` |
| Default Ollama model | `llama3.2:1b` |
| Default Ollama base URL | `http://localhost:11434` |
| OpenAI temperature | `0.3` |
| Ollama temperature | `0.3` |
| Ollama streaming | `false` |
| Local fallback | Enabled automatically when the selected provider is unavailable or returns no summary. |

### Provider Behavior

| Provider | Requirement | Call Path | Fallback Behavior |
| --- | --- | --- | --- |
| `Ollama` | Ollama server running at the configured URL and a selected local model. | `POST {base_url}/api/generate` | Falls back to the deterministic local summary if Ollama cannot be reached, times out, returns invalid JSON, or no model is selected. |
| `OpenAI` | `OPENAI_API_KEY` environment variable and `langchain_openai` package. | `ChatOpenAI(model=<model>, temperature=0.3).invoke(prompt)` | Falls back to the deterministic local summary if the API key is missing or an exception occurs. |
| `Local fallback` | None. | No LLM request. | Uses local deterministic summary only. |

### Ollama Request Structure

```json
{
  "model": "llama3.2:1b",
  "prompt": "<generated forecast prompt>",
  "stream": false,
  "options": {
    "temperature": 0.3
  }
}
```

For supported Ollama reasoning models, the UI can add:

```json
{
  "think": false
}
```

Reasoning-model detection checks whether the selected model name contains one of these markers:

```text
deepseek-r1, gemma4, gpt-oss, qwen3, qwen3.5, reason, thinking, think
```

When such a model is detected, the app shows a `Turn off thinking mode` checkbox. It is checked by default and sends `think: false` to reduce latency and token usage for supported Ollama models.

## Parameter Specification

All user-configurable business inputs are defined in the Streamlit sidebar.

| Parameter | UI Control | Type | Default | Bounds / Options | Used By |
| --- | --- | --- | --- | --- | --- |
| `start_month` | `st.date_input("Start Month")` | Date | `2024-01-01` | Any date accepted by Streamlit. Generated months are normalized to month-end frequency by pandas. | `get_revenue_data()` |
| `historical_months` | `st.slider("Historical Months")` | Integer | `18` | Minimum `6`, maximum `36` | `get_revenue_data()` |
| `forecast_months` | `st.slider("Forecast Months")` | Integer | `3` | Minimum `1`, maximum `12` | `calculate_local_forecast()` and prompt text |
| `starting_revenue` | `st.number_input("Starting Monthly Revenue")` | Number / integer UI | `10000` | Minimum `0`, step `500`, displayed with integer format | Historical revenue generation |
| `monthly_growth` | `st.number_input("Expected Monthly Growth")` | Number / integer UI | `300` | Minimum `-10000`, step `50`, displayed with integer format | Historical revenue generation |
| `volatility` | `st.number_input("Mock Volatility")` | Number / integer UI | `175` | Minimum `0`, step `25`, displayed with integer format | Deterministic synthetic noise |
| `business_context` | `st.text_area("Business Context")` | String | `Revenue is from a subscription business with occasional year-end demand lift.` | Free text, UI height `120` | Prompt and local summary |
| `provider` | `st.radio("Provider")` | String | `Ollama` | `OpenAI`, `Ollama`, `Local fallback` | LLM routing |
| `openai_model` | `st.text_input("OpenAI Model")` | String | `gpt-4o-mini` | Any model name accepted by `ChatOpenAI` | OpenAI provider |
| `ollama_base_url` | `st.text_input("Ollama URL")` | String | `http://localhost:11434` | Any reachable Ollama-compatible base URL | Ollama model listing and generation |
| `ollama_model` | `st.selectbox` or `st.text_input` | String | `llama3.2:1b` | Available Ollama model names, or free text if model listing fails | Ollama provider |
| `ollama_think` | `st.checkbox("Turn off thinking mode")` | Boolean or `None` in session state | `False` for detected reasoning models, otherwise `None` | Shown only for matching reasoning model names | Optional Ollama `think` payload field |

## Internal Data Structure

### Historical Revenue Data

Generated by:

```python
get_revenue_data(start_month, months, starting_revenue, monthly_growth, volatility)
```

Returned structure:

| Column | Type | Description |
| --- | --- | --- |
| `Month` | pandas timestamp | Month-end date generated with `pd.date_range(..., freq="ME")`. |
| `Revenue` | Number | Synthetic monthly revenue rounded to two decimals and floored at `0`. |

Revenue formula per generated month:

```text
seasonal_adjustment =
  starting_revenue * 0.08   for November and December
  starting_revenue * -0.04  for January, February, and July
  0                         otherwise

deterministic_noise = (((index * 37) % 11) - 5) * volatility

Revenue = max(
  round(starting_revenue + (index * monthly_growth) + seasonal_adjustment + deterministic_noise, 2),
  0
)
```

The noise is deterministic, so the same inputs produce the same generated history.

Example logical structure:

```json
[
  {
    "Month": "2024-01-31T00:00:00",
    "Revenue": 8725.0
  },
  {
    "Month": "2024-02-29T00:00:00",
    "Revenue": 10200.0
  }
]
```

### Baseline Forecast Data

Generated by:

```python
calculate_local_forecast(history_df, forecast_months)
```

Returned structure:

| Column | Type | Description |
| --- | --- | --- |
| `Month` | pandas timestamp | Future month-end date after the last historical month. |
| `Forecast Revenue` | Number | Local baseline forecast rounded to two decimals and floored at `0`. |

Forecast calculation:

```text
monthly_deltas = month-over-month differences in historical Revenue
average_growth = mean(monthly_deltas), or 0 when unavailable
recent_growth = mean(last 3 monthly_deltas), or average_growth when fewer than 3 deltas exist
blended_growth = (average_growth * 0.55) + (recent_growth * 0.45)

seasonal_adjustment =
  last_revenue * 0.04    for November and December
  last_revenue * -0.025  for January, February, and July
  0                     otherwise

Forecast Revenue = max(
  round(last_revenue + (blended_growth * forecast_step) + seasonal_adjustment, 2),
  0
)
```

Example logical structure:

```json
[
  {
    "Month": "2025-07-31T00:00:00",
    "Forecast Revenue": 14387.5
  },
  {
    "Month": "2025-08-31T00:00:00",
    "Forecast Revenue": 14950.0
  }
]
```

### Combined Chart Data

Generated by:

```python
build_combined_chart(history_df, forecast_df)
```

Returned structure:

| Column | Type | Description |
| --- | --- | --- |
| `Month` | pandas timestamp | Historical and future month-end dates. |
| `Actual Revenue` | Number or null | Historical revenue values. Future forecast rows are null. |
| `Forecast Revenue` | Number or null | Forecast values. Historical rows are null. |

The app sets `Month` as the chart index before calling `st.line_chart()`.

## Prompt Structure

The app builds an LLM prompt with:

1. Role: financial forecasting analyst.
2. Historical revenue table with `Month` formatted as `YYYY-MM` and `Revenue` formatted as dollars.
3. Baseline statistical forecast table with `Month` formatted as `YYYY-MM` and `Forecast Revenue` formatted as dollars.
4. Business context, or `No additional context provided.` when empty.
5. Tasks:
   - Identify growth patterns, seasonality, or anomalies.
   - Forecast the next selected number of months using the baseline as a guide.
   - Provide practical recommendations for leadership.
6. Style instruction: concise response, clear headings, avoid guarantees.

## Output Data Structure

### Dashboard Metrics

The top-level metrics are rendered from local calculations:

| Metric | Source | Description |
| --- | --- | --- |
| `Latest Revenue` | Last row of `history_df["Revenue"]` | Most recent historical monthly revenue. |
| `Final Forecast` | Last row of `forecast_df["Forecast Revenue"]` | Final month of the selected forecast horizon. |
| Final forecast delta | `(projected_revenue - latest_revenue) / latest_revenue` | Displayed as a percentage next to final forecast. Uses `0` when latest revenue is `0`. |
| `Historical Months` | `historical_months` | Count selected in the sidebar. |
| `Forecast Months` | `forecast_months` | Count selected in the sidebar. |

### Forecast Table Output

Rendered in the `Forecast Values` panel.

| Column | Display Format | Source |
| --- | --- | --- |
| `Month` | `Mon YYYY`, for example `Jul 2025` | `forecast_df["Month"]` |
| `Forecast Revenue` | Currency with no decimals | `forecast_df["Forecast Revenue"]` |

Underlying DataFrame columns remain:

```text
Month, Forecast Revenue
```

### Chart Output

Rendered in the `Revenue Trend` panel with `st.line_chart()`.

Structure:

```text
Index: Month
Columns: Actual Revenue, Forecast Revenue
```

Historical months populate `Actual Revenue`. Forecast months populate `Forecast Revenue`.

### Forecast Summary Output

The summary is rendered with `st.markdown()` and stored in:

```python
st.session_state.forecast_summary
```

When an LLM provider succeeds, the summary is the provider response text.

When no provider succeeds, the local fallback summary uses this markdown structure:

```markdown
### Trend Read
...

### Forecast
...

### Recommendations
...

### Context Note
...
```

`Context Note` appears only when `business_context` is not empty.

### Token Usage Output

When the selected provider is not `Local fallback`, the app attempts to show token usage from:

```python
st.session_state.forecast_token_usage
```

Token usage structure:

```json
{
  "input_tokens": 123,
  "output_tokens": 456,
  "total_tokens": 579
}
```

Token extraction by provider:

| Provider | Input Tokens | Output Tokens | Total Tokens |
| --- | --- | --- | --- |
| OpenAI via LangChain usage metadata | `usage_metadata["input_tokens"]` | `usage_metadata["output_tokens"]` | `usage_metadata["total_tokens"]` |
| OpenAI fallback metadata | `response_metadata["token_usage"]["prompt_tokens"]` | `response_metadata["token_usage"]["completion_tokens"]` | `response_metadata["token_usage"]["total_tokens"]` |
| Ollama | `prompt_eval_count` | `eval_count` | Sum of input and output token counts when both are present |

If any token value is unavailable, the UI displays `Unavailable` for that metric.

## Session State Keys

| Key | Value |
| --- | --- |
| `ollama_think` | `False` when thinking mode is disabled for a detected reasoning model, otherwise `None`. |
| `forecast_summary` | Last generated or fallback markdown summary. Initialized to a local fallback summary before the user runs LLM analysis. |
| `forecast_token_usage` | Last provider token usage dictionary, or `None` when using fallback output. |

## Error Handling and Fallbacks

| Area | Behavior |
| --- | --- |
| Missing `OPENAI_API_KEY` | Shows a Streamlit warning and returns local fallback output. |
| OpenAI exception | Shows a Streamlit warning with exception details and returns local fallback output. |
| Empty Ollama model | Shows a Streamlit warning and returns local fallback output. |
| Ollama model listing failure | Shows a warning and allows manual model entry. |
| Ollama generation URL error, timeout, or JSON parse error | Shows a Streamlit warning with details and returns local fallback output. |
| Empty historical DataFrame | Forecast function returns an empty DataFrame with `Month` and `Forecast Revenue` columns. |

## Function Map

| Function | Responsibility |
| --- | --- |
| `extract_openai_usage(response)` | Normalize token usage from LangChain/OpenAI response metadata. |
| `extract_ollama_usage(payload)` | Normalize token usage from Ollama generation response payload. |
| `render_token_usage(usage)` | Display input, output, and total token metrics. |
| `get_revenue_data(...)` | Build deterministic synthetic historical revenue data. |
| `calculate_local_forecast(...)` | Build local baseline forecast data. |
| `build_prompt(...)` | Convert history, forecast, and context into the LLM prompt. |
| `local_forecast_summary(...)` | Produce deterministic markdown output without an LLM. |
| `build_combined_chart(...)` | Merge actual and forecast data for charting. |
| `is_reasoning_model(model)` | Detect model names that likely support thinking controls. |
| `render_ollama_thinking_control(model)` | Show and store the Ollama thinking-mode control. |
| `get_ollama_models(base_url)` | Fetch available Ollama model names from `/api/tags`; cached for 15 seconds. |
| `openai_forecast(prompt, model)` | Invoke OpenAI through `ChatOpenAI`. |
| `ollama_forecast(prompt, base_url, model)` | Invoke Ollama through the HTTP generate API. |
| `generate_forecast_summary(...)` | Route summary generation to the selected provider. |
| `render_provider_controls()` | Render provider-specific sidebar controls. |
| `main()` | Configure page, collect inputs, render metrics, chart, table, summary, and token usage. |

## Notes

- Core numeric forecast values are calculated locally before any LLM call.
- The LLM is used for interpretation, executive narrative, and recommendations.
- The generated history is deterministic for repeatable demos.
- The app always has a usable local fallback path.
- The final caption states that forecasts are planning estimates for educational use, not financial guarantees.
