# InsightPilot

### AI-Augmented EDA & Model Advisory Tool

InsightPilot is an AI-powered exploratory data analysis (EDA) and machine learning advisory tool that transforms raw CSV data into actionable insights and baseline ML models.

Upload a CSV file, select a target column, and InsightPilot profiles data quality, recommends suitable machine learning models, trains and evaluates them, and generates a stakeholder-ready insight report.

It combines classical machine learning tools such as **pandas, scikit-learn, and XGBoost** with LLM-driven reasoning. Every LLM request is routed through [Prune](https://withprune.com), an AI gateway that provides centralized request logging, cost tracking, caching, and security controls.

**Live Demo:** 

---

## Key Features

- **Automated Data Profiling:** Analyzes missing values, outliers, skewness, cardinality, and other data quality indicators.
- **Intelligent Task Detection:** Automatically identifies whether the problem is classification or regression.
- **AI-Powered Model Recommendations:** Uses an LLM to recommend 2–3 suitable candidate models with structured, schema-validated JSON output.
- **Automated Model Training:** Trains and evaluates recommended models using scikit-learn and XGBoost.
- **Insight Generation:** Produces a stakeholder-ready Markdown report with an executive summary, model comparison, and actionable recommendations.
- **Feature Exclusion:** Allows users to exclude free-text, identifier, or potentially leaking columns before training.
- **Centralized LLM Integration:** Routes all LLM calls through Prune for consistent logging, caching, cost tracking, and security controls.
- **Report Export:** Supports Markdown export, with PDF export available when the required dependencies are installed.

### Why Feature Exclusion Matters

During testing on a legal case dataset, a column named `bail_outcome_label_detailed` was found to restate the target outcome, artificially inflating model accuracy.

InsightPilot includes a feature-exclusion control to help prevent this kind of target leakage by allowing users to remove suspicious columns before training.

---

## How It Works

InsightPilot follows a multi-stage pipeline that combines deterministic data processing with LLM-assisted reasoning.

1. **CSV Upload:** Upload the dataset and select the target column.
2. **Feature Review:** Review the available columns and exclude irrelevant or potentially leaky features.
3. **Data Profiler:** Computes data quality statistics using pandas and generates a plain-English summary using an LLM.
4. **Task Detector:** Determines whether the target represents a classification or regression problem.
5. **Model Advisor:** Recommends suitable ML models using an LLM and validates the structured output.
6. **Model Trainer:** Trains and evaluates the recommended models using scikit-learn and XGBoost.
7. **Insight Reporter:** Generates a comprehensive Markdown report summarizing data quality, model performance, and recommendations.
8. **Results Dashboard:** Displays pipeline progress, model comparisons, generated insights, and downloadable reports through Streamlit.

---

## Architecture

```text
                    ┌──────────────────┐
                    │   CSV Upload     │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ Target Selection │
                    │ & Feature Review │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Data Profiler   │
                    │ pandas + LLM     │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Task Detector   │
                    │ Classification / │
                    │    Regression    │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Model Advisor   │
                    │ LLM + Validation │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Model Trainer   │
                    │ sklearn / XGBoost│
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ Insight Reporter │
                    │    LLM Report    │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Streamlit UI    │
                    │ Results & Export │
                    └──────────────────┘

       All LLM requests are routed through Prune
       for centralized logging and gateway controls.
```

### LLM Request Flow

Every LLM call passes through a shared `PruneClient` wrapper rather than calling providers directly.

```text
InsightPilot Pipeline
        │
        ▼
   PruneClient
        │
        ▼
   Prune Gateway
        │
        ├── OpenAI Models
        ├── Google Models
        └── NVIDIA Models
```

This design centralizes request handling and records request/response information in `logs/prune_calls.jsonl` to support debugging and cost reconciliation. It also provides a single place to manage provider-specific behavior and gateway-related compatibility issues.

---

## Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Data Analysis | pandas, NumPy |
| Machine Learning | scikit-learn, XGBoost |
| LLM Integration | OpenAI SDK, Prune AI Gateway |
| Structured Output Validation | Pydantic |
| User Interface | Streamlit |
| Testing | pytest |
| Deployment | Hugging Face Spaces |
| Reporting | Markdown, optional PDF export |

---

## Getting Started

Follow these steps to run InsightPilot locally.

### Prerequisites

- Python 3.10 or a compatible version supported by the dependencies
- Git
- A Prune API key
- Access to the LLM models configured for the pipeline

### 1. Clone the Repository

```bash
git clone https://github.com/shreya-58/insightpilot/
cd insightpilot
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the example environment file:

**Windows:**

```bash
copy .env.example .env
```

**macOS / Linux:**

```bash
cp .env.example .env
```

Update `.env` with your Prune credentials, gateway configuration, and model names for each pipeline stage.

Example configuration:

```env
PRUNE_API_KEY=your_prune_api_key
PRUNE_BASE_URL=your_prune_gateway_base_url

# Configure the model names required by each pipeline stage.
# Use the exact variable names expected by your application.
```

**Important:** Never commit your `.env` file or expose API keys in your repository.

### 5. Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to access InsightPilot.

---

## Project Structure

```text
insightpilot/
├── app.py                    # Streamlit application entry point
├── pipeline/
│   ├── profiler.py            # Data quality statistics and LLM summary
│   ├── task_detector.py       # Classification vs. regression detection
│   ├── advisor.py             # LLM model recommendations and validation
│   ├── trainer.py             # scikit-learn and XGBoost training
│   └── reporter.py            # Insight report generation and PDF export
├── llm/
│   ├── prune_client.py        # Shared Prune gateway client and logging
│   └── prompts/               # LLM prompt templates for each pipeline stage
├── sample_data/               # Sample datasets for testing
├── tests/                     # pytest suite with mocked LLM calls
├── requirements.txt           # Python dependencies
├── .env.example               # Example environment configuration
└── README.md                  # Project documentation
```

---

## Security, Reliability & Observability

InsightPilot uses Prune as a centralized gateway for LLM requests.

Key integration capabilities include:

- **Centralized Request Logging:** Records LLM interactions in `logs/prune_calls.jsonl` for debugging and cost reconciliation.
- **Caching Support:** Leverages gateway caching where applicable to reduce redundant model requests.
- **Security Controls:** Integrates with gateway-level security features, including sensitive-data filtering.
- **Structured Output Validation:** Uses Pydantic to validate the Model Advisor's structured recommendations.
- **Fallback Handling:** Retries invalid model recommendations and falls back to default model choices when necessary.

---

## Known Limitations

### 1. Reasoning-Model Non-Termination

Some free-tier reasoning models, including NVIDIA Nemotron configurations tested through Prune, did not reliably terminate their internal reasoning phase within the configured token budget. In these cases, the reasoning-token count could reach the maximum without producing a final answer.

A client-side fallback attempts to extract usable text from the reasoning trace, but the resulting output may be lower quality.

Model behavior may vary with gateway routing, configuration, and future provider updates.

### 2. PII False Positives

Prune's Shield security filter was observed flagging some ordinary numeric values as potential credit-card patterns, including matches within longer numbers.

Rounding floating-point values before sending them reduced some false positives during testing. However, this is a mitigation rather than a complete solution.

### 3. Model Advisor Reliability

The quality of model recommendations depends on the selected LLM's ability to follow the expected JSON schema.

If validation fails, the application retries once and then falls back to predefined model recommendations rather than terminating the entire pipeline.

### 4. Automated Leakage Detection

Feature leakage prevention currently relies on manual column exclusion. InsightPilot does not yet automatically detect every form of target leakage.

---

## Future Improvements

Potential enhancements include:

- [ ] **Smarter Preprocessing:** Use one-hot encoding for low-cardinality categorical variables instead of relying solely on label encoding.
- [ ] **Iterative Model Refinement:** Introduce a feedback and critic loop to improve model selection and performance.
- [ ] **Multi-Provider A/B Testing:** Compare models across providers through Prune using consistent evaluation criteria.
- [ ] **Automated Leakage Detection:** Identify suspicious features that may reveal target information.
- [ ] **Enhanced Model Evaluation:** Expand evaluation metrics and preprocessing strategies for different dataset types.
- [ ] **Improved Reporting:** Add richer visualizations and more detailed explanations of model behavior.

---

## About the Prune Integration

InsightPilot also serves as a practical test harness for [Prune](https://withprune.com), helping evaluate how an AI gateway behaves within a real application pipeline.

During development and testing, the integration surfaced several useful findings:

- **Token Configuration:** A `max_tokens` parameter was required for the tested gateway configuration, differing from some upstream provider defaults.
- **Base URL Configuration:** The tested setup required a `/proxy`-suffixed base URL.
- **Model ID Compatibility:** A curated set of verified model IDs was useful as a fallback when models from a broader catalog were unavailable.
- **Security Filter Behavior:** Shield's PII filtering produced false positives on some ordinary numeric data.
- **Reasoning-Model Behavior:** Some models did not reliably return a final answer within the configured token budget.

These observations helped identify practical integration challenges involving gateway compatibility, model behavior, structured output reliability, and sensitive-data filtering.

---

