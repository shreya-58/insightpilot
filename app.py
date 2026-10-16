import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

from pipeline.profiler import load_csv, profile_dataframe, generate_profile_summary
from pipeline.task_detector import detect_task_type
from pipeline.advisor import get_model_recommendation
from pipeline.trainer import train_and_evaluate
from pipeline.reporter import generate_insight_report, report_to_pdf
from llm.prune_client import PruneClient


def run_pipeline(df: pd.DataFrame, target_col: str):
    client = PruneClient()
    progress = st.progress(0, text="Starting...")

    # Stage 1: Profiling
    progress.progress(10, text="Profiling dataset...")
    profile = profile_dataframe(df)

    progress.progress(25, text="Generating data quality summary (LLM)...")
    summary = generate_profile_summary(profile, client, model=os.getenv("PROFILER_MODEL"))

    with st.expander("Data Quality Summary", expanded=True):
        st.write(summary)

    # Stage 2: Task detection
    progress.progress(35, text="Detecting task type...")
    task_type = detect_task_type(df, target_col)
    st.info(f"Detected task type: **{task_type}**")

    # Stage 3: Advisor
    progress.progress(50, text="Getting model recommendations (LLM)...")
    advisor_response = get_model_recommendation(
        task_type, profile["n_rows"], profile["n_columns"], summary,
        client, model=os.getenv("ADVISOR_MODEL"),
    )
    with st.expander("Model Advisor Notes"):
        for m in advisor_response.models:
            st.write(f"**{m.name}**: {m.reasoning}")
        st.caption(advisor_response.preprocessing_notes)

    # Stage 4: Training
    progress.progress(70, text="Training models...")
    results = train_and_evaluate(df, target_col, task_type)

    st.subheader("Model Comparison")
    results_df = pd.DataFrame({
        name: data["metrics"] for name, data in results.items()
    }).T
    st.dataframe(results_df)

    # Stage 5: Reporter
    progress.progress(90, text="Writing final report (LLM)...")
    report = generate_insight_report(
        task_type, summary, results,
        advisor_notes=advisor_response.preprocessing_notes,
        client=client, model=os.getenv("REPORTER_MODEL"),
    )

    progress.progress(100, text="Done!")

    st.subheader("Insight Report")
    st.markdown(report)

    # --- Download buttons ---
    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "Download Report (Markdown)",
            data=report,
            file_name="insightpilot_report.md",
            mime="text/markdown",
        )

    with col2:
        try:
            pdf_path = report_to_pdf(report)
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "Download Report (PDF)",
                    data=f,
                    file_name="insightpilot_report.pdf",
                    mime="application/pdf",
                )
        except Exception as e:
            st.caption(f"PDF export unavailable: {e}")


# --- Page layout / entry point ---

st.set_page_config(page_title="InsightPilot", layout="wide")
st.title("InsightPilot")
st.caption("AI-Augmented EDA & Model Advisory Tool")

with st.sidebar:
    st.subheader("Active Models")
    st.text(f"Profiler: {os.getenv('PROFILER_MODEL')}")
    st.text(f"Advisor: {os.getenv('ADVISOR_MODEL')}")
    st.text(f"Reporter: {os.getenv('REPORTER_MODEL')}")

uploaded_file = st.file_uploader("Upload a CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    st.dataframe(df.head())

    target_col = st.selectbox("Select the target column", df.columns)

    if st.button("Run InsightPilot"):
        try:
            run_pipeline(df, target_col)
        except Exception as e:
            st.error(f"Something went wrong: {e}")