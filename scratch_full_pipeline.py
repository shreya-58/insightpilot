import os
from dotenv import load_dotenv
load_dotenv()

from pipeline.profiler import load_csv, profile_dataframe, generate_profile_summary
from pipeline.task_detector import detect_task_type
from pipeline.advisor import get_model_recommendation
from pipeline.trainer import train_and_evaluate
from pipeline.reporter import generate_insight_report
from llm.prune_client import PruneClient

client = PruneClient()
df = load_csv("sample_data/iris.csv")
target_col = "target"

profile = profile_dataframe(df)
task_type = detect_task_type(df, target_col)
print("Task type:", task_type)

summary = generate_profile_summary(profile, client, model=os.getenv("PROFILER_MODEL"))
print("\n--- Profile Summary ---\n", summary)

advisor_response = get_model_recommendation(
    task_type, profile["n_rows"], profile["n_columns"], summary,
    client, model=os.getenv("ADVISOR_MODEL"),
)
print("\n--- Advisor Recommendation ---\n", advisor_response)

results = train_and_evaluate(df, target_col, task_type)
print("\n--- Training Results ---\n", results)

report = generate_insight_report(
    task_type, summary, results,
    advisor_notes=advisor_response.preprocessing_notes,
    client=client, model=os.getenv("REPORTER_MODEL"),
)
print("\n--- Final Report ---\n", report)

with open("sample_output_report.md", "w") as f:
    f.write(report)