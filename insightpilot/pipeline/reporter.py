import json
from pathlib import Path
import markdown as md_lib
from weasyprint import HTML

PROMPT_PATH = Path("llm/prompts/reporter_prompt.txt")


def generate_insight_report(task_type: str, profile_summary: str, model_results: dict,
                              advisor_notes: str, client, model: str) -> str:
    prompt_template = PROMPT_PATH.read_text()
    user_prompt = prompt_template.format(
        task_type=task_type,
        profile_summary=profile_summary,
        model_results=json.dumps(model_results, default=str),
        advisor_notes=advisor_notes,
    )

    report = client.call(
        model=model,
        system_prompt="You are a clear, concise data science communicator.",
        user_prompt=user_prompt,
        max_tokens=800,
        stage="reporter",
    )
    return report



def report_to_pdf(report_markdown: str, output_path: str = "report.pdf"):
    html_content = md_lib.markdown(report_markdown)
    styled_html = f"""
    <html><head><style>
    body {{ font-family: sans-serif; margin: 40px; }}
    h1, h2 {{ color: #1a1a2e; }}
    </style></head><body>{html_content}</body></html>
    """
    HTML(string=styled_html).write_pdf(output_path)
    return output_path

