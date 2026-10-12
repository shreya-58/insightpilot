import json
from pathlib import Path

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