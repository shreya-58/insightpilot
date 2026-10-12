from pydantic import BaseModel, ValidationError
from typing import List
import json


class ModelRecommendation(BaseModel):
    name: str
    reasoning: str


class AdvisorResponse(BaseModel):
    models: List[ModelRecommendation]
    preprocessing_notes: str
    
    
from pathlib import Path

PROMPT_PATH = Path("llm/prompts/advisor_prompt.txt")


def get_model_recommendation(task_type: str, n_rows: int, n_columns: int,
                               profile_summary: str, client, model: str) -> AdvisorResponse:
    prompt_template = PROMPT_PATH.read_text()
    user_prompt = prompt_template.format(
        task_type=task_type, n_rows=n_rows, n_columns=n_columns,
        profile_summary=profile_summary,
    )

    raw = client.call(
        model=model,
        system_prompt="You are a precise ML advisor. Output only valid JSON, nothing else.",
        user_prompt=user_prompt,
        max_tokens=400,
        stage="advisor",
    )

    try:
        return _parse_and_validate(raw)
    except (json.JSONDecodeError, ValidationError) as e:
        # One retry, telling the model what went wrong
        retry_prompt = user_prompt + f"\n\nYour previous response was invalid JSON: {e}. Return ONLY valid JSON this time."
        raw_retry = client.call(
            model=model,
            system_prompt="You are a precise ML advisor. Output only valid JSON, nothing else.",
            user_prompt=retry_prompt,
            max_tokens=400,
            stage="advisor_retry",
        )
        try:
            return _parse_and_validate(raw_retry)
        except (json.JSONDecodeError, ValidationError):
            # Fall back to a sensible default rather than crashing the pipeline
            return AdvisorResponse(
                models=[
                    ModelRecommendation(name="RandomForest", reasoning="Default fallback: robust baseline."),
                    ModelRecommendation(name="LogisticRegression" if task_type == "classification" else "LinearRegression",
                                         reasoning="Default fallback: simple linear baseline."),
                ],
                preprocessing_notes="Advisor LLM call failed twice; using default model set.",
            )


def _parse_and_validate(raw: str) -> AdvisorResponse:
    # Strip markdown code fences if the model wrapped its JSON in them
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    data = json.loads(cleaned)
    return AdvisorResponse(**data)