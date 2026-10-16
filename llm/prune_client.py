import os
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import re

load_dotenv()

LOG_PATH = Path("logs/prune_calls.jsonl")
LOG_PATH.parent.mkdir(exist_ok=True)


class PruneClient:
    def __init__(self):
        api_key = os.getenv("PRUNE_API_KEY")
        base_url = os.getenv("PRUNE_BASE_URL")
        if not api_key or not base_url:
            raise ValueError("PRUNE_API_KEY and PRUNE_BASE_URL must be set in .env")

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def call(self, model: str, system_prompt: str, user_prompt: str,
              max_tokens: int = 500, stage: str = "unknown") -> str:
        """Call an LLM through Prune, logging the request/response for later reconciliation."""
        start = time.time()
        content = None
        usage = None
        error = None
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
            )
            message = response.choices[0].message
            content = message.content

            # Fallback: some reasoning models (e.g. Nemotron via this Prune routing)
            # never produce a final `content` and instead spend the whole budget on
            # hidden reasoning. If that happens, fall back to the reasoning text.
            if not content:
                reasoning = getattr(message, "reasoning", None)
                if reasoning:
                    content = self._extract_summary_from_reasoning(reasoning)

            usage = response.usage.model_dump() if response.usage else None
        except Exception as e:
            error = str(e)
            raise
        finally:
            self._log_call(stage, model, max_tokens, time.time() - start,
                            usage, error, len(user_prompt))

        return content

    @staticmethod
    def _extract_summary_from_reasoning(reasoning: str) -> str:
        """Best-effort extraction of usable text from a reasoning trace that
        never reached a final answer. Known workaround for models that don't
        terminate reasoning within this Prune routing."""
        for marker in ["Synthesize Key Points", "**Synthesize", "Keep it to"]:
            if marker in reasoning:
                reasoning = reasoning.split(marker, 1)[0]
                break
        reasoning = reasoning.replace("Here's a thinking process:", "").strip()
        return reasoning

    def _log_call(self, stage, model, max_tokens, latency, usage, error, prompt_len):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "model": model,
            "max_tokens": max_tokens,
            "latency_sec": round(latency, 3),
            "usage": usage,
            "error": error,
            "prompt_char_len": prompt_len,
        }
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")


    @staticmethod
    def _extract_summary_from_reasoning(reasoning: str) -> str:
        """Best-effort extraction of usable text from a reasoning trace that
        never reached a final answer. Known workaround for models that don't
        terminate reasoning within this Prune routing."""
        # Drop numbered meta-analysis sections (e.g. "1. **Analyze User Input:**")
        reasoning = re.sub(r"^\d+\.\s+\*\*.*?\*\*:.*?(?=^\d+\.|\Z)", "", reasoning, flags=re.MULTILINE | re.DOTALL)
        reasoning = reasoning.replace("Here's a thinking process:", "").strip()

        # If a synthesis/summary section exists, prefer text after its last heading
        for marker in ["Synthesize", "Final summary", "In summary"]:
            idx = reasoning.rfind(marker)
            if idx != -1:
                return reasoning[idx:].strip()

        # Otherwise just return the last few sentences, which tend to be closer
        # to a conclusion than the opening analysis
        sentences = re.split(r"(?<=[.!?])\s+", reasoning.strip())
        return " ".join(sentences[-4:]) if len(sentences) > 4 else reasoning