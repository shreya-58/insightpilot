import os
import json
from dotenv import load_dotenv
load_dotenv()

from pipeline.profiler import load_csv, profile_dataframe
from llm.prune_client import PruneClient

client = PruneClient()
df = load_csv("sample_data/iris.csv")
profile = profile_dataframe(df)

prompt_template = open("llm/prompts/profiler_prompt.txt").read()
user_prompt = prompt_template.format(profile_json=json.dumps(profile, default=str))

raw_response = client.client.chat.completions.create(
    model=os.getenv("PROFILER_MODEL"),
    messages=[
        {"role": "system", "content": "You are a careful, concise data quality analyst. detailed thinking off"},
        {"role": "user", "content": user_prompt},
    ],
    max_tokens=3000,
    extra_body={"reasoning": {"exclude": True}},  # belt-and-suspenders, in case system-prompt toggle alone isn't enough
)

print("finish_reason:", raw_response.choices[0].finish_reason)
print("content:", repr(raw_response.choices[0].message.content))
print("full message object:", raw_response.choices[0].message.model_dump())
print("usage:", raw_response.usage)