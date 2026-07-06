import json
from openai import OpenAI
from utils.input_output import record_to_json
import os
import argparse
parser = argparse.ArgumentParser(description="Cal embedding Runner")
parser.add_argument(
    "--input_file",
    type=str, 
    help="Input file"
)
parser.add_argument(
    "--record_save_file",
    type=str, 
    help="Output file"
)
args = parser.parse_args()

input_path = args.input_file
output_path = args.record_save_file

client = OpenAI()


# Select the embedding model
EMBED_MODEL = "text-embedding-3-small"

def get_embedding(text, model=EMBED_MODEL):
    if not text:
        return []
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

with open(input_path, "r") as f:
    data = json.load(f)

for item in data:
    safe_code = item.get("safe_code", "")
    unsafe_code = item.get("unsafe_code", "")
    llm_code = item.get("llm_code", "")

    item["safe_embedding"] = get_embedding(safe_code)
    item["unsafe_embedding"] = get_embedding(unsafe_code)
    item["llm_code_embedding"] = get_embedding(llm_code)

    record_to_json(
        filename=output_path,
        **item
    )