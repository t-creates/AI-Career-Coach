import os
from flask import Flask, render_template, request
from ibm_watsonx_ai import Credentials, APIClient
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

app = Flask(__name__)

# ---------- LLM SETUP (shared) ----------

MODEL_ID = "meta-llama/llama-3-2-11b-vision-instruct"  # or whatever is supported in your account
WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "skills-network")
APIKEY = os.getenv("WATSONX_APIKEY")

if not APIKEY:
    raise RuntimeError("WATSONX_APIKEY env var not set")

credentials = Credentials(
    url=WATSONX_URL,
    api_key=APIKEY,
)

client = APIClient(credentials)

params = {
    GenTextParamsMetaNames.MAX_NEW_TOKENS: 521,
    GenTextParamsMetaNames.TEMPERATURE: 0.1,
}

llm = ModelInference(
    model_id=MODEL_ID,
    api_client=client,
    project_id=PROJECT_ID,
    params=params,
)

def generate_text(prompt: str) -> str:
    """Shared helper to call the LLM once."""
    resp = llm.generate_text(prompt=prompt)
    # Adjust indexing if needed based on your actual response structure
    return resp["results"][0]["generated_text"].strip()
