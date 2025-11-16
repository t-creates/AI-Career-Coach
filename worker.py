import os
import logging

from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials, APIClient
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

# ----------------- Logging -----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------- Env & LLM Setup -----------------

load_dotenv()  # load .env if present

WATSONX_APIKEY = os.getenv("WATSONX_APIKEY")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "skills-network")
WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

# allow override, but give a safe default that should exist in most watsonx envs
MODEL_ID = os.getenv("WATSONX_MODEL_ID", "ibm/granite-3-8b-instruct")

if not WATSONX_APIKEY:
    raise RuntimeError("WATSONX_APIKEY is not set. Put it in your .env file.")

logger.info("Initializing Watsonx LLM...")
logger.info("Model: %s", MODEL_ID)

credentials = Credentials(
    url=WATSONX_URL,
    api_key=WATSONX_APIKEY,
)

client = APIClient(credentials)

params = {
    GenTextParamsMetaNames.MAX_NEW_TOKENS: 512,
    GenTextParamsMetaNames.TEMPERATURE: 0.2,
}

llm = ModelInference(
    model_id=MODEL_ID,
    api_client=client,
    project_id=WATSONX_PROJECT_ID,
    params=params,
)

logger.info("Watsonx LLM initialized.")


def _generate_text(prompt: str) -> str:
    """
    Low-level helper to call the watsonx model.
    Handles both string and dict-style responses.
    """
    logger.debug("Sending prompt to LLM: %s", prompt[:500])
    try:
        resp = llm.generate_text(prompt=prompt)
        logger.debug("Raw LLM response type=%s: %r", type(resp), resp)

        # Case 1: watsonx returns a plain string
        if isinstance(resp, str):
            return resp.strip()

        # Case 2: watsonx returns a dict with "results"
        if isinstance(resp, dict):
            if "results" in resp and resp["results"]:
                first = resp["results"][0]
                # some models use "generated_text", others "text"
                text = first.get("generated_text") or first.get("text") or ""
                return text.strip()

            # Fallback: if it has a direct "generated_text" field
            if "generated_text" in resp:
                return str(resp["generated_text"]).strip()

        # Last fallback: stringify whatever it is
        return str(resp).strip()

    except Exception as e:
        logger.exception("Error calling LLM: %s", e)
        return "Sorry, I ran into an error while generating a response."


# ----------------- High-level Tools -----------------


def generate_career_advice(
    position_applied: str,
    job_description: str,
    resume_content: str,
) -> str:
    """
    Use the LLM to generate tailored career advice.
    Called by /api/career-advice.
    """
    prompt = f"""
You are a professional career advisor.

The user is applying for this position:
{position_applied}

Here is the job description:
{job_description}

Here is the user's resume:
{resume_content}

Task:
1. Evaluate how well the resume matches the job description.
2. Suggest specific, practical improvements to the resume.
3. Highlight key skills, phrases, and experiences that should be emphasized.
4. Use clear bullet points where appropriate.

Return your advice as readable text.
"""
    return _generate_text(prompt)


def generate_cover_letter(
    position_applied: str,
    job_description: str,
    resume_content: str,
    company_name: str = "",
) -> str:
    """
    Use the LLM to generate a customized cover letter.
    Called by /api/cover-letter.
    """
    company_part = f" for the company {company_name}" if company_name else ""

    prompt = f"""
You are an expert cover letter writer.

Write a professional, concise cover letter{company_part}
for the position: {position_applied}

Job description:
{job_description}

Candidate resume:
{resume_content}

Requirements:
- Use a polite, professional tone.
- Make the candidate sound like a strong fit for the role.
- Do not invent fake experience; only rephrase and highlight what is present.
- Include a brief opening, 1–2 body paragraphs, and a closing.

Return only the cover letter text, ready to be copied into an application.
"""
    return _generate_text(prompt)


def generate_polished_resume(
    resume_content: str,
    job_description: str,
) -> str:
    """
    Use the LLM to polish the resume to better match the target job.
    Called by /api/resume-polisher.
    """
    prompt = f"""
You are an expert resume editor.

Here is the job description:
{job_description}

Here is the current resume:
{resume_content}

Task:
- Rewrite or reorganize the resume content so that it is a strong match for the job description.
- Keep all facts truthful; do not fabricate experience.
- Emphasize relevant skills, technologies, and achievements.
- Use clear section headings and bullet points.
- Keep it in English.

Return the improved resume text only.
"""
    return _generate_text(prompt)


def chat_with_llm(message: str) -> str:
    """
    Generic chat endpoint.
    Called by /api/chat.
    """
    prompt = f"""
You are a helpful assistant focused on careers, jobs, and professional development.

Instructions:
- Answer the user's question in a single, clear response.
- Do NOT repeat the user's question.
- Do NOT include the phrases "User message" or "Assistant response".
- Do NOT invent or include additional Q&A examples.
- Just return your answer text.

User question:
{message}

Assistant answer:
"""
    return _generate_text(prompt)

