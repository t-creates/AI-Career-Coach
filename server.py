import os
from dotenv import load_dotenv

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

import worker  # all LLM logic is in worker.py

load_dotenv()

app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return render_template("index.html")


# ---------- PAGE ROUTES (HTML only, GET) ----------

@app.route("/career-advisor", methods=["GET"])
def career_advisor_page():
    return render_template("career_advisor.html")


@app.route("/cover-letter", methods=["GET"])
def cover_letter_page():
    return render_template("cover_letter.html")


@app.route("/resume-polisher", methods=["GET"])
def resume_polisher_page():
    return render_template("resume_polisher.html")


@app.route("/chat", methods=["GET"])
def chat_page():
    return render_template("chat.html")


# ---------- API ROUTES (JSON only, POST) ----------

@app.route("/api/career-advice", methods=["POST"])
def api_career_advice():
    data = request.get_json(force=True, silent=True) or {}
    position = data.get("position_applied", "")
    job_desc = data.get("job_description", "")
    resume = data.get("resume_content", "")

    advice = worker.generate_career_advice(
        position_applied=position,
        job_description=job_desc,
        resume_content=resume,
    )
    return jsonify({"advice": advice})


@app.route("/api/cover-letter", methods=["POST"])
def api_cover_letter():
    data = request.get_json(force=True, silent=True) or {}
    position = data.get("position_applied", "")
    job_desc = data.get("job_description", "")
    resume = data.get("resume_content", "")
    company = data.get("company_name", "")

    letter = worker.generate_cover_letter(
        position_applied=position,
        job_description=job_desc,
        resume_content=resume,
        company_name=company,
    )
    return jsonify({"cover_letter": letter})


@app.route("/api/resume-polisher", methods=["POST"])
def api_resume_polisher():
    data = request.get_json(force=True, silent=True) or {}
    resume = data.get("resume_content", "")
    job_desc = data.get("job_description", "")

    polished_resume = worker.generate_polished_resume(
        resume_content=resume,
        job_description=job_desc,
    )
    return jsonify({"polished_resume": polished_resume})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True, silent=True) or {}
    message = data.get("message", "")

    if not message:
        return jsonify({"reply": ""})

    reply = worker.chat_with_llm(message)
    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True)
