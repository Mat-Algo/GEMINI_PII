import os
import re
import json
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import JSONResponse

# New Gemini SDK
from google import genai
from google.genai import types

from dotenv import load_dotenv
load_dotenv()

# ---------------------------
# Config
# ---------------------------
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if not os.getenv("GEMINI_API_KEY"):
    # The new client reads GEMINI_API_KEY from the env automatically;
    # we check early to fail fast on Render if it's missing.
    raise RuntimeError("GEMINI_API_KEY environment variable is not set")

client = genai.Client()  # picks up GEMINI_API_KEY from environment

# ---------------------------
# App
# ---------------------------
app = FastAPI(title="Resume Anonymizer API (Python + Gemini SDK)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Request models
# ---------------------------
class AnonymizeRequest(BaseModel):
    resumeText: str

# ---------------------------
# Prompt Template
# ---------------------------
PROMPT_TEMPLATE = """
You are a professional resume anonymizer. Your task is to remove ALL personally identifiable information (PII) from the following resume while preserving ALL other content exactly as written.

**CRITICAL RULES:**
1. Remove ONLY these PII elements:
   - Person's full name (first name, middle name, last name)
   - Email addresses (all formats)
   - Phone numbers (all formats and countries)
   - Physical addresses (street, city, state, zip/postal codes)
   - URLs and domains (LinkedIn, GitHub, personal websites, portfolio links, etc.)
   - Social media handles (Twitter, Instagram, etc.)
   - Government IDs (SSN, Aadhaar, PAN, Passport numbers)
   - Any credential IDs or license numbers
   - Geographic locations that are personally identifying (replace city names with just "Major City" or "Metropolitan Area")

2. PRESERVE EXACTLY (do not modify or remove):
   - All job titles and roles
   - All company names (keep them as-is)
   - All university/college names (keep them as-is)
   - All degree names and majors
   - All skills, technologies, and tools
   - All project descriptions and achievements
   - All dates and durations
   - All metrics, numbers, and percentages (except PII numbers)
   - All certifications and course names
   - All professional language exactly as written
   - All bullet points and descriptions word-for-word

3. Return response as valid JSON only with this exact structure:
{
  "summary": "professional summary text here (or empty string if not present)",
  "experience": [
    {
      "title": "Job Title",
      "company": "Company Name",
      "duration": "Jan 2020 - Present",
      "location": "Metropolitan Area",
      "responsibilities": [
        "First responsibility exactly as written",
        "Second responsibility exactly as written"
      ]
    }
  ],
  "education": [
    {
      "degree": "Degree Name",
      "institution": "University Name",
      "duration": "2016 - 2020",
      "location": "Metropolitan Area",
      "details": "Any additional details exactly as written"
    }
  ],
  "skills": {
    "technical": ["skill1", "skill2", "skill3"],
    "other": ["skill1", "skill2"]
  },
  "projects": [
    {
      "name": "Project Name",
      "description": "Full description exactly as written",
      "technologies": ["tech1", "tech2"]
    }
  ],
  "certifications": [
    {
      "name": "Certification Name",
      "issuer": "Issuing Organization",
      "date": "Month Year"
    }
  ],
  "awards": ["Award 1 exactly as written", "Award 2 exactly as written"],
  "publications": ["Publication 1 exactly as written"],
  "languages": ["Language 1: Proficiency", "Language 2: Proficiency"],
  "volunteer": [
    {
      "role": "Volunteer Role",
      "organization": "Organization Name",
      "duration": "dates",
      "description": "Description exactly as written"
    }
  ],
  "piiRemoved": 15
}

**IMPORTANT:** 
- Return ONLY valid JSON, no markdown formatting, no code blocks, no extra text
- Keep ALL original wording and phrasing for non-PII content
- Do not add, embellish, or modify any professional content
- Empty sections should be empty arrays [] or empty strings ""
- piiRemoved should be an integer count of PII items removed

Here is the resume text to anonymize:

{RESUME_TEXT}
"""

# ---------------------------
# Helpers
# ---------------------------
def _strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` fences if the model adds them."""
    t = text.strip()
    t = re.sub(r'^```(?:json)?\s*', '', t, flags=re.IGNORECASE)
    t = re.sub(r'\s*```$', '', t)
    return t.strip()

def _validate_and_fill(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure all expected keys exist with safe defaults."""
    if not isinstance(parsed, dict):
        raise ValueError("AI response is not a JSON object")

    skills = parsed.get("skills") or {}
    if not isinstance(skills, dict):
        skills = {}
    skills.setdefault("technical", [])
    skills.setdefault("other", [])

    return {
        "summary": parsed.get("summary", "") or "",
        "experience": parsed.get("experience", []) if isinstance(parsed.get("experience"), list) else [],
        "education": parsed.get("education", []) if isinstance(parsed.get("education"), list) else [],
        "skills": skills,
        "projects": parsed.get("projects", []) if isinstance(parsed.get("projects"), list) else [],
        "certifications": parsed.get("certifications", []) if isinstance(parsed.get("certifications"), list) else [],
        "awards": parsed.get("awards", []) if isinstance(parsed.get("awards"), list) else [],
        "publications": parsed.get("publications", []) if isinstance(parsed.get("publications"), list) else [],
        "languages": parsed.get("languages", []) if isinstance(parsed.get("languages"), list) else [],
        "volunteer": parsed.get("volunteer", []) if isinstance(parsed.get("volunteer"), list) else [],
        "piiRemoved": parsed.get("piiRemoved", 0) if isinstance(parsed.get("piiRemoved", 0), int) else 0,
    }

# ---------------------------
# Routes
# ---------------------------
@app.get("/health")
def health():
    return {"status": "ok", "message": "Resume Anonymizer API is running"}

@app.post("/api/anonymize")
def anonymize(payload: AnonymizeRequest):
    resume_text = payload.resumeText

    if not resume_text or not isinstance(resume_text, str):
        raise HTTPException(status_code=400, detail="Invalid request: resumeText is required")

    if len(resume_text) < 50:
        raise HTTPException(status_code=400, detail="Resume text is too short")

    prompt = PROMPT_TEMPLATE.replace("{RESUME_TEXT}", resume_text)

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                # Disable "thinking" for speed/cost parity with your Node version
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                temperature=0.1,
                # Encourage strict JSON
                response_mime_type="application/json",
            ),
        )

        generated_text = getattr(response, "text", None)
        if not generated_text:
            raise RuntimeError("No response text from Gemini")

        cleaned = _strip_code_fences(generated_text)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            sample = cleaned[:600]
            raise HTTPException(status_code=500, detail=f"Failed to parse AI response as JSON. Sample: {sample}")

        validated = _validate_and_fill(parsed)
        return JSONResponse(validated)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to anonymize resume: {str(e)}")

# Local dev entrypoint (Render uses startCommand)
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
