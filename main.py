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
   - All bullet points and descriptions word-for-word

3. OUTPUT FORMAT (STRICT JSON):
Return ONLY valid JSON with:
- A dynamic sections array that mirrors the original resume’s section titles and order exactly as they appear (preserve casing and punctuation). Each element has:
  {
    "title": "Section title exactly from resume",
    "type": "experience|education|projects|skills|languages|certifications|awards|volunteer|paragraphs|bullets|entries",
    // Content shape depends on type:
    // paragraphs: { "paragraphs": [ "para1", "para2", ... ] }
    // bullets:    { "bullets": [ "bullet1", "bullet2", ... ] }
    // experience: { "items": [ { "title": "...", "company": "...", "duration": "...", "location": "Metropolitan Area", "responsibilities": ["...", "..."] } ] }
    // education:  { "items": [ { "degree": "...", "institution": "...", "duration": "...", "location": "Metropolitan Area", "details": "..." } ] }
    // projects:   { "items": [ { "name": "...", "description": "...", "technologies": ["...","..."] } ] }
    // skills:     { "technical": ["..."], "other": ["..."] }
    // languages:  { "bullets": ["Language: Proficiency", "..."] }
    // certifications/awards/volunteer/publications may use "bullets" or "items" with sensible fields:
    // entries:    { "items": [ { "title"|"name"|"role": "...", "organization"|"institution": "...", "duration": "...", "location": "Metropolitan Area", "description": "...", "technologies": ["..."] } ] }
  }

- AND include the legacy fields for backward compatibility:
  "summary", "experience", "education", "skills", "projects", "certifications", "awards", "publications", "languages", "volunteer", "piiRemoved"

Notes:
- It is OK if the dynamic sections duplicate content that also appears in the legacy fields.
- If a section in the resume does not fit any known type, set "type": "bullets" with its bullet list, or "type": "paragraphs" with text blocks.
- Preserve the resume’s original section order under "sections".

**IMPORTANT:**
- Return ONLY JSON (no markdown, no code fences, no extra commentary)
- Keep ALL original wording for non-PII content
- Use empty arrays [] or empty strings "" where applicable
- "piiRemoved" is an integer count of PII items removed
- Make sure you do not rephrase, reduce, or add extra anything but you can correct spelling mistakes if any which would have occured during text extraction.
- Make sure to fix errors like :
  c ustomer -> customer, E EG-> EEG, T oyota -> Toyota

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
