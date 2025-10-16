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

---

**CRITICAL RULES**

1. REMOVE ONLY the following PII elements:
   - Full name (first name, middle name, last name)
   - Email addresses (all formats)
   - Phone numbers (all formats and countries)
   - Physical addresses (street, city, state, zip/postal codes)
   - Website URLs and domains (LinkedIn, GitHub, personal sites, etc.)
   - Social media handles (Twitter, Instagram, Stack Overflow, etc.)
   - Government IDs (SSN, Aadhaar, PAN, Passport numbers)
   - Credential or certification IDs
   - Contact names and phone numbers (e.g. “Raj Abhyanker – +1 (408) 398-3126”)
   - Any specific city or neighborhood name (replace with "Metropolitan Area")

2. PRESERVE EXACTLY:
   - All job titles, company names, and roles
   - All university and school names
   - All degree names and majors
   - All skills, technologies, frameworks, tools
   - All project descriptions and achievements
   - All metrics, statistics, percentages
   - All bullet points and line formatting
   - All certifications and course names
   - All dates and durations

3. CORRECT and NORMALIZE:
   - Fix any split words or spacing issues from OCR/text extraction
     e.g., "c ustomer" → "customer", "T oyota" → "Toyota", "Josh Skill, bsc YIT" → proper tokens
   - Fix common typos or broken lines where possible
   - Do not rephrase meaningful content

---

**OUTPUT FORMAT (STRICT JSON)**

Return **ONLY valid JSON** containing:
- `sections`: an array that preserves **original section titles and order** from the resume
- `summary`, `experience`, `education`, `skills`, `projects`, `certifications`, `awards`, `publications`, `languages`, `volunteer` — legacy fields for compatibility
- `piiRemoved`: integer number of PII items redacted

Each section in `sections` must include:
```json
{
  "title": "Exact Section Title From Resume",
  "type": "experience|education|projects|skills|languages|certifications|awards|volunteer|paragraphs|bullets|entries",
  // Content shape based on type:
  // paragraphs: { "paragraphs": ["..."] }
  // bullets: { "bullets": ["..."] }
  // experience: { "items": [ { "title": "...", "company": "...", "duration": "...", "location": "Metropolitan Area", "responsibilities": ["..."] } ] }
  // education: { "items": [ { "degree": "...", "institution": "...", "duration": "...", "location": "Metropolitan Area", "details": "..." } ] }
  // projects: { "items": [ { "name": "...", "description": "...", "technologies": ["..."] } ] }
  // skills: { "technical": ["..."], "other": ["..."] }
  // entries: { "items": [ { "title|name|role": "...", "organization": "...", "duration": "...", "description": "...", "location": "Metropolitan Area" } ] }
}


**IMPORTANT:**
- Return ONLY JSON (no markdown, no code fences, no extra commentary)
- Preserve the resume’s original section order under "sections".
- Keep ALL original wording for non-PII content
- Use empty arrays [] or empty strings "" where applicable
- "piiRemoved" is an integer count of PII items removed
- Make sure you do not rephrase, reduce, or add extra anything but you can correct spelling mistakes if any which would have occured during text extraction.

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
