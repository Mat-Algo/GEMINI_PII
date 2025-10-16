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


Notes:
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
- Make VERY SURE in the output we have ALL the content of the resume excelt PII data
- Never output thigns liek [object], etc

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

    if len(resume_text.strip()) < 50:
        raise HTTPException(status_code=400, detail="Resume text is too short")

    prompt = PROMPT_TEMPLATE.replace("{RESUME_TEXT}", resume_text)

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )

        generated_text = getattr(response, "text", None)
        if not generated_text:
            raise RuntimeError("No response text from Gemini")

        cleaned = _strip_code_fences(generated_text)
        print(cleaned)  # Optional: for debugging in logs

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Failed to parse Gemini response as JSON")

        # Strictly require `sections` and `piiRemoved`
        if "sections" not in parsed or "piiRemoved" not in parsed:
            raise HTTPException(status_code=500, detail="Missing required fields in Gemini response")

        return JSONResponse({
            "sections": parsed["sections"],
            "piiRemoved": parsed["piiRemoved"]
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to anonymize resume: {str(e)}")


# Local dev entrypoint (Render uses startCommand)
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
