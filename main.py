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
You are a professional resume anonymizer. Your task is to remove ALL personally identifiable information (PII) from the following resume while preserving ALL non-PII content exactly as written.

---

🔒 CRITICAL RULES

1. REMOVE ONLY the following PII elements:
- Email addresses (all formats)
- Phone numbers (all formats and countries)
- Physical addresses (street, city, state, zip/postal codes)
- Website URLs and domains (LinkedIn, GitHub, personal sites, etc.)
- Social media handles (Twitter, Instagram, Stack Overflow, etc.)
- Government-issued IDs (SSN, Aadhaar, PAN, Passport numbers)
- Credential or certification IDs (e.g. Coursera, Google IDs)
- Contact references (e.g. “Raj Abhyanker – +1 (408) 398-3126”)
- Specific location mentions (city/town/neighborhood) 

2. PRESERVE EXACTLY (if present):
- All job titles, company names, and role designations
- All university, college, and school names — across any level of education
- All degrees, majors, minors, specializations, academic honors (e.g. GPA, "Summa Cum Laude")
- All skills, technologies, languages, frameworks, tools, platforms, libraries
- All project titles, descriptions, outcomes, and features
- All quantitative metrics (e.g. "reduced costs by 30%", "led a 12-member team")
- All line breaks, bullet points, and original resume formatting
- All certifications, course names, and issuing organizations
- All dates, durations, and timeframes (e.g. "Jun 2022 – Present", "3 months")
- All language proficiencies (e.g. "Fluent", "Native Proficiency")
- All awards, honors, recognitions, hackathon wins, etc.
- All conferences, publications, speaking engagements, talks
- All volunteer positions, extracurricular leadership, committee involvement
- All organizations, clubs, student teams, initiatives, and affiliations
- All startups, products, and project brand names (e.g. "Digital Beti", "NearCast")
- All domains of focus (e.g. "BioTech", "AI Safety", "Human-Computer Interaction")
- All architectures, workflows, and methodologies (e.g. "Agile", "RAG", "CI/CD")
- All tools used in process (e.g. Figma, Notion, Jira, Tableau, Looker Studio)
- Any non-identifying dataset, repo, or tool links (e.g. "UCI HAR dataset", "OpenAI Cookbook")

3. CORRECT and NORMALIZE text:
- Fix any spacing, word breaks, or tokenization issues from PDF/OCR extraction  
  e.g. "c ustomer" → "customer", "T oyota" → "Toyota", "E xcel" → "Excel"
- Fix obvious typos or broken lines ONLY if clearly wrong  
  Do not rephrase or paraphrase any content
- Normalize common resume content artifacts (e.g. "Josh Skill, bsc YIT" → properly parsed text)

---

🔧 OUTPUT FORMAT — STRICT JSON

Return a single, valid JSON object with the following fields:

Top-Level Keys:
{
  "candidateName": ""
  "sections": [ ... ],
  "summary": [],
  "experience": [],
  "education": [],
  "skills": [],
  "projects": [],
  "certifications": [],
  "awards": [],
  "publications": [],
  "languages": [],
  "volunteer": [],
  "piiRemoved": 0
}
- candidateName: The original candidate’s full name (exactly as written) before anonymization.
  * If no name is found, return an empty string.
  * Do NOT anonymize this field — extract it *before* redaction.
- sections: List of parsed resume sections in original order
- piiRemoved: Integer count of PII elements that were redacted
- Legacy fields (summary, experience, etc.) should mirror data from sections where applicable for compatibility

EACH SECTION OBJECT:
Each object inside sections must follow this structure:
{
  "title": "Original Section Header (as in resume)",
  "type": "experience|education|projects|skills|languages|certifications|awards|volunteer|paragraphs|bullets|entries",
  
  // Content shape by type:
  // paragraphs:       { "paragraphs": ["..."] }
  // bullets:          { "bullets": ["..."] }
  // experience:       { "items": [{ "title": "...", "company": "...", "duration": "...", "location": "Metropolitan Area", "responsibilities": ["..."] }] }
  // education:        { "items": [{ "degree": "...", "institution": "...", "duration": "...", "location": "Metropolitan Area", "details": "..." }] }
  // projects:         { "items": [{ "name": "...", "description": "...", "technologies": ["..."] }] }
  // skills:           { "technical": ["..."], "other": ["..."] }
  // languages:        { "items": [{ "name": "English", "proficiency": "Fluent" }] }
  // certifications:   { "items": [{ "name": "...", "duration": "...", "issuing_organization": "...", "credential_id": "" }] }
  // awards:           { "items": [{ "name": "...", "duration": "...", "description": "..." }] }
  // volunteer/entries:{ "items": [{ "role": "...", "organization": "...", "duration": "...", "location": "Metropolitan Area", "description": "..." }] }
}

---

⚠️ IMPORTANT NOTES

- Preserve the original section ordering
- If a section exists with no content (empty), omit it from the final output
- Use empty arrays or strings if needed, but never null or undefined
- Do NOT return: [object Object], "undefined", or broken JSON
- NEVER include: Markdown, code fences, or commentary — just pure JSON
- Do not add anything from your side but make SURE EVERYTHING from the resume is included except PII
- Do NOT hallucinate new sections, skills, or roles — only include what’s actually in the input

---

📝 Input Placeholder
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

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Failed to parse Gemini response as JSON")

        # Strictly require `sections` and `piiRemoved`
        if "sections" not in parsed or "piiRemoved" not in parsed:
            raise HTTPException(status_code=500, detail="Missing required fields in Gemini response")

        candidate_name = parsed.get("candidateName", "").strip()
        print(candidate_name)
        return JSONResponse({
            "candidateName": candidate_name,
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
