import os
import re
import json
from typing import Any, Dict
import uvicorn

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import JSONResponse

# New Gemini SDK
from google import genai
from google.genai import types
from fastapi import FastAPI, HTTPException, Header
from starlette.responses import JSONResponse

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
- Contact references (e.g. "Raj Abhyanker – +1 (408) 398-3126")
- Specific location mentions (city/town/neighborhood)
- Any standalone "Contact" or "Contact Information" sections (remove entirely)

2. PRESERVE EXACTLY (if present):
- All job titles, company names, and role designations
- All university, college, and school names
- All degrees, majors, minors, specializations, academic honors (e.g. GPA, "Summa Cum Laude")
- All skills, technologies, languages, frameworks, tools, platforms, libraries
- All project titles, descriptions, outcomes, and features
- All quantitative metrics (e.g. "reduced costs by 30%", "led a 12-member team")
- All line breaks, bullet points, and original resume formatting
- All certifications, course names, and issuing organizations
- All dates, durations, and timeframes (e.g. "Jun 2022 - Present", "3 months")
- All language proficiencies (e.g. "Fluent", "Native Proficiency")
- All awards, honors, recognitions, hackathon wins, etc.
- All conferences, publications, speaking engagements, talks
- All volunteer positions, extracurricular leadership, committee involvement
- All organizations, clubs, student teams, initiatives, and affiliations
- All startups, products, and project brand names
- All domains of focus (e.g. "BioTech", "AI Safety")
- All architectures, workflows, and methodologies (e.g. "Agile", "RAG", "CI/CD")
- All tools used in process (e.g. Figma, Notion, Jira, Tableau)

3. CORRECT and NORMALIZE text:
- Fix any spacing, word breaks, or tokenization issues from PDF/OCR extraction  
  e.g. "c ustomer" -> "customer", "T oyota" -> "Toyota"
- Fix obvious typos or broken lines ONLY if clearly wrong  
- Do not rephrase or paraphrase any content

---

🔧 OUTPUT FORMAT - STRICT JSON

You MUST return the sections array in this EXACT order, regardless of how they appear in the original resume:

1. Summary (if present)
2. Education (if present)
3. Experience (if present)
4. Projects (if present)
5. Skills (if present)
6. Certifications (if present)
7. Awards/Honors (if present)
8. Publications (if present)
9. Languages (if present)
10. Volunteer/Organizations (if present)

Return a single, valid JSON object structured EXACTLY like this:

{
  "candidateName": "Akshat Gupta",
  "sections": [
    {
      "title": "Summary",
      "type": "paragraphs",
      "paragraphs": ["..."]
    },
    {
      "title": "Education",
      "type": "education",
      "items": [...]
    },
    {
      "title": "Experience",
      "type": "experience",
      "items": [...]
    },
    {
      "title": "Projects",
      "type": "projects",
      "items": [...]
    },
    {
      "title": "Skills",
      "type": "bullets",
      "bullets": ["..."]
    },
    {
      "title": "Certifications",
      "type": "certifications",
      "items": [...]
    },
    {
      "title": "Honors-Awards",
      "type": "awards",
      "items": [...]
    },
    {
      "title": "Publications",
      "type": "publications",
      "items": [...]
    },
    {
      "title": "Languages",
      "type": "languages",
      "items": [...]
    },
    {
      "title": "Volunteer",
      "type": "volunteer",
      "items": [...]
    }
  ],
  "piiRemoved": 9
}

SECTION TYPE DEFINITIONS:

**paragraphs**: For text-heavy sections like Summary/Objective
{
  "title": "Summary",
  "type": "paragraphs",
  "paragraphs": ["paragraph 1", "paragraph 2"]
}

**bullets**: For simple list sections like Top Skills
{
  "title": "Top Skills",
  "type": "bullets",
  "bullets": ["Skill 1", "Skill 2", "Skill 3"]
}

**experience**: For work history
{
  "title": "Experience",
  "type": "experience",
  "items": [
    {
      "title": "Job Title",
      "company": "Company Name",
      "duration": "Jan 2020 - Present",
      "responsibilities": ["responsibility 1", "responsibility 2"]
    }
  ]
}

**education**: For academic background
{
  "title": "Education",
  "type": "education",
  "items": [
    {
      "degree": "Bachelor of Technology - BTech, Computer Science",
      "institution": "University Name",
      "duration": "2018 - 2022",
      "details": null
    }
  ]
}

**projects**: For project work
{
  "title": "Projects",
  "type": "projects",
  "items": [
    {
      "name": "Project Name",
      "description": "Project description",
      "technologies": ["Tech 1", "Tech 2"]
    }
  ]
}

**languages**: For spoken languages
{
  "title": "Languages",
  "type": "languages",
  "items": [
    {
      "name": "English",
      "proficiency": "Native or Bilingual"
    }
  ]
}

**certifications**: For professional certifications
{
  "title": "Certifications",
  "type": "certifications",
  "items": [
    {
      "name": "Certification Name",
      "duration": null,
      "issuing_organization": null,
      "credential_id": ""
    }
  ]
}

**awards**: For honors and awards
{
  "title": "Honors/Awards", (Choose the right title)
  "type": "awards",
  "items": [
    {
      "name": "Award Name",
      "duration": null,
      "description": null
    }
  ]
}

**volunteer**: For volunteer work and organizations
{
  "title": "Volunteer",
  "type": "volunteer",
  "items": [
    {
      "role": "Role Title",
      "organization": "Organization Name",
      "duration": "2020 - 2021",
      "location": "Metropolitan Area",
      "description": "Description of work"
    }
  ]
}

---

⚠️ CRITICAL ORDERING INSTRUCTIONS

THIS IS ABSOLUTELY MANDATORY - DO NOT DEVIATE:

Step 1: Parse ALL sections from the resume (do not skip any content except PII)
Step 2: Identify which category each section belongs to
Step 3: Separate sections into two groups:
   - Group A: Standard sections (Summary, Education, Experience, Projects, Skills, Certifications, Awards, Publications, Languages, Volunteer)
   - Group B: Any other sections not in the standard list (e.g., "Patents", "Conferences", "Research Interests", "Professional Affiliations", etc.)
Step 4: Output sections in THIS EXACT ORDER:
   FIRST - Standard sections in priority order:
   1. Summary
   2. Education
   3. Experience
   4. Projects
   5. Skills (including "Top Skills", "Technical Skills", "Core Competencies")
   6. Certifications
   7. Awards/Honors
   8. Publications
   9. Languages
   10. Volunteer
   
   THEN - All other sections (Group B) in their original order from the resume

CRITICAL: You MUST include ALL sections from the resume except "Contact" sections. If there are sections like "Patents", "Speaking Engagements", "Professional Memberships", "Research", "Interests", etc., include them AFTER the standard sections in their original order.

EXAMPLE: If the resume has these sections in this order:
- Contact (SKIP - it's PII)
- Top Skills
- Patents
- Languages
- Certifications
- Honors-Awards
- Summary
- Experience
- Research Interests
- Education

You MUST reorder them to:
- Summary (standard section #1)
- Education (standard section #2)
- Experience (standard section #3)
- Top Skills (standard section #5)
- Certifications (standard section #6)
- Honors-Awards (standard section #7)
- Languages (standard section #9)
- Patents (other section - keep original position relative to other "other" sections)
- Research Interests (other section - appears after Patents in original)

---

⚠️ ADDITIONAL RULES

- Do NOT output any "Contact" or "Contact Information" sections
- MUST include ALL other sections from the resume, even if not in the standard list
- For non-standard sections (e.g., "Patents", "Conferences", "Interests"), preserve their original title and structure
- If you encounter a section type not defined above, use type "entries" or "paragraphs" or "bullets" as appropriate
- CRITICAL: Do NOT use em dashes (—), use regular hyphens (-) for date ranges ONLY (e.g., "2020 - 2022")
- CRITICAL: Do NOT use bullet characters (•) anywhere in the output
- CRITICAL: Do NOT prefix list items with dashes (-) or any other symbols - return plain text strings in arrays
- CRITICAL: When parsing responsibilities/bullets that start with "- " or "• ", remove these prefixes completely
- Do NOT include null values - use empty strings "" or empty arrays []
- Return ONLY valid JSON, no markdown, no code fences, no commentary
- Do not hallucinate any information not in the original resume
- candidateName should be the person's full name as it appears at the top of the resume
- PRESERVE ALL CONTENT except PII - missing any section content is a critical error
- You MUST NOT create a section unless the resume text contains actual content for that section.
- If the resume does not include meaningful content for a section, completely omit that section from the output.
- Do NOT output empty sections and do NOT output section titles without content.
- If a section exists but contains no extractable non-PII content, OMIT the section entirely instead of outputting empty arrays or empty strings.
- Do NOT infer or assume any additional sections. Only create a section if the resume contains a clear heading or identifiable grouped content.



FORMATTING EXAMPLES:

WRONG:
"responsibilities": [
  "- Streamlined internal processes",
  "• Led a team of 10 engineers"
]

CORRECT:
"responsibilities": [
  "Streamlined internal processes",
  "Led a team of 10 engineers"
]

WRONG: "duration": "July 2021 – December 2022"
CORRECT: "duration": "July 2021 - December 2022"

### ⚠️ FINAL COMPLIANCE CHECK
Before outputting, verify:
1. Are standard sections in the correct order (Summary, Education, Experience, Projects, Skills, Certs, Awards, Pubs, Languages, Volunteer)?
2. Are ALL other sections included after the standard ones?
3. Is there any "Contact" section? (Remove it)
4. Have you removed ALL bullet characters (•) and dash prefixes (-) from list items?
5. Are em dashes (—) replaced with regular hyphens (-)?
6. Is the PII (phones, emails, links, addresses) gone?
7. Is ALL non-PII content from the original resume preserved?
8. Is the output valid JSON?

📝 Here is the resume text to anonymize:

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
def anonymize(
    payload: AnonymizeRequest,
    authorization: str = Header(None),
    org_id: str = Header(None)
):
    if not authorization:
        raise HTTPException(401, "Missing Authorization header")

    if not org_id:
        raise HTTPException(400, "Missing org-id header")

    resume_text = payload.resumeText
    if not resume_text or len(resume_text.strip()) < 50:
        raise HTTPException(400, "Invalid resumeText")

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

        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError("No response text from Gemini")

        cleaned = _strip_code_fences(text)
        parsed = json.loads(cleaned)

        return JSONResponse({
            "data": {
                "candidateName": parsed.get("candidateName", ""),
                "sections": parsed.get("sections", []),
                "piiRemoved": parsed.get("piiRemoved", 0),
                "id": "local-test-id",
                "processedBy": "gemini-local",
                "processingTime": 1200
            }
        })

    except Exception as e:
        raise HTTPException(500, f"Failed to anonymize resume: {str(e)}")


# Local dev entrypoint (Render uses startCommand)
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
