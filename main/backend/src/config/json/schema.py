JSON_JD_SCHEMA = {
    "title": "JobDescription",
    "description": "The structured representation of a job description.",
    "type": "object",
    "properties": {
        "job_title": {"type": "string", "description": "The title of the job position."},
        "company": {"type": "string", "description": "The name of the company hiring."},
        "location": {"type": "string", "description": "The location of the job (e.g., city, state, remote)."},
        "company_summary": {"type": "string", "description": "A brief summary of the company."},
        "responsibilities": {
            "type": "array",
            "description": "A list of key responsibilities for the role.",
            "items": {"type": "string"},
        },
        "required_qualifications": {
            "type": "array",
            "description": "A list of essential qualifications and skills.",
            "items": {"type": "string"},
        },
        "preferred_qualifications": {
            "type": "array",
            "description": "A list of desired but not essential qualifications.",
            "items": {"type": "string"},
        },
    },
    "required": ["job_title", "company", "responsibilities", "required_qualifications"],
}

JSON_RESUME_SCHEMA = {
    "title": "Resume",
    "description": "The structured representation of a resume.",
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "The full name of the candidate."},
        "summary": {"type": "string", "description": "A brief summary of the candidate's profile."},
        "work_experience": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "description": "The job title or role."},
                    "company": {"type": "string", "description": "The name of the company."},
                    "start_date": {"type": "string", "description": "The start date of the employment."},
                    "end_date": {"type": "string", "description": "The end date of the employment (or 'Present')."},
                    "responsibilities": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["role", "company", "responsibilities"],
            },
        },
        "education": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "degree": {"type": "string", "description": "The degree obtained."},
                    "institution": {"type": "string", "description": "The name of the institution."},
                    "graduation_date": {"type": "string", "description": "The graduation date."},
                },
                "required": ["degree", "institution"],
            },
        },
        "skills": {"type": "array", "items": {"type": "string"}},
        "extras": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "summary", "work_experience", "education", "skills"],
}


MARKDOWN_REPORT_SCHEMA = """
"# Candidate Evaluation Report  "

"## 1. Executive Summary "
"*Begin with: [mention the candidate name mentioned in CANDIDATE_RESUME] is being evaluated for the role of [Role Name].' Provide a concise 3–5 sentence summary of the candidate’s fit. End with a clear bottom-line recommendation on suitability.*  "

"## 2. Alignment with Core Job Requirements "
"*For each requirement listed below, assess alignment using direct evidence from the resume or transcript. Evaluate the candidate's profile against the most critical requirements from the job description*  "
"### Evidence of Alignment (Strengths) "
"*  **Requirement:** [Insert exact requirement from job description] "
"   **Evidence:** [Cite specific resume content or transcript quote or direct quotes from the interview transcript that demonstrate this skill] "
"*  **Requirement:** [Next requirement] "
"   **Evidence:** [Cite evidence]  "


"### Potential Gaps or Weaknesses "
"* **Requirement:** [Insert requirement where candidate shows weakness] "
"    **Evidence:** [Explain the gap, referencing a lack of experience on the resume or a weak/evasive answer from the transcript.]  "

"## 3. Technical and Behavioral Competencies "
"*Based on their interview answers, evaluate the following competencies.*  "
"### Technical Depth & Problem-Solving "
"*Analyze and assess the candidate's technical knowledge beyond surface-level claims. How the candidate approached technical questions. Was their logic sound? Did they articulate edge cases or fallbacks?*  "
"### Behavioral Indicators (e.g., Ownership, Collaboration) "
"*Based on situational and behavioral questions, assess work style, initiative, and team dynamics. Use transcript quotes where possible. What can be inferred about their work style? Look for indicators of ownership, proactivity, and how they might collaborate within a team.*  "


"## 4. Communication and Professionalism "
"*Evaluate clarity, professionalism, structure, and conciseness of the candidate's responses. Assess their professionalism, confidence, and ability to convey complex ideas effectively. Did they communicate ideas effectively? Were responses structured and concise?*  "

"## 5. Areas for Further Probing / Red Flags "
"*List any inconsistencies, knowledge gaps, or concerning statements. If none, write: 'No significant red flags identified.'*  "

"## 6. Detailed Interview Performance Analysis "
"*Provide a brief, insightful analysis for each key question from the transcript.* "
"*   **Question:** [Exact question from transcript] "
"    **Answer Summary:** [Brief summary the candidate's response.] "
"    **Evaluation:** [Assess the quality of the answer. Was it specific, generic, confident, hesitant? Did it fully address the question?] "
"*   **Question:** [Next question...] "
"    **Answer Summary:** [...] "
"    **Evaluation:** [...]  "

"## 7. Final Recommendation "
"**Recommendation:** [Choose one: **Strongly Recommend** / **Recommend** / **Recommend with Reservations** / **Do Not Recommend**]  "
"**Justification:** "
"*Synthesize strengths, weaknesses, and role priorities into a final, decisive justification. Weigh the candidate's strengths against their identified gaps in the context of the role's priorities.*  "
"**Proposed Next Steps:** "
"*Suggest a clear next action (e.g., 'Proceed to final round with team lead', 'Schedule a technical deep-dive on X', 'Reject at this stage').*"
"""


RUBRIC_METRIC_SCHEMA =  """
# Candidate Evaluation Rubric Template

## Overall Scoring Guide
- **5 (Exceptional)**: Clearly exceeds expectations; outstanding demonstration of skill/behavior
- **4 (Strong)**: Consistently meets and sometimes exceeds expectations; solid evidence
- **3 (Proficient)**: Meets expectations; adequate demonstration with room for growth
- **2 (Developing)**: Partially meets expectations; inconsistent or limited evidence
- **1 (Concern)**: Does not meet expectations; significant gaps or concerns observed

---

## 1. Alignment with Core Requirements (20%)
*How well the candidate's experience, skills and certifications map to the must-have items in the JD.*

| Submetric | Weight | Description | Score (0-5) | Evidence & Notes |
|-----------|--------|-------------|-------------|------------------|
| **Functional Skill Match** | 40% | Direct evidence that the candidate has performed the same duties / uses the same tools. | | |
| **Industry Knowledge** | 30% | Understanding of domain-specific practices, regulations or market trends. | | |
| **Relevant Certifications / Licenses** | 20% | Formal credentials that are required or highly valued for the role. | | |
| **Impactful Experience** | 10% | Quantified results (e.g., revenue growth, cost savings) that demonstrate impact. | | |

**Category Score:** ___ / 5  
**Weighted Contribution:** ___ / 1.0

---

## 2. Technical Depth & Problem-Solving (25%)
*Depth of knowledge, logical rigor and ability to design/implement solutions.*

| Submetric | Weight | Description | Score (0-5) | Evidence & Notes |
|-----------|--------|-------------|-------------|------------------|
| **Algorithmic Thinking** | 30% | Ability to break a problem into steps, reason about complexity, propose optimal solutions. | | |
| **System Design & Architecture** | 30% | Design of scalable, maintainable systems; awareness of trade-offs and edge-cases. | | |
| **Debugging & Troubleshooting** | 20% | Systematic approach to isolate bugs, use of instrumentation/logs, recovery strategies. | | |
| **Code Quality & Best Practices** | 20% | Readability, test coverage, adherence to style guides, security awareness. | | |

**Category Score:** ___ / 5  
**Weighted Contribution:** ___ / 1.25

---

## 3. Behavioral Indicators (20%)
*Soft-skill evidence that predicts how the candidate will work with others and handle uncertainty.*

| Submetric | Weight | Description | Score (0-5) | Evidence & Notes |
|-----------|--------|-------------|-------------|------------------|
| **Ownership & Accountability** | 25% | Taking responsibility for results, admitting mistakes, following through. | | |
| **Collaboration & Teamwork** | 25% | Evidence of effective partnership, knowledge-sharing, conflict resolution. | | |
| **Adaptability & Learning Agility** | 25% | Flexibility in changing environments, rapid up-skilling, openness to feedback. | | |
| **Initiative & Proactiveness** | 25% | Coming up with ideas, driving projects forward without being asked. | | |

**Category Score:** ___ / 5  
**Weighted Contribution:** ___ / 1.0

---

## 4. Communication & Professionalism (15%)
*Clarity, structure and professionalism in verbal & written exchanges.*

| Submetric | Weight | Description | Score (0-5) | Evidence & Notes |
|-----------|--------|-------------|-------------|------------------|
| **Clarity & Conciseness** | 30% | Answers are easy to follow, free of jargon, and to the point. | | |
| **Structured Thinking** | 30% | Logical flow, use of frameworks (STAR, PREP, etc.) to organise thoughts. | | |
| **Tone & Professionalism** | 20% | Polite, respectful, appropriate level of formality. | | |
| **Listening & Responsiveness** | 20% | Accurately addresses the asked question, asks clarifying questions when needed. | | |

**Category Score:** ___ / 5  
**Weighted Contribution:** ___ / 0.75

---

## 5. Cultural Fit & Values Alignment (10%)
*Alignment with the organization's mission, values and inclusive culture.*

| Submetric | Weight | Description | Score (0-5) | Evidence & Notes |
|-----------|--------|-------------|-------------|------------------|
| **Values Alignment** | 40% | Explicit statements or behaviours that match core corporate values. | | |
| **Motivation for the Role** | 30% | Genuine enthusiasm for the job, product or team. | | |
| **Diversity & Inclusion Mindset** | 20% | Evidence of championing inclusive practices or diverse perspectives. | | |
| **Long-Term Potential** | 10% | Willingness to grow with the company, career trajectory fit. | | |

**Category Score:** ___ / 5  
**Weighted Contribution:** ___ / 0.5

---

## 6. Growth Potential & Future Impact (10%)
*How likely the candidate is to develop into higher-impact roles.*

| Submetric | Weight | Description | Score (0-5) | Evidence & Notes |
|-----------|--------|-------------|-------------|------------------|
| **Learning Agility** | 30% | Quickly grasps new concepts, self-directed learning, curiosity. | | |
| **Leadership Potential** | 30% | Shows influence, coaching ability, vision-setting beyond the current role. | | |
| **Strategic Thinking** | 20% | Considers long-term implications, trade-offs, and business impact. | | |
| **Innovation & Creativity** | 20% | Proposes novel solutions, challenges status-quo constructively. | | |

**Category Score:** ___ / 5  
**Weighted Contribution:** ___ / 0.5

---

## Overall Assessment

### Final Weighted Score: ___ / 5.0

### Summary of Strengths:

### Areas for Development:

### Recommendation:
- [ ] Strong Recommend
- [ ] Recommend
- [ ] Recommend with Reservations
- [ ] Not Recommended

### Next Steps:
"""