uvicorn main:app --host 0.0.0.0 --port 8000 --reload
npm run dev

npm install next-themes lucide-react @radix-ui/react-dialog tailwind-merge tailwindcss-animate @tailwindcss/typography mini-svg-data-uri


1. Implement "Chain of Thought" (CoT) Decision Making
Currently, your prompt asks for a JSON decision immediately. LLMs perform significantly better when forced to "think" before they "speak."

The Upgrade: Change the JSON schema in your PROFESSIONAL_INTERVIEWER_PROMPT to require a reasoning field before the decision field. This forces the model to generate logical tokens that ground its final decision.

New Prompt Structure (src/config/prompts.py):

Python

STRATEGIC_THINKING_PROMPT = """
... (Context Setup) ...

**Your Mental Process:**
Before generating the next question, you must perform a strategic review:
1. **Analyze the Candidate's Last Answer:** Was it shallow, theoretical, or deeply practical? Did they miss the core concept?
2. **Check Coverage:** Look at the JD. What critical skills (e.g., System Design, DB optimization) haven't we touched yet?
3. **Determine Difficulty:** If they answered the last question easily, INCREASE difficulty. If they struggled, SIMPLIFY or guide them.

**Response Format:**
You must output a JSON object in this exact order:
{
    "analysis": "Candidate explained the WHAT of Redis, but missed the HOW of eviction policies. Answer was B-grade.",
    "strategy": "I need to push them harder on caching strategies to see if they have production experience.",
    "difficulty_adjustment": "INCREASE",
    "decision": "DEEPEN",
    "question": "You mentioned using Redis for caching. Can you walk me through how you handled cache invalidation in that specific scenario, and what happened when the cache filled up?",
    "new_topic": "Redis - Eviction Policies"
}
"""
Why this works: By generating the analysis and strategy strings first, the LLM conditions itself to ask a specific, targeted question rather than a generic one.

2. Add "Competency Tracking" (Stateful Memory)
Currently, the bot relies on history (text) to know what happened. It doesn't "know" that it has covered Python but missed SQL.

The Upgrade: Add a structured competency_scorecard to the InterviewBot class.

Backend Logic Update (src/core/interview_bot.py):

Initialize: When the bot starts, parse the JD (using an LLM call) to extract a list of 5-7 target_competencies (e.g., ["Python", "FastAPI", "AWS", "Communication"]).

Update Loop: After every answer, ask the LLM (in the background) to "tick off" or "grade" the competency discussed.

Inject into Prompt: Pass this scorecard into the PROFESSIONAL_INTERVIEWER_PROMPT.

Python Logic:

Python

class InterviewBot:
    def __init__(self, ...):
        # ... existing init ...
        # New State
        self.target_competencies = ["API Design", "Database", "Cloud", "Leadership"] # Extracted from JD
        self.covered_competencies = {} # e.g., {"API Design": "Strong", "Database": "Pending"}

    def _generate_next_question(self):
        # Filter out what we've already covered
        remaining_topics = [t for t in self.target_competencies if t not in self.covered_competencies]
        
        inputs = {
            # ... existing inputs ...
            "remaining_checklist": ", ".join(remaining_topics),
            "current_coverage_status": json.dumps(self.covered_competencies)
        }
        # ... call LLM ...
Prompt Update:

"You have currently covered: {current_coverage_status}. You still MUST assess: {remaining_checklist}. If the current topic is exhausted, prioritize a topic from the 'MUST assess' list."

3. Adaptive Difficulty (The "Pressure Test")
A real interviewer adjusts based on the candidate's performance. If a candidate is crushing it, you stop asking "What is X?" and start asking "Design a system that handles X at scale."

The Upgrade: Pass a difficulty_level state (1-5) into the prompt.

Backend Logic:

Evaluate: Use your existing AUTHENTICITY check (or the new CoT analysis) to grade the previous answer.

Adjust:

If Grade == Great -> difficulty_level += 1

If Grade == Poor -> difficulty_level -= 1 (or switch topics).

Prompt Instruction:

Prompt Injection:

"Current Difficulty Level: {difficulty_level}/5"

Level 1: Definitional questions ("What is polymorphism?")

Level 3: Application questions ("When would you use polymorphism over if/else chains?")

Level 5: Edge-case/Architectural questions ("How does polymorphism impact runtime performance in high-frequency trading systems?")

Instruction: Generate a question strictly matching the Current Difficulty Level.

Summary of the "Advanced" Flow
User Answers.

LLM 1 (Analyst): specific breakdown of the answer. Updates competency_scorecard and calculates next_difficulty.

Logic: Selects the next target topic from the remaining_checklist.

LLM 2 (Strategist): Uses CoT prompting to formulate a question that:

Targets the selected topic.

Matches the calculated difficulty.

Uses the strategy derived from the Analyst step.

This moves your bot from a random question generator to a targeted, adaptive interviewer.
