# interviewbot.py
import os, re, json, random, time
from typing import List, Dict, Tuple, Set

from src.config.logging_config import get_logger
from src.config.settings import get_llm, get_embedding_model
from src.config.prompts import (
    PROFESSIONAL_INTERVIEWER_PROMPT,
    AUTHENTICITY_CHECK_PROMPT,
    COMPETENCY_EXTRACTION_PROMPT,
    COMPETENCY_EVALUATION_PROMPT,
    IDENTITY_VERIFICATION_PROMPT,
)

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

logger = get_logger("interview_bot")

class InterviewBot:
    """
    Core interview bot that drives a live interview session.
    Features added:
      • Candidate‑experience guidelines (no aggressive anti‑cheating UX).
      • Identity‑verification questions generated from the JD.
      • Competency extraction & tracking (5‑10 core competencies).
      • Adaptive difficulty (1‑5) based on answer quality.
      • Chain‑of‑thought reasoning required in the LLM decision JSON.
    """
    def __init__(self, llm, jd_db, resume_db, interview_duration_minutes: int):
        self.llm = llm
        self.jd_db = jd_db
        self.resume_db = resume_db

        # Time management
        self.interview_duration_minutes = interview_duration_minutes
        self.start_time = time.time()  # Record the start time

        # # Pre-fetch key topics for context
        # self.jd_topics_docs = self.jd_db.similarity_search(
        #     "all key responsibilities and qualifications", k=5
        # )
        # self.resume_topics_docs = self.resume_db.similarity_search(
        #     "all work experience and projects", k=5
        # )
        # self.jd_context = "\n".join([d.page_content for d in self.jd_topics_docs])
        # self.resume_context = "\n".join([d.page_content for d in self.resume_topics_docs])


        self.jd_context = "\n--- KEY JOB DESCRIPTION INFO ---\n"
        if hasattr(jd_db, 'docstore') and hasattr(jd_db.docstore, '_dict'):
            self.jd_topics_docs = list(jd_db.docstore._dict.values())
            self.jd_context += "\n".join([doc.page_content for doc in self.jd_topics_docs])

        self.resume_context = "\n\n--- KEY RESUME INFO ---\n"
        if hasattr(resume_db, 'docstore') and hasattr(resume_db.docstore, '_dict'):
            self.resume_topics_docs = list(resume_db.docstore._dict.values())
            self.resume_context += "\n".join([doc.page_content for doc in self.resume_topics_docs])

        # State Management for Cohesive Interview Flow
        self.interview_log: List[Dict[str, str]] = [] 
        self.conversation_history: List[str] = [] 

        self.current_topic = "Opening"
        self.current_question = ""
        self.interview_ended = False

        # Competency tracking
        self.target_competencies: List[str] = self._extract_target_competencies()
        self.completed_competencies: Set[str] = set()


        # Difficulty level (1-5)
        self.difficulty_level = 1

        self.verification_questions: List[str] = self._generate_verification_questions()
        self.verification_step: int = 0
        self.in_verification: bool = len(self.verification_questions) > 0

        logger.info("[InterviewBot | __init__] Professional InterviewBot instance created.")
        logger.info(f"[InterviewBot | __init__] Target competencies: {self.target_competencies}")

        if self.in_verification:
            logger.info(
                f"[InterviewBot | __init__] Verification phase enabled with {len(self.verification_questions)} questions."
            )

    def _generate_verification_questions(self) -> List[str]:
        """Generates 2-3 identity verification questions based on the JD."""
        logger.info("[InterviewBot] Generating identity verification questions...")
        prompt = IDENTITY_VERIFICATION_PROMPT
        chain = prompt | self.llm

        response = self.log_and_invoke_llm(
            prompt, chain,
            {"jd_context": self.jd_context, "resume_context": self.resume_context},
            "Generate Verification Questions"
        )

        parsed = self._parse_llm_json_response(response.content, "Verification Questions")
        questions = parsed.get("questions", [])
        return questions


    def _extract_target_competencies(self) -> List[str]:
        """Extract 5-10 target competencies from the job description"""
        logger.info("[InterviewBot] Extracting target competencies from job description...")
        prompt = COMPETENCY_EXTRACTION_PROMPT
        chain = prompt | self.llm

        response = self.log_and_invoke_llm(
            prompt, chain,
            {"jd_context": self.jd_context},
            "Extract Target Competencies"
        )

        parsed = self._parse_llm_json_response(response.content, "Competency Extraction")
        competencies = parsed.get("competencies", [])

        # Limit to 5-10 competencies
        return competencies[:10] if len(competencies) > 10 else competencies

    def log_and_invoke_llm(self, prompt: ChatPromptTemplate, chain, inputs: dict, purpose: str):
        log_safe_inputs = {k: (v.page_content if isinstance(v, Document) else str(v)) for k, v in inputs.items()}
        formatted_prompt = prompt.format_prompt(**inputs).to_string()

        logger.info(f"[InterviewBot | log_and_invoke_llm]\n--- LLM Request Start ---")
        logger.info(f"Purpose: {purpose}")
        logger.debug(f"Raw Inputs: {json.dumps(log_safe_inputs, indent=2)}")
        logger.debug(f"Formatted Prompt for LLM:\n---\n{formatted_prompt}\n---")

        response = chain.invoke(inputs)
        response_content = response.content if hasattr(response, 'content') else str(response)

        logger.debug(f"LLM Raw Output: {response_content}")
        logger.info(f"--- LLM Request End ---\n[InterviewBot | log_and_invoke_llm]")
        return response

    def check_skip_answer(self, answer: str) -> bool:
        skip_phrases = [
            "i don't know", "i do not know", "don't know", "not sure",
            "no idea", "can't answer", "cannot answer", "skip", "pass",
            "i'm not sure", "im not sure", "not familiar"
        ]
        answer_lower = answer.lower().strip()
        return any(phrase in answer_lower for phrase in skip_phrases)

    def _add_to_history(self, role: str, content: str):
        """Helper to manage both the log and the prompt history."""
        self.interview_log.append({"role": role, "content": content})
        self.conversation_history.append(f"{role.title()}: {content}")
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

    def _get_history_str(self) -> str:
        """Get the string representation of the history for the prompt."""
        if not self.conversation_history:
            return "No conversation history yet."
        return "\n".join(self.conversation_history)

    def _parse_llm_json_response(self, response_content: str, purpose: str) -> Dict:
        """Safely parse JSON from LLM, even with surrounding text."""
        try:
            match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
            else:
                logger.warning(f"No JSON found in {purpose} response: {response_content}")
                return {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON for {purpose}: {e}\nResponse was: {response_content}")
            return {}
        except Exception as e:
            logger.exception(f"Unexpected error parsing JSON for {purpose}: {e}")
            return {}

    def _check_answer_authenticity(self, question: str, answer: str) -> Tuple[str, str | None]:
        """
        Runs the plagiarism/authenticity check.
        """
        logger.info("[InterviewBot] Checking answer authenticity...")
        prompt = AUTHENTICITY_CHECK_PROMPT
        chain = prompt | self.llm

        response = self.log_and_invoke_llm(
            prompt, chain, 
            {"question": question, "answer": answer},
            "Check Answer Authenticity"
        )

        parsed = self._parse_llm_json_response(response.content, "Authenticity Check")

        status = parsed.get("status", "AUTHENTIC").upper()
        follow_up = parsed.get("follow_up_question")

        if status == "GENERIC" and follow_up:
            logger.warning(f"[InterviewBot] Generic answer detected. Asking follow-up: {follow_up}")
            return "GENERIC", follow_up

        logger.info("[InterviewBot] Answer appears authentic.")
        return "AUTHENTIC", None
    
    
    def _assess_competency_coverage(self, question: str, answer: str) -> None:
        """Assess which competencies were covered in the answer"""
        logger.info("[InterviewBot] Assessing competency coverage...")

        # Create a prompt to assess which competencies were covered
        competency_assessment_prompt = COMPETENCY_EVALUATION_PROMPT
        chain = competency_assessment_prompt | self.llm

        remaining_competencies = list(set(self.target_competencies) - self.completed_competencies)
        if not remaining_competencies:
            logger.info("[InterviewBot] All target competencies have been covered.")
            return
        
        response = self.log_and_invoke_llm(
            competency_assessment_prompt, chain,
            {
                "remaining_competencies": json.dumps(remaining_competencies),
                "question": question,
                "answer": answer
            },
            "Assess Competency Coverage"
        )

        parsed = self._parse_llm_json_response(response.content, "Competency Assessment")
        assessed_competencies = parsed.get("assessed_competencies", [])

        # Update competency tracking
        for competency in assessed_competencies:
            if competency in remaining_competencies:
                self.completed_competencies.add(competency)
        
        remaining_after = list(set(self.target_competencies) - self.completed_competencies)
        logger.info(f"[InterviewBot] Assessed competencies in this turn: {assessed_competencies}")
        logger.info(f"[InterviewBot] Completed competencies so far: {list(self.completed_competencies)}")
        logger.info(f"[InterviewBot] Remaining competencies: {remaining_after}")

    def _adjust_difficulty_level(self, authenticity_status: str, answer: str) -> None:
        """Adjust difficulty level based on answer quality"""
        logger.info(f"[InterviewBot] Adjusting difficulty level. Current: {self.difficulty_level}")

        # Create a prompt to assess answer quality
        quality_assessment_prompt = ChatPromptTemplate.from_template(
            """You are an expert interviewer evaluating answer quality.

            Based on the authenticity status and answer content, assess the quality of the response.

            Authenticity Status: {authenticity_status}
            Answer: {answer}

            Respond with a JSON object in this format:
            {{
                "quality": "excellent" or "good" or "average" or "poor"
            }}
            """
        )

        chain = quality_assessment_prompt | self.llm

        response = self.log_and_invoke_llm(
            quality_assessment_prompt, chain,
            {
                "authenticity_status": authenticity_status,
                "answer": answer
            },
            "Assess Answer Quality"
        )

        parsed = self._parse_llm_json_response(response.content, "Quality Assessment")
        quality = parsed.get("quality", "average").lower()

        # Adjust difficulty based on quality
        if quality in ["excellent", "good"] and self.difficulty_level < 5:
            self.difficulty_level += 1
            logger.info(f"[InterviewBot] Increased difficulty level to {self.difficulty_level}")
        elif quality in ["poor", "average"] and self.difficulty_level > 1:
            self.difficulty_level -= 1
            logger.info(f"[InterviewBot] Decreased difficulty level to {self.difficulty_level}")

    def _generate_next_question(self) -> str:
        """
        The new core logic for generating the next question or ending the interview.
        """
        logger.info("[InterviewBot] Generating next professional question...")
        prompt = PROFESSIONAL_INTERVIEWER_PROMPT
        chain = prompt | self.llm

        elapsed_seconds = time.time() - self.start_time
        elapsed_time_minutes = int(elapsed_seconds // 60)
        
        remaining = list(set(self.target_competencies) - self.completed_competencies)
        coverage_status = {
            "assessed": list(self.completed_competencies),
            "remaining": remaining
        }
        
        inputs = {
            "current_topic": self.current_topic,
            "history": self._get_history_str(),
            "jd_context": self.jd_context,
            "resume_context": self.resume_context,
            "elapsed_time_minutes": elapsed_time_minutes,
            "total_duration_minutes": self.interview_duration_minutes,
            "current_coverage_status": json.dumps(coverage_status),
            "difficulty_level": self.difficulty_level
        }

        response = self.log_and_invoke_llm(prompt, chain, inputs, "Generate Next Question")
        parsed = self._parse_llm_json_response(response.content, "Next Question")

        if not parsed:
            logger.error("[InterviewBot] Failed to get valid JSON from interviewer LLM. Ending session.")
            self.interview_ended = True
            return "I seem to have lost my train of thought. We'll have to end the session here. Thank you."

        decision = parsed.get("decision", "END_INTERVIEW").upper()
        question = parsed.get("question", "Thank you. That's all I have.")
        self.current_topic = parsed.get("new_topic", self.current_topic)

        logger.info(f"[InterviewBot] LLM Decision: {decision} | New Topic: {self.current_topic} | Time: {elapsed_time_minutes}/{self.interview_duration_minutes} min")

        if decision == "END_INTERVIEW":
            self.interview_ended = True

        self.current_question = question
        self._add_to_history("bot", question)
        return question

    def start_interview(self) -> str:
        welcome_message = (
            "Hello! Thank you for joining me today. Before we begin, please review the following guidelines:\n\n"
            "1. This is a secure interview environment designed to assess your skills fairly.\n"
            "2. Please ensure you're in a quiet environment with a stable internet connection.\n"
            "3. Have your resume and the job description handy for reference.\n"
            "4. Answer honestly and provide specific examples from your experience.\n"
            "5. If you need clarification on any question, please ask.\n\n"
            "Let's begin."
        )
        self._add_to_history("bot", welcome_message)

        # Start with verification questions if they exist
        if self.in_verification:
            first_question = self.verification_questions[0]
            self.current_question = first_question
            self._add_to_history("bot", first_question)
            return f"{welcome_message}\n\n{first_question}"
        else:
            # Otherwise, generate the first real question
            first_question = self._generate_next_question()
            return f"{welcome_message}\n\n{first_question}"

    def process_user_answer(self, answer: str) -> str:
        # Check if interview should end due to time limit
        if self.interview_ended or answer == "TIME_UP_SIGNAL":
            self.interview_ended = True
            return "END_OF_INTERVIEW"

        self._add_to_history("user", answer)

        # Handle verification phase first
        if self.in_verification:
            self.verification_step += 1
            if self.verification_step < len(self.verification_questions):
                next_question = self.verification_questions[self.verification_step]
                self.current_question = next_question
                self._add_to_history("bot", next_question)
                return next_question
            else:
                self.in_verification = False # End of verification
                logger.info("[InterviewBot] Verification phase complete.")
                # Fall through to generate the first real question

        if self.check_skip_answer(answer):
            logger.info("[InterviewBot] Candidate skipped question.")
            return self._generate_next_question()

        status, follow_up = self._check_answer_authenticity(self.current_question, answer)
        self._assess_competency_coverage(self.current_question, answer)
        self._adjust_difficulty_level(status, answer)

        if status == "GENERIC" and follow_up:
            self.current_question = follow_up
            self._add_to_history("bot", follow_up)
            return follow_up

        return self._generate_next_question()

    def save_interview_log(self, transcript_path: str):
        os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
        with open(transcript_path, "w", encoding='utf-8') as f:
            json.dump(self.interview_log, f, indent=4)
        logger.info(f"\nInterview log has been saved to '{transcript_path}'")

def create_interview_bot(FAISS_JD_PATH: str, FAISS_RESUME_PATH: str, interview_duration: int) -> InterviewBot:
    logger.info("Initializing models and loading vector stores for interview...")

    if not os.path.exists(FAISS_JD_PATH) or not os.path.exists(FAISS_RESUME_PATH):
        logger.error("FAISS index directories not found.")
        raise FileNotFoundError("FAISS index directories not found.")

    try:
        jd_db = FAISS.load_local(FAISS_JD_PATH, get_embedding_model(), allow_dangerous_deserialization=True)
        resume_db = FAISS.load_local(FAISS_RESUME_PATH, get_embedding_model(), allow_dangerous_deserialization=True)
        logger.info("Vector stores loaded successfully.")
    except Exception as e:
        logger.exception("Failed to load FAISS indexes.")
        raise

    return InterviewBot(
        llm=get_llm(),
        jd_db=jd_db,
        resume_db=resume_db,
        interview_duration_minutes=interview_duration
    )