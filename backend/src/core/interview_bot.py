# interview_bot.py - Optimized version
import os, re, json, random, time
from typing import List, Dict, Tuple, Set, Optional
from enum import Enum

from src.config.logging_config import get_logger
from src.config.settings import get_llm, get_embedding_model
from src.config.prompts import (
    PROFESSIONAL_INTERVIEWER_PROMPT,
    AUTHENTICITY_CHECK_PROMPT,
    COMPETENCY_EXTRACTION_PROMPT,
    COMPETENCY_EVALUATION_PROMPT,
    IDENTITY_VERIFICATION_PROMPT,
    PRE_INTERVIEW_VALIDATION_PROMPT,
    PRE_INTERVIEW_ANSWER_VALIDATION_PROMPT,
    IDENTITY_VERIFICATION_ANSWER_PROMPT,
)

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

logger = get_logger("interview_bot")

class ValidationState(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class InterviewBot:
    """
    Optimized interview bot with enhanced validation and user-friendly features.
    """
    
    def __init__(self, llm, jd_db, resume_db, interview_duration_minutes: int):
        self.llm = llm
        self.jd_db = jd_db
        self.resume_db = resume_db

        # Time management
        self.interview_duration_minutes = interview_duration_minutes
        self.start_time = time.time()
        self.time_warnings_sent = set()  # Track which time warnings have been sent

        # Context setup
        self.jd_context = self._extract_context(jd_db, "JD")
        self.resume_context = self._extract_context(resume_db, "Resume")

        self.question_generators = {}
        if self.jd_context:
            self.question_generators["SITUATIONAL"] = (self.generate_situational_question, {"jd_context": lambda: random.choice(self.jd_context)})
            self.question_generators["JD"] = (self.generate_jd_question, {"jd_context": lambda: random.choice(self.jd_context)})
            if self.resume_topics:
                self.question_generators["RAG"] = (self.generate_rag_question, {"jd_context": lambda: random.choice(self.jd_context), "resume_context": lambda: random.choice(self.resume_context)})

        if self.resume_context:
            self.question_generators["RESUME"] = (self.generate_resume_question, {"resume_context": lambda: random.choice(self.resume_context)})

        self.question_types = list(self.question_generators.keys())
        logger.info(f"Initialized with question types: {self.question_types}")
      

        # State Management
        self.interview_log: List[Dict[str, str]] = [] 
        self.conversation_history: List[str] = [] 
        self.current_topic = "Opening"
        self.current_question = ""
        self.interview_ended = False

        # Competency tracking
        self.target_competencies: List[str] = self._extract_target_competencies()
        self.completed_competencies: Set[str] = set()
        self.difficulty_level = 1

        # Pre-interview validation
        self.pre_interview_questions: List[str] = self._generate_pre_interview_questions()
        self.pre_interview_answers: List[Dict] = []
        self.validation_state: ValidationState = ValidationState.NOT_STARTED
        self.validation_attempts: int = 0
        self.max_validation_attempts = 2

        # Identity verification
        self.verification_questions: List[str] = self._generate_verification_questions()
        self.verification_step: int = 0
        self.in_verification: bool = len(self.verification_questions) > 0

        logger.info("[InterviewBot] Professional InterviewBot instance created with enhanced validation.")

    def _extract_context(self, db, db_type: str) -> str:
        context = f"\n--- {db_type} CONTEXT ---\n"
        try:
            if hasattr(db, 'docstore') and hasattr(db.docstore, '_dict'):
                docs = list(db.docstore._dict.values())
                context += "\n".join([doc.page_content for doc in docs])
            else:
                logger.warning(f"Could not extract context from {db_type} DB: {e}")

        except Exception as e:
            logger.warning(f"Could not extract context from {db_type} DB: {e}")
            context += "No specific information could be extracted."
        return context

    def _generate_pre_interview_questions(self) -> List[str]:
        """Generate 2-3 pre-interview validation questions."""
        logger.info("[InterviewBot] Generating pre-interview validation questions...")
        
        prompt = PRE_INTERVIEW_VALIDATION_PROMPT
        chain = prompt | self.llm

        response = self.log_and_invoke_llm(
            prompt, chain,
            {
                "jd_context": self.jd_context,
                "resume_context": self.resume_context
            },
            "Generate Pre-interview Questions"
        )

        parsed = self._parse_llm_json_response(response.content, "Pre-interview Questions")
        questions = parsed.get("questions", [])
        
        # Ensure we have at least 2 questions
        if len(questions) < 2:
            fallback_questions = [
                "Can you briefly summarize your most relevant experience for this role?",
                "What specific skills from the job description do you feel most confident about?",
                "Why are you interested in this particular position and company?"
            ]
            questions = fallback_questions[:2]
        
        return questions[:3]  # Max 3 questions

    def _validate_pre_interview_answer(self, question: str, answer: str, question_index: int) -> Dict:
        """Validate a pre-interview answer with detailed feedback."""
        logger.info(f"[InterviewBot] Validating pre-interview answer for question {question_index + 1}")
        
        validation_prompt = PRE_INTERVIEW_ANSWER_VALIDATION_PROMPT

        chain = validation_prompt | self.llm
        
        response = self.log_and_invoke_llm(
            validation_prompt, chain,
            {"question": question, "answer": answer},
            f"Validate Pre-interview Answer {question_index + 1}"
        )

        parsed = self._parse_llm_json_response(response.content, f"Answer Validation {question_index + 1}")
        
        # Default response if parsing fails
        # default_response = {
        #     "validation_status": "VAGUE",
        #     "confidence_score": 0.5,
        #     "feedback": "Unable to validate answer properly.",
        #     "needs_clarification": True,
        #     "suggested_followup": "Could you provide more specific details about your experience?"
        # } **default_response
        
        return {**parsed}

    def _generate_verification_questions(self) -> List[str]:
        """Generates identity verification questions."""
        logger.info("[InterviewBot] Generating identity verification questions...")
        
        if not self.jd_context.strip() or "No specific information" in self.jd_context:
            logger.warning("[InterviewBot] Insufficient JD context for verification questions")
            return []  # Skip verification if no JD context
            
        prompt = IDENTITY_VERIFICATION_PROMPT
        chain = prompt | self.llm

        response = self.log_and_invoke_llm(
            prompt, chain,
            {"jd_context": self.jd_context, "resume_context": self.resume_context},
            "Generate Verification Questions"
        )

        parsed = self._parse_llm_json_response(response.content, "Verification Questions")
        questions = parsed.get("questions", [])
        return questions[:3]  # Limit to 3 questions

    def _validate_verification_answer(self, question: str, answer: str) -> Dict:
        """Validate a verification answer."""
        logger.info(f"[InterviewBot] Validating verification answer for question {self.verification_step + 1}")

        validation_prompt = IDENTITY_VERIFICATION_ANSWER_PROMPT

        chain = validation_prompt | self.llm
        response = self.log_and_invoke_llm(
            validation_prompt, chain,
            {"resume_context": self.resume_context, "question": question, "answer": answer},
            f"Validate Verification Answer {self.verification_step + 1}"
        )

        parsed = self._parse_llm_json_response(response.content, f"Verification Validation {self.verification_step + 1}")
        default_response = {"validation_status": "VAGUE", "confidence_score": 0.5, "feedback": "Unable to validate answer.", "needs_clarification": True, "suggested_followup": "Can you elaborate on that?"}
        return {**default_response, **parsed}

    def _extract_target_competencies(self) -> List[str]:
        """Extract target competencies from job description."""
        logger.info("[InterviewBot] Extracting target competencies...")
        
        if not self.jd_context.strip() or "No specific information" in self.jd_context:
            logger.warning("[InterviewBot] Insufficient JD context for competency extraction")
            return ["Problem Solving", "Communication", "Technical Skills"]  # Fallback competencies
            
        prompt = COMPETENCY_EXTRACTION_PROMPT
        chain = prompt | self.llm

        response = self.log_and_invoke_llm(
            prompt, chain,
            {"jd_context": self.jd_context},
            "Extract Target Competencies"
        )

        parsed = self._parse_llm_json_response(response.content, "Competency Extraction")
        competencies = parsed.get("competencies", [])
        return competencies[:8] if len(competencies) > 8 else competencies

    def log_and_invoke_llm(self, prompt: ChatPromptTemplate, chain, inputs: dict, purpose: str):
        """Enhanced logging for LLM interactions."""
        log_safe_inputs = {k: (v.page_content if isinstance(v, Document) else str(v)[:500] + "..." if len(str(v)) > 500 else str(v)) 
                          for k, v in inputs.items()}
        
        logger.info(f"[InterviewBot | {purpose}]\n--- LLM Request Start ---")
        logger.info(f"Purpose: {purpose}")
        logger.debug(f"Inputs: {json.dumps(log_safe_inputs, indent=2)}")

        try:
            response = chain.invoke(inputs)
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            logger.debug(f"LLM Response: {response_content[:1000]}...")  # Limit log size
            logger.info(f"--- LLM Request End ---")
            return response
        except Exception as e:
            logger.error(f"LLM invocation failed for {purpose}: {e}")
            raise

    def _parse_llm_json_response(self, response_content: str, purpose: str) -> Dict:
        """Safely parse JSON from LLM response."""
        try:
            # Clean the response content
            cleaned_content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', response_content)
            match = re.search(r'\{.*\}', cleaned_content, re.DOTALL)
            
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
            else:
                logger.warning(f"No JSON found in {purpose} response")
                return {}
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Failed to parse JSON for {purpose}: {e}")
            return {}

    def _add_to_history(self, role: str, content: str):
        """Manage conversation history with size limits."""
        self.interview_log.append({"role": role, "content": content})
        self.conversation_history.append(f"{role.title()}: {content}")
        
        # Keep last 12 exchanges (6 back-and-forth)
        if len(self.conversation_history) > 12:
            self.conversation_history = self.conversation_history[-12:]

    def _get_history_str(self) -> str:
        """Get formatted conversation history."""
        if not self.conversation_history:
            return "No conversation history yet."
        return "\n".join(self.conversation_history[-10:]) 

    def _get_time_status(self) -> Dict[str, any]:
        """Get current time status with warnings."""
        elapsed_seconds = time.time() - self.start_time
        elapsed_minutes = int(elapsed_seconds // 60)
        remaining_minutes = max(0, self.interview_duration_minutes - elapsed_minutes)
        
        # Calculate percentage of time used
        time_used_percentage = (elapsed_minutes / self.interview_duration_minutes) * 100
        
        # Determine if we need to send time warnings
        time_warnings = []
        if remaining_minutes <= 5 and "5_min" not in self.time_warnings_sent:
            time_warnings.append("5 minutes remaining")
            self.time_warnings_sent.add("5_min")
        elif remaining_minutes <= 2 and "2_min" not in self.time_warnings_sent:
            time_warnings.append("2 minutes remaining - please conclude your thoughts")
            self.time_warnings_sent.add("2_min")
        elif remaining_minutes <= 0 and "time_up" not in self.time_warnings_sent:
            time_warnings.append("TIME_UP_SIGNAL")
            self.time_warnings_sent.add("time_up")
        
        return {
            "elapsed_minutes": elapsed_minutes,
            "remaining_minutes": remaining_minutes,
            "time_used_percentage": time_used_percentage,
            "warnings": time_warnings
        }

    def start_interview(self) -> str:
        """Start the interview with pre-interview validation phase."""
        welcome_message = (
            "Welcome to your interview! 👋\n\n"
            "We'll start with a few brief questions to better understand your background.\n"
            "Please provide specific, detailed answers based on your experience.\n\n"
            "Let's begin when you're ready."
        )
        
        self._add_to_history("bot", welcome_message)
        self.validation_state = ValidationState.IN_PROGRESS
        
        # Start with first pre-interview question
        if self.pre_interview_questions:
            first_question = self.pre_interview_questions[0]
            self.current_question = first_question
            self._add_to_history("bot", first_question)
            return f"{welcome_message}\n\n**First Question:** {first_question}"
        
        # Fallback if no pre-interview questions
        self.validation_state = ValidationState.COMPLETED
        return self._proceed_to_main_interview()

    def _proceed_to_main_interview(self) -> str:
        """Transition from pre-interview to main interview."""
        transition_message = (
            "Thank you for those initial answers. Now let's move to the main interview questions.\n\n"
            "Remember to provide specific examples from your experience where possible."
        )
        
        self._add_to_history("bot", transition_message)
        
        # Generate first main question
        if self.in_verification:
            first_question = self.verification_questions[0]
            self.current_question = first_question
            self._add_to_history("bot", first_question)
            return f"{transition_message}\n\n{first_question}"
        else:
            first_question = self._generate_next_question()
            return f"{transition_message}\n\n{first_question}"

    def process_user_answer(self, answer: str) -> str:
        """Process user answer with enhanced validation logic."""
        if self.interview_ended or answer == "TIME_UP_SIGNAL":
            self.interview_ended = True
            return "END_OF_INTERVIEW"

        self._add_to_history("user", answer)

        # Check time status
        time_status = self._get_time_status()
        if "TIME_UP_SIGNAL" in time_status["warnings"]:
            self.interview_ended = True
            return "END_OF_INTERVIEW"

        # Handle pre-interview validation phase
        if self.validation_state == ValidationState.IN_PROGRESS:
            return self._handle_pre_interview_validation(answer)

        # Handle verification phase
        if self.in_verification:
            return self._handle_verification_phase(answer)

        # Handle skip answers
        if self.check_skip_answer(answer):
            logger.info("[InterviewBot] Candidate skipped question.")
            return self._generate_next_question_with_time_status(time_status)

        # Main interview processing
        status, follow_up = self._check_answer_authenticity(self.current_question, answer)
        self._assess_competency_coverage(self.current_question, answer)
        self._adjust_difficulty_level(status, answer)

        if status == "GENERIC" and follow_up:
            self.current_question = follow_up
            self._add_to_history("bot", follow_up)
            return follow_up

        return self._generate_next_question_with_time_status(time_status)

    def _handle_pre_interview_validation(self, answer: str) -> str:
        """Handle pre-interview question validation."""
        current_question_index = len(self.pre_interview_answers)
        
        if current_question_index >= len(self.pre_interview_questions):
            # All questions answered, validate completion
            self.validation_state = ValidationState.COMPLETED
            return self._proceed_to_main_interview()

        # Validate current answer
        validation_result = self._validate_pre_interview_answer(
            self.pre_interview_questions[current_question_index],
            answer,
            current_question_index
        )

        # Store the answer and validation result
        self.pre_interview_answers.append({
            "question": self.pre_interview_questions[current_question_index],
            "answer": answer,
            "validation": validation_result
        })

        # Check if answer needs clarification
        if validation_result.get("needs_clarification", False) and self.validation_attempts < self.max_validation_attempts:
            self.validation_attempts += 1
            follow_up = validation_result.get("suggested_followup", 
                "Could you please provide more specific details about that?")
            
            self.current_question = follow_up
            self._add_to_history("bot", follow_up)
            return follow_up
        else:
            # Move to next question or proceed
            self.validation_attempts = 0  # Reset for next question
            
            if current_question_index + 1 < len(self.pre_interview_questions):
                # Next pre-interview question
                next_question = self.pre_interview_questions[current_question_index + 1]
                self.current_question = next_question
                self._add_to_history("bot", next_question)
                
                # Add encouragement for good answers
                if validation_result.get("validation_status") in ["COMPREHENSIVE", "ADEQUATE"]:
                    encouragement = "Thank you for that detailed answer. "
                else:
                    encouragement = "Thank you. "
                
                return f"{encouragement}Next question: {next_question}"
            else:
                # All pre-interview questions completed
                self.validation_state = ValidationState.COMPLETED
                return self._proceed_to_main_interview()

    def _handle_verification_phase(self, answer: str) -> str:
        current_question = self.verification_questions[self.verification_step]

        # Validate the answer
        validation_result = self._validate_verification_answer(current_question, answer)
        self.verification_answers.append({
            "question": current_question,
            "answer": answer,
            "validation": validation_result
        })

        # If answer is vague and we have attempts left, ask for clarification
        if validation_result.get("needs_clarification", False) and self.validation_attempts < self.max_validation_attempts:
            self.validation_attempts += 1
            follow_up = validation_result.get("suggested_followup", "Could you elaborate on that for me?")
            self.current_question = follow_up
            self._add_to_history("bot", follow_up)
            return follow_up

        # Otherwise, reset attempts and move to the next step
        self.validation_attempts = 0
        self.verification_step += 1

        if self.verification_step < len(self.verification_questions):
            next_question = self.verification_questions[self.verification_step]
            self.current_question = next_question
            self._add_to_history("bot", next_question)
            return next_question
        else:
            self.in_verification = False
            logger.info("[InterviewBot] Verification phase complete.")
            return self._generate_next_question()

    def _generate_next_question_with_time_status(self, time_status: Dict) -> str:
        """Generate next question with time considerations."""
        question = self._generate_next_question()
        if time_status["warnings"]:
            warning_text = "\n\n".join(time_status["warnings"])
            return f"{warning_text}\n\n{question}"
        return question

    def check_skip_answer(self, answer: str) -> bool:
        skip_phrases = [
            "i don't know", "i do not know", "don't know", "not sure",
            "no idea", "can't answer", "cannot answer", "skip", "pass",
            "i'm not sure", "im not sure", "not familiar", "repeat that",
            "can you repeat", "what was the question"
        ]
        answer_lower = answer.lower().strip()
        return any(phrase in answer_lower for phrase in skip_phrases)

    def _check_answer_authenticity(self, question: str, answer: str) -> Tuple[str, str | None]:
        """Enhanced authenticity check with better error handling."""
        logger.info("[InterviewBot] Checking answer authenticity...")
        
        try:
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
                logger.warning(f"Generic answer detected. Asking follow-up: {follow_up}")
                return "GENERIC", follow_up

            logger.info("Answer appears authentic.")
            return "AUTHENTIC", None
            
        except Exception as e:
            logger.error(f"Authenticity check failed: {e}")
            return "AUTHENTIC", None  # Fallback to authentic on error

    def _assess_competency_coverage(self, question: str, answer: str) -> None:
        """Assess competency coverage with error handling."""
        logger.info("[InterviewBot] Assessing competency coverage...")
        
        try:
            remaining_competencies = list(set(self.target_competencies) - self.completed_competencies)
            if not remaining_competencies:
                logger.info("All target competencies have been covered.")
                return

            prompt = COMPETENCY_EVALUATION_PROMPT
            chain = prompt | self.llm

            response = self.log_and_invoke_llm(
                prompt, chain,
                {
                    "remaining_competencies": json.dumps(remaining_competencies),
                    "question": question,
                    "answer": answer
                },
                "Assess Competency Coverage"
            )

            parsed = self._parse_llm_json_response(response.content, "Competency Assessment")
            assessed_competencies = parsed.get("assessed_competencies", [])

            for comp in assessed_competencies:
                if isinstance(comp, dict) and "competency" in comp:
                    competency_name = comp["competency"]
                    if competency_name in remaining_competencies:
                        self.completed_competencies.add(competency_name)
                elif isinstance(comp, str) and comp in remaining_competencies:
                    self.completed_competencies.add(comp)

            logger.info(f"Completed competencies: {list(self.completed_competencies)}")
            
        except Exception as e:
            logger.error(f"Competency assessment failed: {e}")

    def _adjust_difficulty_level(self, authenticity_status: str, answer: str) -> None:
        """Adjust difficulty level with safe error handling."""
        logger.info(f"Adjusting difficulty level. Current: {self.difficulty_level}")
        
        try:
            # Simple rule-based adjustment for reliability
            answer_length = len(answer.split())
            contains_examples = any(word in answer.lower() for word in 
                                  ['example', 'for instance', 'specifically', 'when i'])
            
            if authenticity_status == "AUTHENTIC" and answer_length > 30 and contains_examples:
                if self.difficulty_level < 5:
                    self.difficulty_level += 0.5
            elif answer_length < 15 or "not sure" in answer.lower():
                if self.difficulty_level > 1:
                    self.difficulty_level -= 0.5
                    
            self.difficulty_level = max(1, min(5, self.difficulty_level))
            logger.info(f"New difficulty level: {self.difficulty_level}")
            
        except Exception as e:
            logger.error(f"Difficulty adjustment failed: {e}")


    def generate_rag_question(self, jd_topic: Document, resume_db: FAISS) -> str:
        resume_context_docs = resume_db.similarity_search(jd_topic.page_content, k=1)
        resume_context = resume_context_docs[0].page_content if resume_context_docs else "No specific context found."
        prompt = ChatPromptTemplate.from_template(
            "You are an expert technical interviewer conducting a professional interview."
            "Create ONE focused, conversational question that naturally connects a job requirement to the candidate's background.\n\n"
            "Guidelines:\n"
            "- Ask only a single, clear question (no preambles, no bullet points).\n"
            "- Keep it conversational and professional\n"
            "- Focus on their actual experience and approach\n"
            "- Avoid overly generic or theoretical questions\n\n" \
            "- Keep the tone natural (e.g., “Can you tell me…”, “How did you …”).\n"
            "- Focus on concrete actions, tools, or outcomes the candidate has demonstrated.\n"
            "- **Maximum 30 words**.\n"
            "- Respond **only with the question** – no explanation or extra text."
            "JOB REQUIREMENT: {jd_context}\n"
            "CANDIDATE'S BACKGROUND: {resume_context}\n\n"
            "Your Question:"
        )
        chain = prompt | self.llm
        inputs = {"jd_context": jd_topic.page_content, "resume_context": resume_context}
        return self.log_and_invoke_llm(prompt, chain, inputs, "Generate RAG Question").content

    def generate_jd_question(self, jd_topic: Document) -> str:
        prompt = ChatPromptTemplate.from_template(
            "You are an expert technical interviewer conducting a professional interview."
            "Based on the job requirement below, ask ONE clear, professional question to assess the candidate's understanding and experience.\n\n"
            "Keep the question:\n"
            "- Ask only a single, clear question (no preambles, no bullet points).\n"
            "- Practical and relevant\n"
            "- Professional in tone\n\n"
            "- Avoid overly generic or theoretical questions\n\n" \
            "- Use a natural tone (e.g., “Can you tell me…”, “How did you …”).\n"
            "- Target concrete experience (e.g., tools, processes, outcomes).\n"
            "- **Maximum 20 words**.\n"
            "- Return **only the question**."
            "JOB REQUIREMENT: {jd_context}\n\n"
            "Your Question:"
        )
        chain = prompt | self.llm
        inputs = {"jd_context": jd_topic.page_content}
        return self.log_and_invoke_llm(prompt, chain, inputs, "Generate JD-focused Question").content

    def generate_resume_question(self, resume_topic: Document) -> str:
        prompt = ChatPromptTemplate.from_template(
            "You are an expert technical interviewer conducting a professional interview."
            "Based on the candidate's background below, ask ONE specific question to understand the candidate’s role, contributions, and impact.\n\n"
            "Guidelines:\n"
            "- Ask only a single, clear question (no preambles, no bullet points).\n"
            "- Probe for specific details about their work\n"
            "- Keep it conversational and professional\n"
            "- Avoid overly generic or theoretical questions\n" \
            "- Use a natural tone (e.g., “Can you tell me…”, “How did you …”).\n"
            "- **Maximum 20 words**.\n"
            "- Return **only the question**."
            "CANDIDATE'S BACKGROUND: {resume_context}\n\n"
            "Your Question:"
        )
        chain = prompt | self.llm
        inputs = {"resume_context": resume_topic.page_content}
        return self.log_and_invoke_llm(prompt, chain, inputs, "Generate Resume-focused Question").content

    def generate_situational_question(self, jd_topic: Document) -> str:
        prompt = ChatPromptTemplate.from_template(
            "You are a hiring manager. Based on the job responsibility below, create ONE practical scenario-based question that tests problem-solving and decision-making skills.\n\n"
            "Requirements:\n"
            "- Present a realistic scenario (use 'Imagine...' or 'Suppose...')\n"
            "- Make it specific to the role requirements, Keep the scenario tightly tied to the listed responsibility.\n"
            "- Ask only a single, clear question (no bullet points).\n"
            "- **Maximum 35 words**.\n"
            "- Avoid overly complex or multi-part scenarios\n\n"
            "JOB RESPONSIBILITY: {jd_context}\n\n"
            "Your Question:"
        )
        chain = prompt | self.llm
        inputs = {"jd_context": jd_topic.page_content}
        return self.log_and_invoke_llm(prompt, chain, inputs, "Generate Situational Question").content

    def _generate_candidate_question_from_strategy(self) -> str:
        if not self.question_types:
            logger.warning("No question generation strategies available. Using fallback.")
            return

        strategy_name = random.choice(self.question_types)
        strategy_function, args_config = self.question_generators[strategy_name]
        try:
            resolved_args = {key: value() if callable(value) else value for key, value in args_config.items()}
        except IndexError:
            logger.warning(f"Could not generate topic for strategy '{strategy_name}'. Using fallback.")
            return "What is a recent technical challenge you faced and how did you overcome it?"

        logger.info(f"Executing question strategy: {strategy_name}")
        return strategy_function(**resolved_args)
    


    def _generate_next_question(self) -> str:
        logger.info("[InterviewBot] Generating next question using dynamic strategy...")

        time_status = self._get_time_status()
        if time_status["remaining_minutes"] <= 0:
            self.interview_ended = True
            return "Thank you for your time. We've reached the end of our allocated time. This concludes our interview."

        try:
            candidate_question = self._generate_candidate_question_from_strategy()

            prompt = PROFESSIONAL_INTERVIEWER_PROMPT
            chain = prompt | self.llm

            remaining_competencies = list(set(self.target_competencies) - self.completed_competencies)
            coverage_status = {"assessed": list(self.completed_competencies), "remaining": remaining_competencies}

            inputs = {
                "current_topic": self.current_topic,
                "history": self._get_history_str(),
                "jd_context": self.jd_context,
                "resume_context": self.resume_context,
                "elapsed_time_minutes": time_status["elapsed_minutes"],
                "total_duration_minutes": self.interview_duration_minutes,
                "current_coverage_status": json.dumps(coverage_status),
                "difficulty_level": self.difficulty_level,
                "time_remaining_minutes": time_status["remaining_minutes"],
                "candidate_question": candidate_question 
            }

            response = self.log_and_invoke_llm(prompt, chain, inputs, "Generate Next Question (Refinement)")
            parsed = self._parse_llm_json_response(response.content, "Next Question")

            if not parsed:
                self.interview_ended = True
                return "I appreciate your time. We'll conclude the interview here. Thank you."

            decision = parsed.get("decision", "END_INTERVIEW").upper()
            question = parsed.get("question", "Thank you. That's all I have.")
            self.current_topic = parsed.get("new_topic", self.current_topic)

            if time_status["remaining_minutes"] < 2 and decision != "END_INTERVIEW":
                decision = "END_INTERVIEW"
                question = "We're nearly out of time. Thank you for this discussion."

            if decision == "END_INTERVIEW":
                self.interview_ended = True

            self.current_question = question
            self._add_to_history("bot", question)
            return question

        except Exception as e:
            logger.error(f"Question generation failed: {e}")
            self.interview_ended = True
            return "Thank you for your time. We'll conclude our session here."

    def save_interview_log(self, transcript_path: str):
        """Save interview log with pre-interview and verification results."""
        os.makedirs(os.path.dirname(transcript_path), exist_ok=True)

        enhanced_log = {
            "metadata": {
                "session_start_time": self.start_time,
                "duration_minutes": self.interview_duration_minutes,
                "target_competencies": self.target_competencies,
                "completed_competencies": list(self.completed_competencies),
                "pre_interview_validation": {
                    "state": self.validation_state.value,
                    "questions": self.pre_interview_questions,
                    "answers": self.pre_interview_answers
                },
                "identity_verification": { # NEW
                    "questions": self.verification_questions,
                    "answers": self.verification_answers
                }
            },
            "transcript": self.interview_log
        }

        with open(transcript_path, "w", encoding='utf-8') as f:
            json.dump(enhanced_log, f, indent=4)
        logger.info(f"Interview log saved to '{transcript_path}'")

def create_interview_bot(FAISS_JD_PATH: str, FAISS_RESUME_PATH: str, interview_duration: int) -> InterviewBot:
    """Create interview bot with enhanced error handling."""
    logger.info("Initializing optimized interview bot...")

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

