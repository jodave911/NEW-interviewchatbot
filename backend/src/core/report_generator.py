# reportgenerator.py
import os
import json
from typing import Dict, List

from src.config.logging_config import get_logger
from src.config.settings import get_llm, get_embedding_model
from src.config.prompts import EVALUATION_REPORT_PROMPT

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

logger = get_logger("report_generator")

def _gather_comprehensive_context(jd_db: FAISS, resume_db: FAISS, transcript_log: List[Dict]) -> str:
    logger.info("[Report Generator] Aggregating context from all sources...")

    transcript_str = "--- INTERVIEW TRANSCRIPT ---\n"
    for item in transcript_log:
        role = item.get("role", "unknown").title()
        content = item.get("content", "")
        transcript_str += f"{role}: {content}\n"

    jd_context_str = "\n--- KEY JOB DESCRIPTION INFO ---\n"
    if hasattr(jd_db, 'docstore') and hasattr(jd_db.docstore, '_dict'):
        jd_docs = list(jd_db.docstore._dict.values())
        jd_context_str += "\n".join([doc.page_content for doc in jd_docs])

    resume_context_str = "\n\n--- KEY RESUME INFO ---\n"
    if hasattr(resume_db, 'docstore') and hasattr(resume_db.docstore, '_dict'):
        resume_docs = list(resume_db.docstore._dict.values())
        resume_context_str += "\n".join([doc.page_content for doc in resume_docs])

    logger.info("[Report Generator] Context aggregation complete.")

    return transcript_str, jd_context_str, resume_context_str

def _log_and_invoke_llm(prompt: ChatPromptTemplate, chain, inputs: dict, purpose: str):
    formatted_prompt = prompt.format_prompt(**inputs).to_string()

    logger.info(f"[Report Generator | _log_and_invoke_llm]\n--- LLM Request Start ---")
    logger.info(f"Purpose: {purpose}")
    logger.debug(f"Formatted Prompt for LLM:\n---\n{formatted_prompt}\n---")

    response = chain.invoke(inputs)
    response_content = response.content if hasattr(response, 'content') else str(response)

    logger.debug(f"LLM Raw Output: {response_content}")
    logger.info(f"--- LLM Request End ---\n[Report Generator | _log_and_invoke_llm]")
    return response

def _generate_evaluation_report(llm, transcript_str: str, jd_context_str: str, resume_context_str: str) -> str:
    prompt = EVALUATION_REPORT_PROMPT

    chain = prompt | llm
    logger.info("[Report Generator] Generating the final report...")

    response = _log_and_invoke_llm(prompt, chain, {
        "transcript_str": transcript_str,
        "jd_context_str": jd_context_str,
        "resume_context_str": resume_context_str
    }, "Generate Full Candidate Report")

    return response.content

def _setup_environment(jd_index_path: str, resume_index_path: str, transcript_path: str):
    logger.info("[Report Generator] Initializing models and loading data sources...")

    try:
        jd_db = FAISS.load_local(jd_index_path, get_embedding_model(), allow_dangerous_deserialization=True)
        resume_db = FAISS.load_local(resume_index_path, get_embedding_model(), allow_dangerous_deserialization=True)
        logger.info("[Report Generator] Vector stores loaded successfully.")
    except Exception as e:
        logger.exception("[Report Generator] Failed to load FAISS indexes.")
        raise

    if not os.path.exists(transcript_path):
        logger.error(f"[Report Generator] Interview transcript not found at '{transcript_path}'.")
        raise FileNotFoundError(f"Interview transcript not found at '{transcript_path}'.")

    with open(transcript_path, "r", encoding='utf-8') as f:
        transcript_log = json.load(f)

    if not transcript_log:
        logger.error("[Report Generator] Interview transcript is empty.")
        raise ValueError("Interview transcript is empty.")

    logger.info("[Report Generator] Setup complete. All data sources loaded.")
    return jd_db, resume_db, transcript_log

def create_report(jd_index_path: str, resume_index_path: str, transcript_path: str, output_path: str):
    logger.info(f"\n[Report Generator] --- Generating candidate evaluation report ---")

    try:
        jd_db, resume_db, transcript_log = _setup_environment(jd_index_path, resume_index_path, transcript_path)

        transcript_str, jd_context_str, resume_context_str = _gather_comprehensive_context(jd_db, resume_db, transcript_log)

        report = _generate_evaluation_report(
            get_llm(), 
            transcript_str, 
            jd_context_str, 
            resume_context_str
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(f"Report has been saved to '{output_path}'")

    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Error during report generation: {e}")
        raise
    except Exception as e:
        logger.exception(f"An unexpected error occurred during report generation")
        raise