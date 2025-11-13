# datapreprocess.py
import os, re, json
from typing import List, Dict

from src.config.logging_config import get_logger
from src.config.settings import get_llm, get_embedding_model

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda

from docling.document_converter import DocumentConverter, ConversionStatus

logger = get_logger("data_processor")

def safe_convert(file_path: str) -> str:
    try:
        logger.info(f"[Data Processor | safe_convert] Attempting to convert document at {file_path}...")
        converter = DocumentConverter()
        result = converter.convert(file_path)

        if result.status == ConversionStatus.SUCCESS:
            text_content = result.document.export_to_text()
            logger.info(f"[Data Processor | safe_convert] Document conversion successful for {file_path}.")
            return text_content
        else:
            logger.error(f"[Data Processor | safe_convert] Error: Document conversion failed for {file_path} with status {result.status}.")
            raise Exception(f"Conversion failed: {result.status}")
    except Exception as e:
        logger.warning(f"[Data Processor | safe_convert] Warning: Document conversion failed for {file_path}. Error: {str(e)}")
        raise Exception(f"File conversion error: {str(e)}")

def save_as_md(file_path: str, output_path: str = None) -> str:
    text_content = safe_convert(file_path)

    if output_path is None:
        base_name = file_path.rsplit('.', 1)[0]  
        output_path = f"{base_name}.md"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text_content)

    return output_path

def extract_json_from_string(text: str) -> str:
    text_content = text.content if hasattr(text, 'content') else str(text)
    match = re.search(r'{.*}', text_content, re.DOTALL)

    if match:
        return match.group(0)
    else:
        logger.warning(f"[Data Processor | extract_json_from_string] Warning: No JSON object found in the LLM output. Full output was: {text_content}")
        return "{}"

def extract_data(text: str, schema: dict, doc_type: str) -> Dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert JSON extraction engine. "
         "Given a text and a schema, your sole task is to extract the information and respond with a single, valid JSON object. "
         "Adhere strictly to the provided JSON schema. "
         "IMPORTANT: Your entire response must be only the JSON object, with no other text, thoughts, or explanations before or after it. "
         "Your response MUST start with {{ and end with }}."),
        ("human", "Extract information from the following {doc_type} text that matches this schema:\n"
         "---SCHEMA---\n{schema}\n---END SCHEMA---\n\n"
         "---TEXT---\n{text_input}\n---END TEXT---"),
    ])

    parser = JsonOutputParser()

    # FIX: Call the get_llm() function to get the model instance
    chain = (
        prompt
        | get_llm() 
        | RunnableLambda(extract_json_from_string)
        | parser
    )

    logger.info(f"[Data Processor | extract_data] Invoking LLM for agentic extraction of {doc_type}...")
    result = chain.invoke({
        "text_input": text,
        "schema": json.dumps(schema, indent=2),
        "doc_type": doc_type
    })
    logger.info("[Data Processor | extract_data] Extraction complete.")
    return result    

def create_semantic_chunks(data: Dict, doc_type: str) -> List[Document]:
    chunks = []
    name = data.get('name', '')
    company = data.get('company', 'N/A')
    job_title = data.get('job_title', 'N/A')

    if doc_type == "Resume":
        chunks.append(Document(page_content=f"Candidate Name: {name}. Summary: {data.get('summary', '')}", metadata={"category": "summary", "name": name}))
        for job in data.get('work_experience', []):
            content = f"Role: {job.get('role')} at {job.get('company')} ({job.get('start_date')} - {job.get('end_date')}). Responsibilities: {' '.join(job.get('responsibilities', []))}"
            chunks.append(Document(page_content=content, metadata={"category": "work_experience", "company": job.get('company'), "role": job.get('role'), "name": name}))

        for edu in data.get('education', []):
            content = f"Degree: {edu.get('degree')} from {edu.get('institution')} (Graduated: {edu.get('graduation_date')})."
            chunks.append(Document(page_content=content, metadata={"category": "education", "institution": edu.get('institution'), "name": name}))

        if skills := data.get('skills', []):
            chunks.append(Document(page_content=f"Skills: {', '.join(skills)}", metadata={"category": "skills", "name": name}))

    elif doc_type == "Job Description":
        overview_content = f"Job Title: {job_title} at {company}. Location: {data.get('location', 'N/A')}. Company Summary: {data.get('company_summary', '')}"
        chunks.append(Document(page_content=overview_content.strip(), metadata={"category": "overview", "company": company, "job_title": job_title}))

        if responsibilities := data.get('responsibilities', []):
            chunks.append(Document(page_content=f"Key Responsibilities: {' '.join(responsibilities)}", metadata={"category": "responsibilities", "company": company, "job_title": job_title}))

        if required := data.get('required_qualifications', []):
            chunks.append(Document(page_content=f"Required Qualifications: {' '.join(required)}", metadata={"category": "required_qualifications", "company": company, "job_title": job_title}))

        if preferred := data.get('preferred_qualifications', []):
            chunks.append(Document(page_content=f"Preferred Qualifications: {' '.join(preferred)}", metadata={"category": "preferred_qualifications", "company": company, "job_title": job_title}))

    logger.info(f"[Data Processor | create_semantic_chunks] Created {len(chunks)} semantic chunks for the {doc_type}.")
    return chunks

def process_and_vectorize(file_path: str, schema: dict, index_path: str, doc_type: str):
    logger.info(f"\n[Data Processor] --- Processing {doc_type} from {file_path} ---")

    try:
        logger.info(f"[Data Processor | process_and_vectorize] Docling document conversion started...")
        text = safe_convert(file_path)
        logger.info(f"[Data Processor | process_and_vectorize] Docling document conversion complete.")

        logger.info(f"[Data Processor | process_and_vectorize] File read successfully. Length: {len(text)} characters")

        structured_data = extract_data(text, schema, doc_type)
        logger.info(f"[Structred Data {doc_type}]: \n {json.dumps(structured_data, indent=2)}")


        documents = create_semantic_chunks(structured_data, doc_type)
        logger.debug(f"[Semantic Chunks {doc_type}]: \n {documents}")

        if not documents:
            logger.error(f"No documents were created for {doc_type}. Check chunking logic.")
            raise ValueError(f"No documents were created for {doc_type}.")

        logger.info(f"[Data Processor | process_and_vectorize] Initializing embedding model for {doc_type}...")

        # FIX: Call the get_embedding_model() function
        db = FAISS.from_documents(documents, get_embedding_model())

        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        db.save_local(index_path)

        logger.info(f"[Data Processor | process_and_vectorize] ✅ FAISS index for {doc_type} saved to '{index_path}'.")

    except FileNotFoundError:
        logger.error(f"File not found at path: {file_path}")
        raise

    except Exception as e:
        logger.exception(f"An unexpected error occured while processing {file_path}")
        raise