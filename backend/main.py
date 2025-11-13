# main.py
import os
import json
import time
import uuid
import asyncio
import re
import logging
import shutil
from pathlib import Path
import logging.handlers

import aiofiles
from fastapi import (
    FastAPI,
    Request,
    UploadFile,
    File,
    Form,
    HTTPException,
    BackgroundTasks,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware


# -------------------------------------------------------------
# Logging (once at start)
# -------------------------------------------------------------
try:
    from src.config.logging_config import setup_base_logging, get_logger
except ImportError:
    print("Error: Could not import logging config. Make sure src/config/logging_config.py exists.")
    exit(1)

setup_base_logging()
logger = get_logger("api")

# -------------------------------------------------------------
# Settings & JSON schemas
# -------------------------------------------------------------
try:
    from src.config.settings import (
        ROOT_DIR,
        JSON_JD_SCHEMA,
        JSON_RESUME_SCHEMA,
    )
except ImportError:
    logger.error("Error: Could not import settings. Make sure src/config/settings.py exists.")
    ROOT_DIR = Path(__file__).parent
    JSON_JD_SCHEMA = {}
    JSON_RESUME_SCHEMA = {}


try:
    from src.core.interview_bot import create_interview_bot, InterviewBot
    from src.core.report_generator import create_report          
    from src.core.data_preprocessing import process_and_vectorize

except ImportError as e:
    logger.error(f"Fatal Error: Could not import core logic from 'src' or 'src/core'. {e}")
    logger.error("Please ensure your Python files (interview_bot.py, etc.) are in the correct 'src' directory.")
    exit(1)


# -------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------
app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost:3001",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------
# Health‑check
# -------------------------------------------------------------
@app.get("/ping")
async def ping():
    return {"msg": "pong"}

# -------------------------------------------------------------
# Helper – sanitise a candidate name for folder names
# -------------------------------------------------------------
def _sanitize_name(name: str) -> str:
    """Replace any non‑alphanumeric run with '_' and lower‑case."""
    safe = re.sub(r"\W+", "_", name.strip())
    return safe.strip("_").lower()

# -------------------------------------------------------------
# Helper – build a session id (optionally prefixed by a safe name)
# -------------------------------------------------------------
def _new_session_id(name: str | None = None) -> str:
    """<sanitised_name>-<uuid>   or just <uuid>"""
    uid = uuid.uuid4().hex
    if name:
        safe = _sanitize_name(name)
        if safe:
            return f"{safe}-{uid}"
    return uid

# -------------------------------------------------------------
# ①  Upload endpoint – creates a per‑session FAISS index
# -------------------------------------------------------------
@app.post("/upload")
async def upload_files(
    jd_file: UploadFile = File(..., description="Job Description (txt or json)"),
    resume_file: UploadFile = File(..., description="Candidate Resume (txt or json)"),
    candidate_name: str = Form(..., description="Candidate name, e.g. John Doe"),
    interview_duration: int = Form(15, description="Interview duration in minutes"),
):
    """
    * Saves the raw JD & Resume files under ``uploads/<session_id>/``.
    * Writes a tiny ``metadata.json`` (candidate name, duration,
      interview_completed flag + report_generated flag).
    * Creates a folder ``data/<sanitised_candidate_name>/`` that holds:
        - logs/
        - vector_store/<session_id>/…
        - transcripts/
        - reports/
    * Runs ``process_and_vectorize`` (LLM → structured data → FAISS) **synchronously**.
    * Returns the newly generated ``session_id``.
    """
    if not candidate_name or candidate_name.isspace():
        raise HTTPException(status_code=400, detail="Candidate Name is required.")

    session_id = _new_session_id(candidate_name)
    logger.info(f"🚀 New interview session: {session_id}")

    # -----------------------------------------------------------------
    # Prepare candidate‑specific folder under ``data/``
    # -----------------------------------------------------------------
    safe_name = _sanitize_name(candidate_name)
    if not safe_name:
        safe_name = "default"

    candidate_dir = ROOT_DIR / "data" / safe_name
    candidate_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Create a per‑candidate rotating file logger (writes to logs/)
    # ---------------------------------------------------------------
    candidate_log_dir = candidate_dir / "logs"
    candidate_log_dir.mkdir(parents=True, exist_ok=True)
    candidate_log_path = candidate_log_dir / f"{safe_name}.log"

    candidate_logger = logging.getLogger(f"candidate.{safe_name}")
    candidate_logger.setLevel(logging.DEBUG)

    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(candidate_log_path) for h in candidate_logger.handlers):
        file_handler = logging.handlers.RotatingFileHandler(
            candidate_log_path,
            maxBytes=15 * 1024 * 1024,
            backupCount=5,
            encoding="utf8",
        )
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        file_handler.setLevel(logging.DEBUG)
        candidate_logger.addHandler(file_handler)

    candidate_logger.info(f"Starting upload for session {session_id}")


    # -----------------------------------------------------------------
    # Save the raw files (still under ``uploads/<session_id>/``)
    # -----------------------------------------------------------------
    upload_dir = ROOT_DIR / "uploads" / session_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    jd_path = upload_dir / "job_description.txt"
    resume_path = upload_dir / "resume.txt"

    try:
        async with aiofiles.open(jd_path, "wb") as out:
            await out.write(await jd_file.read())
        async with aiofiles.open(resume_path, "wb") as out:
            await out.write(await resume_file.read())
    except Exception as e:
        logger.error(f"Failed to write uploaded files: {e}")
        raise HTTPException(status_code=500, detail="Error saving uploaded files.")

    # -----------------------------------------------------------------
    # Store metadata
    # -----------------------------------------------------------------
    metadata = {
        "candidate_name": candidate_name,
        "interview_duration": int(interview_duration), # NEW
        "interview_completed": False,
        "report_generated": False,
        "report_filename": "",
    }
    meta_path = upload_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    candidate_logger.debug(f"Metadata for {session_id}: {metadata}")

    # -----------------------------------------------------------------
    # FAISS index destinations (inside the candidate folder)
    # -----------------------------------------------------------------
    jd_index_path = candidate_dir / "vector_store" / session_id / "faiss_jd_index"
    resume_index_path = (
        candidate_dir / "vector_store" / session_id / "faiss_resume_index"
    )

    # -----------------------------------------------------------------
    # Vectorise
    # -----------------------------------------------------------------
    loop = asyncio.get_event_loop()
    try:
        candidate_logger.info(f"🔹 Starting vectorisation for session {session_id} …")
        await loop.run_in_executor(
            None,
            process_and_vectorize,
            str(jd_path),
            JSON_JD_SCHEMA,
            str(jd_index_path),
            "Job Description",
        )
        await loop.run_in_executor(
            None,
            process_and_vectorize,
            str(resume_path),
            JSON_RESUME_SCHEMA,
            str(resume_index_path),
            "Resume",
        )
        candidate_logger.info(f"✅ Vectorisation finished for session {session_id}")
    except Exception as exc:
        candidate_logger.exception(f"❌ Index creation failed for session {session_id}")
        if upload_dir.exists():
            shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(exc))

    # -----------------------------------------------------------------
    # Return the session id. The frontend will construct the URL.
    # -----------------------------------------------------------------
    return JSONResponse(
        {
            "session_id": session_id,
            "message": "Files uploaded and indexed successfully",
        }
    )

# -------------------------------------------------------------
# ②  List all existing sessions (used by Manager UI)
# -------------------------------------------------------------
@app.get("/sessions")
async def list_sessions():
    """
    Returns a list like:
        [
            {"id": "<session_id>", "name": "<candidate_name>",
             "completed": bool, "report_generated": bool},
            …
        ]
    """
    base = ROOT_DIR / "uploads"
    if not base.is_dir():
        return []

    sessions = []
    for p in base.iterdir():
        if not p.is_dir():
            continue
        meta_path = p / "metadata.json"
        name = ""
        completed = False
        report_generated = False
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                name = meta.get("candidate_name", "")
                completed = bool(meta.get("interview_completed", False))
                report_generated = bool(meta.get("report_generated", False))
                report_filename = meta.get("report_filename", "")
            except Exception:
                pass
        sessions.append(
            {
                "id": p.name,
                "name": name,
                "completed": completed,
                "report_generated": report_generated,
                "report_filename": report_filename,  # Add this field

            }
        )
    return sessions

# -------------------------------------------------------------
# ③  WebSocket – interview lives here (Used by Candidate UI)
# -------------------------------------------------------------
# main.py - Updated WebSocket handler
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    logger.info(f"WebSocket connection established for session: {session_id}")
    
    try:
        # Validate session exists
        meta_path = ROOT_DIR / "uploads" / session_id / "metadata.json"
        if not meta_path.is_file():
            logger.error(f"Session not found: {session_id}")
            await websocket.send_json({
                "type": "error",
                "content": "Session not found. Please create a new interview session."
            })
            await websocket.close(code=1008, reason="Session not found")
            return
        
        # Read metadata
        with open(meta_path, "r", encoding='utf-8') as f:
            meta = json.load(f)
        
        if meta.get("interview_completed", False):
            logger.info(f"Interview already completed for session: {session_id}")
            await websocket.send_json({
                "type": "error", 
                "content": "This interview has already been completed."
            })
            await websocket.close(code=1008, reason="Interview completed")
            return

        # Initialize interview bot
        candidate_name = meta.get("candidate_name", "Unknown")
        interview_duration = int(meta.get("interview_duration", 15))
        
        safe_name = _sanitize_name(candidate_name) or "default"
        candidate_dir = ROOT_DIR / "data" / safe_name
        jd_index_path = candidate_dir / "vector_store" / session_id / "faiss_jd_index"
        resume_index_path = candidate_dir / "vector_store" / session_id / "faiss_resume_index"

        if not jd_index_path.exists() or not resume_index_path.exists():
            logger.error(f"FAISS indexes not found for session: {session_id}")
            await websocket.send_json({
                "type": "error",
                "content": "Interview data not found. Please recreate the session."
            })
            await websocket.close(code=1011, reason="Data not found")
            return

        # Create bot instance
        bot = create_interview_bot(str(jd_index_path), str(resume_index_path), interview_duration)
        
        # Send initial configuration
        await websocket.send_json({
            "type": "config", 
            "duration": interview_duration,
            "session_id": session_id
        })
        
        # Start interview and send first message
        first_message = bot.start_interview()
        await websocket.send_json({
            "type": "message", 
            "role": "bot", 
            "content": first_message
        })

        # Main message loop
        while True:
            try:
                # Set a timeout for receiving messages
                data = await asyncio.wait_for(websocket.receive_text(), timeout=300.0)  # 5 minute timeout
                
                message_data = json.loads(data)
                logger.debug(f"Received message: {message_data}")
                
                # Handle different message types
                if message_data.get("type") == "time_up":
                    user_input = "TIME_UP_SIGNAL"
                elif message_data.get("type") == "ping":
                    # Respond to ping messages to keep connection alive
                    await websocket.send_json({"type": "pong"})
                    continue
                else:
                    user_input = message_data.get("content", "").strip()
                
                if not user_input and message_data.get("type") != "ping":
                    await websocket.send_json({
                        "type": "error",
                        "content": "Empty message received"
                    })
                    continue
                
                # Process user response
                bot_reply = bot.process_user_answer(user_input)
                
                if bot_reply == "END_OF_INTERVIEW":
                    # Save transcript
                    timestamp = int(time.time())
                    transcript_dir = candidate_dir / "transcripts"
                    transcript_dir.mkdir(parents=True, exist_ok=True)
                    transcript_path = transcript_dir / f"{session_id}_transcript.json"
                    bot.save_interview_log(str(transcript_path))
                    
                    # Update metadata
                    meta["interview_completed"] = True
                    meta["completed_at"] = timestamp
                    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding='utf-8')
                    
                    final_msg = "Thank you for your time. The interview is now complete. You may close this window."
                    await websocket.send_json({
                        "type": "message", 
                        "role": "system", 
                        "content": final_msg
                    })
                    
                    # Send completion signal
                    await websocket.send_json({
                        "type": "interview_complete",
                        "session_id": session_id
                    })
                    
                    logger.info(f"Interview completed for session: {session_id}")
                    break
                
                # Send bot response
                await websocket.send_json({
                    "type": "message", 
                    "role": "bot", 
                    "content": bot_reply
                })
                
            except asyncio.TimeoutError:
                logger.warning(f"WebSocket timeout for session: {session_id}")
                await websocket.send_json({
                    "type": "error",
                    "content": "Connection timeout. Please refresh the page."
                })
                break
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected normally for session: {session_id}")
                break
                
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received for session: {session_id}")
                await websocket.send_json({
                    "type": "error",
                    "content": "Invalid message format"
                })
                continue
                
            except Exception as e:
                logger.error(f"Error processing message for session {session_id}: {e}")
                await websocket.send_json({
                    "type": "error",
                    "content": f"Error processing message: {str(e)}"
                })
                continue
    
    except Exception as e:
        logger.error(f"Error in WebSocket handler for session {session_id}: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "content": "Internal server error. Please try again."
            })
        except:
            pass  # Connection might already be closed
    
    finally:
        try:
            await websocket.close()
        except:
            pass
        logger.info(f"WebSocket connection closed for session: {session_id}")

# -------------------------------------------------------------
# ④  Report generation endpoint (on‑demand by Manager UI)
# -------------------------------------------------------------
@app.post("/generate_report/{session_id}")
async def generate_report_endpoint(session_id: str):
    """
    Generates a markdown report for the most-recent transcript of the given session.
    """
    logger.info(f"🔍 Generating report for session: {session_id}")
    
    # Validate session exists
    meta_path = ROOT_DIR / "uploads" / session_id / "metadata.json"
    if not meta_path.is_file():
        logger.error(f"❌ Session metadata not found: {session_id}")
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        with open(meta_path, "r", encoding='utf-8') as f:
            meta = json.load(f)
        
        candidate_name = meta.get("candidate_name", "")
        report_generated = bool(meta.get("report_generated", False))
        existing_report_filename = meta.get("report_filename", "")

        safe_name = _sanitize_name(candidate_name) or "default"
        candidate_dir = ROOT_DIR / "data" / safe_name
        report_dir = candidate_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)

        # If report already exists and is valid, return it
        
        if report_generated and existing_report_filename:
            report_file_path = report_dir / existing_report_filename
            if report_file_path.is_file():
                download_url = f"/download/{session_id}/{existing_report_filename}"
                logger.info(f"✅ Report already exists: {report_file_path}")
                return JSONResponse({
                    "download_url": download_url,
                    "message": "Report already exists",
                    "filename": existing_report_filename
                })
            else:
                logger.warning(f"⚠️  Metadata says report exists but file is missing: {report_file_path}")
                meta["report_generated"] = False
                meta["report_filename"] = ""

        # Check if transcript exists
        transcript_dir = candidate_dir / "transcripts"
        if not transcript_dir.is_dir():
            raise HTTPException(status_code=404, detail="No transcripts found for this session")

        transcripts = sorted(transcript_dir.glob("*_transcript.json"))
        if not transcripts:
            raise HTTPException(status_code=404, detail="No transcript file found")
        
        latest_transcript = transcripts[-1]
        logger.info(f"📄 Found transcript: {latest_transcript.name}")

        # Check if FAISS indexes exist
        jd_index_path = candidate_dir / "vector_store" / session_id / "faiss_jd_index"
        resume_index_path = candidate_dir / "vector_store" / session_id / "faiss_resume_index"
        
        if not jd_index_path.exists() or not resume_index_path.exists():
            logger.error(f"❌ FAISS indexes not found for session: {session_id}")
            raise HTTPException(status_code=404, detail="FAISS indexes not found for this session")

        # Generate report filename
        report_filename = f"{safe_name}_{int(time.time())}_report.md"
        report_path = report_dir / report_filename

        try:
            logger.info(f"🛠️  Starting report generation for {session_id}")

            # Generate the report
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                create_report,
                str(jd_index_path),
                str(resume_index_path),
                str(latest_transcript),
                str(report_path)
            )
            if not report_path.exists():
                raise Exception("Report file was not created")
            logger.info(f"✅ Report generated successfully: {report_path}")

        except Exception as exc:
            logger.error(f"❌ Report generation failed for {session_id}: {exc}")
            # Clean up if report file was partially created
            if report_path.exists():
                report_path.unlink()
            raise HTTPException(status_code=500, detail=f"Report generation error: {str(exc)}")

        # Update metadata
        meta["report_generated"] = True
        meta["report_filename"] = report_filename
        meta["report_generated_at"] = int(time.time())
        
        with open(meta_path, "w", encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📝 Metadata updated for session: {session_id}")
        logger.info(f"📊 Metadata content: {json.dumps(meta, indent=2)}")

        download_url = f"/download/{session_id}/{report_filename}"
        return JSONResponse({
            "download_url": download_url,
            "message": "Report generated successfully",
            "filename": report_filename
        })

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"❌ Unexpected error in generate_report: {exc}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(exc)}")


# -------------------------------------------------------------
# ⑤  Download endpoint – serves reports, transcripts, etc.
# -------------------------------------------------------------
@app.get("/download/{session_id}/{filename}")
async def download_file(session_id: str, filename: str):
    """
    Looks in the candidate‑specific ``data/<candidate_name>/`` folder
    for the requested file.
    """
    meta_path = ROOT_DIR / "uploads" / session_id / "metadata.json"
    if not meta_path.is_file():
        raise HTTPException(status_code=404, detail="Metadata not found")

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        candidate_name = meta.get("candidate_name", "")
    except Exception:
        raise HTTPException(status_code=500, detail="Corrupt metadata")

    safe_name = _sanitize_name(candidate_name)
    if not safe_name:
        safe_name = "default"
    base_folder = ROOT_DIR / "data" / safe_name

    report_file = base_folder / "reports" / filename
    if report_file.is_file():
        return FileResponse(
            str(report_file),
            filename=filename,
            media_type="text/markdown"
        )

    logger.warning(f"File not found for download: {report_file}")
    raise HTTPException(status_code=404, detail="File not found")

# -------------------------------------------------------------
# ⑥  NEW: Delete Session endpoint
# -------------------------------------------------------------
@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    Deletes a session.
    This removes the 'uploads/<session_id>' folder.
    It also removes the associated 'data/<safe_name>/vector_store/<session_id>' folder.
    It does NOT delete transcripts or reports, as they might be shared.
    """
    logger.info(f"Attempting to delete session: {session_id}")

    upload_dir = ROOT_DIR / "uploads" / session_id
    meta_path = upload_dir / "metadata.json"

    if not upload_dir.is_dir() or not meta_path.is_file():
        logger.error(f"Delete failed: Directory or metadata not found at {upload_dir}")
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        candidate_name = meta.get("candidate_name", "")
        safe_name = _sanitize_name(candidate_name)
        if not safe_name:
            safe_name = "default"

        candidate_dir = ROOT_DIR / "data" / safe_name
        session_vector_store = candidate_dir / "vector_store" / session_id

        shutil.rmtree(upload_dir, ignore_errors=True)

        if session_vector_store.exists():
            shutil.rmtree(session_vector_store, ignore_errors=True)

        logger.info(f"🗑️ Deleted session {session_id} and its vector store.")

        return JSONResponse(
            status_code=200,
            content={"message": f"Session {session_id} deleted successfully."}
        )

    except Exception as exc:
        logger.exception(f"Error during session deletion for {session_id}: {exc}")
        raise HTTPException(status_code=500, detail=f"Error deleting session: {exc}")