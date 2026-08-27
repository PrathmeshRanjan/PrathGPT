# ===========================================================================
# PrathGPT Backend Server (FastAPI)
# ===========================================================================
# This file serves as the main entry point and backend server for PrathGPT.
# It connects the Frontend (HTML/JS) with:
# 1. LangGraph Agent (agent.py) for conversational reasoning & tool execution.
# 2. SQLite Database (database.py) for conversation history & session lists.
# 3. ChromaDB / RAG Pipeline (rag.py) for document parsing & vector search.
# 4. Agent Tools (tools.py) for thread-aware tool execution.
# ===========================================================================

from dotenv import load_dotenv
load_dotenv()

import json
import uuid
from pathlib import Path
import uvicorn
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    AIMessageChunk,
    ToolMessage
)

from agent import get_agent, DEFAULT_MODEL, ALLOWED_MODELS
from database import (
    init_db,
    save_chat_message,
    get_chat_history,
    create_or_update_conversation,
    list_conversations
)
from rag import add_document_to_rag
from tools import set_current_thread_id

# ---------------------------------------------------------------------------
# 1. FastAPI App & Template Configuration
# ---------------------------------------------------------------------------
app = FastAPI(title="PrathGPT API", version="1.0.0")

@app.get("/models")
async def get_models():
    """
    HTTP GET /models
    Returns the list of supported Mistral models and default model selection.
    """
    return {
        "models": sorted(list(ALLOWED_MODELS)),
        "default": DEFAULT_MODEL
    }

# Jinja2Templates allows rendering HTML files located in the 'template' directory.
# (If your template directory is named 'template', we point directly to it).
template_dir = "template" if Path("template").exists() else "templates"
templates = Jinja2Templates(directory=template_dir)

# Ensure required persistent storage directories exist on server startup
Path("uploads").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

# Initialize SQLite database tables (conversations, messages, memories)
init_db()


# ---------------------------------------------------------------------------
# 2. Frontend Web Routes (HTML Serving)
# ---------------------------------------------------------------------------

@app.get("/")
async def home(request: Request):
    """
    HTTP GET /
    Serves the main single-page application (index.html) to the browser.
    """
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# ---------------------------------------------------------------------------
# 3. Conversation & History REST API Endpoints
# ---------------------------------------------------------------------------

@app.get("/conversations")
async def conversations():
    """
    HTTP GET /conversations
    Returns a JSON list of all previous conversations for the sidebar.
    Sorted by most recently active first.
    """
    items = list_conversations()

    return {
        "conversations": [
            {
                "thread_id": item.thread_id,
                "title": item.title,
                "created_at": item.created_at.isoformat(),
                "updated_at": item.updated_at.isoformat()
            }
            for item in items
        ]
    }


@app.get("/history/{thread_id}")
async def history(thread_id: str):
    """
    HTTP GET /history/{thread_id}
    Retrieves all past messages (chronological order) for a specific chat thread.
    Used to repopulate the chat when a user clicks a conversation in the sidebar.
    """
    messages = get_chat_history(thread_id)

    return {
        "messages": [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in messages
        ]
    }


# ---------------------------------------------------------------------------
# 4. Document Ingestion Endpoint (RAG Pipeline)
# ---------------------------------------------------------------------------

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    thread_id: str = Form(...)
):
    """
    HTTP POST /upload (Multipart form-data)
    Receives an uploaded document from the client:
    1. Validates the file extension (.pdf, .docx, .txt, .md, .py, .csv).
    2. Saves the file with a unique UUID to the 'uploads/' folder.
    3. Calls 'add_document_to_rag()' to split text & store vectors in ChromaDB.
    4. Attaches the current thread_id so search results stay isolated to this chat.
    """
    try:
        allowed_extensions = [".pdf", ".docx", ".txt", ".md", ".py", ".csv"]

        filename = file.filename or "uploaded_file"
        suffix = Path(filename).suffix.lower()

        if suffix not in allowed_extensions:
            return JSONResponse(
                {
                    "success": False,
                    "message": "Unsupported file type. Upload PDF, DOCX, TXT, MD, PY, or CSV."
                },
                status_code=400
            )

        # Generate a unique path to prevent filename collisions
        file_id = str(uuid.uuid4())
        safe_filename = filename.replace(" ", "_")
        file_path = f"uploads/{file_id}_{safe_filename}"

        # Write uploaded binary content to disk
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Update or create conversation entry in database
        create_or_update_conversation(thread_id, "Uploaded document")

        # Ingest document into Chroma vector store
        result = add_document_to_rag(
            file_path=file_path,
            thread_id=thread_id
        )

        return JSONResponse({
            "success": True,
            "message": f"Uploaded {result['filename']} and created {result['chunks']} chunks."
        })

    except Exception as e:
        return JSONResponse(
            {
                "success": False,
                "message": str(e)
            },
            status_code=500
        )


# ---------------------------------------------------------------------------
# 5. Helper Functions for Streaming & Chunk Filtering
# ---------------------------------------------------------------------------

def sse_data(payload: dict) -> str:
    """Formats a Python dictionary into standard Server-Sent Events (SSE) data format: 'data: {...}\n\n'"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def should_stream_chunk(chunk, metadata) -> bool:
    """
    Filters out raw internal tool data and intermediate JSON payloads from being
    streamed to the user's chat bubble.
    
    Returns True ONLY for final text generated by the AI model (AIMessage / AIMessageChunk).
    """
    metadata = metadata or {}
    node_name = str(metadata.get("langgraph_node", "")).lower()

    # Do not stream internal tool execution nodes
    if "tool" in node_name:
        return False

    # Do not stream raw ToolMessage outputs
    if isinstance(chunk, ToolMessage):
        return False

    # Only stream AI model messages
    if not isinstance(chunk, (AIMessage, AIMessageChunk)):
        return False

    # Do not stream tool calling request objects/metadata
    if getattr(chunk, "tool_calls", None):
        return False

    if getattr(chunk, "invalid_tool_calls", None):
        return False

    additional_kwargs = getattr(chunk, "additional_kwargs", {}) or {}
    if additional_kwargs.get("tool_calls"):
        return False

    return True


def extract_text_from_chunk(chunk) -> str:
    """
    Safely extracts plain text strings from various message chunk structures
    (string content, multimodal lists, or dict blocks).
    """
    content = getattr(chunk, "content", "")

    if not content:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
                elif isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    text_parts.append(item["content"])

        return "".join(text_parts)

    return ""


# ---------------------------------------------------------------------------
# 6. Real-Time Streaming Chat Endpoint (Server-Sent Events)
# ---------------------------------------------------------------------------

@app.post("/chat/stream")
async def chat_stream(request: Request):
    """
    HTTP POST /chat/stream
    Core chat endpoint that:
    1. Parses user prompt, thread_id, and chosen model from request body.
    2. Persists the user message to SQLite.
    3. Sets active thread_id for tool context (RAG & memory).
    4. Invokes the LangGraph Agent via stream_mode='messages'.
    5. Yields tokens in real-time as Server-Sent Events (SSE) to the browser.
    6. Saves the completed AI response to SQLite once streaming finishes.
    """
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(
            {"error": "Invalid JSON body."},
            status_code=400
        )

    user_message = data.get("message", "")
    thread_id = data.get("thread_id", "default")
    selected_model = data.get("model", DEFAULT_MODEL)

    if not user_message.strip():
        return JSONResponse(
            {"error": "Message is required."},
            status_code=400
        )

    # Fetch cached or newly built LangGraph agent for the chosen model
    agent = get_agent(selected_model)

    # Record user message in DB
    create_or_update_conversation(thread_id, user_message)
    save_chat_message(thread_id, "user", user_message)

    # Set thread context for tools (used by search_uploaded_documents & memory tools)
    set_current_thread_id(thread_id)

    # Thread config for LangGraph checkpoint persistence
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    def event_generator():
        """Generator function that streams message tokens using Server-Sent Events."""
        final_answer = ""

        try:
            inputs = {
                "messages": [
                    HumanMessage(content=user_message)
                ]
            }

            # Stream message chunks from the LangGraph workflow
            for chunk, metadata in agent.stream(
                inputs,
                config=config,
                stream_mode="messages"
            ):
                # Ignore tool messages and raw tool call dictionaries
                if not should_stream_chunk(chunk, metadata):
                    continue

                token = extract_text_from_chunk(chunk)

                if token:
                    final_answer += token
                    yield sse_data({"token": token})

            # Save the full assistant response to the database after generation completes
            if final_answer.strip():
                save_chat_message(thread_id, "assistant", final_answer)

            # Signal the frontend that streaming has successfully finished
            yield sse_data({"done": True})

        except Exception as e:
            yield sse_data({"error": str(e)})
            yield sse_data({"done": True})

    # Return HTTP 200 StreamingResponse with event-stream media type
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ---------------------------------------------------------------------------
# 7. Application Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Uvicorn is the ASGI web server that runs FastAPI apps
    # host="0.0.0.0" makes the server accessible across the local network
    # port=8080 is the listening port
    # reload=True automatically restarts the server when code changes are saved
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        reload=True
    )