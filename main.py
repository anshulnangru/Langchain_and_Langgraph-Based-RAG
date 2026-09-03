import logfire
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
import json
from fastapi import BackgroundTasks
import asyncio
load_dotenv()
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))

# Now safe to import app modules - logfire is already active
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from src.agents.graph import rag_agent
from src.guardrails import initialize_rails, guard

from pydantic import BaseModel
from typing import Optional


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once, before the app starts accepting requests
    initialize_rails()
    logfire.info(" App startup complete — guardrails initialised.")
    yield
    # (optional) any shutdown/cleanup logic goes here


app = FastAPI(title="Enterprise Agentic RAG API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    q: str
    thread_id: Optional[str] = "default_user"


@app.get("/")
def home():
    return {"message": "Enterprise LangGraph RAG API is live."}

@app.post("/stream")
async def stream_query(request: QueryRequest):
    q = request.q
    thread_id = request.thread_id

    async def event_generator():
        try:
            loop = asyncio.get_event_loop()

            # Run guard() in a thread pool — it's sync/blocking
            rail_fired, rail_response = await loop.run_in_executor(
                None, guard, q
            )

            if rail_fired:
                logfire.info(f"🛡️ Request blocked by guardrails | thread={thread_id}")
                yield f"data: {json.dumps({'token': rail_response})}\n\n"
                yield "data: [DONE]\n\n"
                return

            initial_state = {
                "messages": [{"role": "user", "content": q}],
                "current_query": q,
                "documents": [],
                "plan": ["Start"],
                "status": "Initializing Graph..."
            }
            config = {"configurable": {"thread_id": thread_id}}

            # Run rag_agent.invoke() in a thread pool — also sync/blocking
            final_output = await loop.run_in_executor(
                None,
                lambda: rag_agent.invoke(initial_state, config=config)
            )

            answer = final_output.get("final_answer", "") or ""

            for word in answer.split(" "):
                yield f"data: {json.dumps({'token': word + ' '})}\n\n"
                await asyncio.sleep(0.02)  # small delay makes the typing effect visible

            yield "data: [DONE]\n\n"

        except Exception as e:
            logfire.error(f"❌ Stream Execution Failed: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/graph")
def get_graph_image():
    """
    Returns the Mermaid image of the agent's workflow.
    """
    try:
        png_bytes = rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        return {"error": f"Could not generate graph image: {e}"}


@app.post("/query")
def query(request: QueryRequest):
    """
    Executes the LangGraph RAG flow with memory using a POST request.
    """
    q = request.q
    thread_id = request.thread_id

    initial_state = {
        "messages": [{"role": "user", "content": q}],
        "current_query": q,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph..."
    }

    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Gate 1: NeMo Guardrails — blocks off-topic, jailbreaks, and handles dialog
        rail_fired, rail_response = guard(q)
        if rail_fired:
            logfire.info(f"🛡️ Request blocked by guardrails | thread={thread_id}")
            return {
                "question": q,
                "answer": rail_response,
                "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
                "status": "Blocked by guardrails.",
                "sources": []
            }

        # Gate 2: LangGraph RAG pipeline
        final_output = rag_agent.invoke(initial_state, config=config)

        return {
            "question": q,
            "answer": final_output.get("final_answer"),
            "thought_process": final_output.get("plan"),
            "status": final_output.get("status"),
            "sources": final_output.get("documents", [])
        }
    except Exception as e:
        logfire.error(f"❌ Backend Execution Failed: {e}")
        return {
            "question": q,
            "answer": "I apologize, but I encountered an internal error while processing your request. Please try again later.",
            "thought_process": ["Error encountered during execution."],
            "status": "error",
            "sources": []
        }