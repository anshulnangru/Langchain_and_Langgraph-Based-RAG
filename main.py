import asyncio
import hmac
import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import logfire
from dotenv import load_dotenv

load_dotenv()
logfire.configure(send_to_logfire="if-token-present")

# Now safe to import app modules - logfire is already active
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.responses import JSONResponse

from src.agents.graph import rag_agent
from src.guardrails import guard, initialize_rails

APP_API_KEY = os.getenv("APP_API_KEY")
RATE_LIMIT = os.getenv("RATE_LIMIT", "10/minute")
EXTRA_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]


def client_key(request: Request) -> str:
    # Behind a host proxy the real client IP is in X-Forwarded-For
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else get_remote_address(request)


limiter = Limiter(key_func=client_key)


def require_key(x_api_key: str = Header(default="")):
    if not hmac.compare_digest(x_api_key, APP_API_KEY or ""):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail loudly at startup instead of serving wrong answers later.
    if not APP_API_KEY:
        raise RuntimeError("APP_API_KEY is not set. Refusing to start an open endpoint.")

    from src.retrieval.embeddings import embed_query

    dim = len(embed_query("preflight"))
    if dim != 1024:
        raise RuntimeError(f"Embedding dim {dim} != 1024. Check the Jina key and balance.")

    initialize_rails()
    logfire.info("App startup complete - guardrails initialised.")
    yield


app = FastAPI(title="Enterprise Agentic RAG API", lifespan=lifespan)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limited(request: Request, exc: RateLimitExceeded):
    return JSONResponse({"detail": "Rate limit exceeded. Try again shortly."}, status_code=429)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", *EXTRA_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    q: str
    thread_id: Optional[str] = None


def _thread(tid: Optional[str]) -> str:
    # Never share one memory thread between anonymous callers
    return tid if tid and tid != "default_user" else str(uuid.uuid4())


def _initial_state(q: str) -> dict:
    return {
        "messages": [{"role": "user", "content": q}],
        "current_query": q,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph...",
    }


@app.get("/")
def home():
    return {"message": "Enterprise LangGraph RAG API is live."}


@app.get("/health")
def health():
    # Cheap on purpose: hosts ping this often, and it must not spend tokens.
    return {"status": "ok"}


@app.post("/stream", dependencies=[Depends(require_key)])
@limiter.limit(RATE_LIMIT)
async def stream_query(request: Request, body: QueryRequest):
    q = body.q
    thread_id = _thread(body.thread_id)

    async def event_generator():
        try:
            loop = asyncio.get_event_loop()
            config = {"configurable": {"thread_id": thread_id}}

            # Run guardrails and LangGraph concurrently
            guard_task = loop.run_in_executor(None, guard, q)
            graph_task = loop.run_in_executor(
                None, lambda: rag_agent.invoke(_initial_state(q), config=config)
            )

            rail_fired, rail_response = await guard_task
            if rail_fired:
                graph_task.cancel()
                logfire.info(f"Request blocked by guardrails | thread={thread_id}")
                yield f"data: {json.dumps({'token': rail_response})}\n\n"
                yield "data: [DONE]\n\n"
                return

            final_output = await graph_task
            answer = final_output.get("final_answer", "") or ""
            for word in answer.split(" "):
                yield f"data: {json.dumps({'token': word + ' '})}\n\n"
                await asyncio.sleep(0.02)

            yield "data: [DONE]\n\n"

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logfire.error(f"Stream execution failed: {type(e).__name__}: {str(e)[:200]}")
            # Generic message: do not leak internal error text to callers
            yield f"data: {json.dumps({'error': 'Something went wrong. Please try again.'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/graph", dependencies=[Depends(require_key)])
def get_graph_image():
    """Returns the Mermaid image of the agent's workflow."""
    try:
        png_bytes = rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        logfire.error(f"Graph image failed: {e}")
        return {"error": "Could not generate graph image."}


@app.post("/query", dependencies=[Depends(require_key)])
@limiter.limit(RATE_LIMIT)
def query(request: Request, body: QueryRequest):
    """Executes the LangGraph RAG flow with memory using a POST request."""
    q = body.q
    thread_id = _thread(body.thread_id)
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Gate 1: NeMo Guardrails: blocks off-topic, jailbreaks, handles dialog
        rail_fired, rail_response = guard(q)
        if rail_fired:
            logfire.info(f"Request blocked by guardrails | thread={thread_id}")
            return {
                "question": q,
                "answer": rail_response,
                "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
                "status": "Blocked by guardrails.",
                "sources": [],
                "thread_id": thread_id,
            }

        # Gate 2: LangGraph RAG pipeline
        final_output = rag_agent.invoke(_initial_state(q), config=config)

        return {
            "question": q,
            "answer": final_output.get("final_answer"),
            "thought_process": final_output.get("plan"),
            "status": final_output.get("status"),
            "sources": final_output.get("documents", []),
            "thread_id": thread_id,
        }
    except Exception as e:
        logfire.error(f"Backend execution failed: {type(e).__name__}: {str(e)[:200]}")
        return {
            "question": q,
            "answer": "I apologize, but I encountered an internal error while processing your request. Please try again later.",
            "thought_process": ["Error encountered during execution."],
            "status": "error",
            "sources": [],
            "thread_id": thread_id,
        }