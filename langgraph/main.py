from psycopg_pool import AsyncConnectionPool
import uvicorn
from fastapi import FastAPI, Depends, HTTPException,Request
from fastapi.security import HTTPBearer
from config import Settings
from typing import List
from src.langgraph_service import generate_response,_init_graph
from src.tools import get_calendar_service
from contextlib import asynccontextmanager
from models import ChatRequest,ChatResponse
from src.supabase_service import SupabaseService
from supabase import create_client
import logging
import asyncio


settings = Settings()
settings.configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):

    connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
    }     

    # Build a shared Postgres connection pool
    app.state.pg_pool = AsyncConnectionPool(
        conninfo=settings.DB_URI,
        min_size=2,       # keep a few idle connections
        open=False,
        max_size=20,      # cap total concurrent clients
        kwargs=connection_kwargs
    )
    await app.state.pg_pool.open()
    
    # Initialize Google Calendar service 
    async def _init_calendar():
        return await asyncio.to_thread(get_calendar_service)
    try:
        app.state.calendar_service = await asyncio.wait_for(_init_calendar(), timeout=30)
        logging.info("Calendar service initialized")
    except asyncio.TimeoutError:
        logging.error("Calendar init timed out after 30s; skipping calendar")
        app.state.calendar_service = None
    except Exception:
        logging.exception("Calendar service init failed; continuing without calendar")
        app.state.calendar_service = None

    # Compile LangGraph using this pool
    app.state.graph_app = await _init_graph(pool=app.state.pg_pool)
    yield


app = FastAPI(lifespan=lifespan)
@app.post("/chat", response_model=ChatResponse)
async def chat(request : ChatRequest) -> ChatResponse:
    """
    wa_id: unique thread identifier (e.g. WhatsApp sender)
    message: the user input or conversation history
    """
    workflow_app = app.state.graph_app
    messages = await generate_response(request.message, request.wa_id, workflow_app)
    return ChatResponse(reply=messages)

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)