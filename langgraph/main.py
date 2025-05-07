import uvicorn
from fastapi import FastAPI, Depends, HTTPException,Request
from fastapi.security import HTTPBearer
from config import Settings
from typing import List
from src.langgraph_service import generate_response,_init_graph
from contextlib import asynccontextmanager
from models import ChatRequest,ChatResponse

settings = Settings()
settings.configure_logging()

graph_app=None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. On startup, initialize the LangGraph workflow
    app.state.graph_app = await _init_graph()
    yield
    # 2. (Optional) On shutdown, you could close resources:
    # await app.state.graph_app.close()  # if you implement a shutdown hook

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