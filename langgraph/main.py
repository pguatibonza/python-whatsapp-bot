import uvicorn
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer
from config import Settings
from typing import List
from src.langgraph_service import generate_response,_init_graph

settings = Settings()
app = FastAPI()

graph_app=None

@app.on_event("startup")
async def startup_event():
    global graph_app
    # initialize the state-graph once
    graph_app = await _init_graph()

@app.post("/chat", )
async def chat(wa_id: str, message: str) -> List[str]:
    """
    wa_id: unique thread identifier (e.g. WhatsApp sender)
    message: the user input or conversation history
    """
    return await generate_response(message, wa_id,graph_app)

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)