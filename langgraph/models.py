
from pydantic import BaseModel

class ChatRequest(BaseModel):
    wa_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str
