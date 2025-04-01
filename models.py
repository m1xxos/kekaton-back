from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field


class MsgPayload(BaseModel):
    msg_id: Optional[int]
    msg_name: str


class OllamaRequest(BaseModel):
    model: str
    prompt: str
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None
    options: Optional[Dict[str, Any]] = None
    format: Optional[str] = None
    stream: Optional[bool] = False
    raw: Optional[bool] = False
    keep_alive: Optional[Union[str, int]] = None


class OllamaResponse(BaseModel):
    model: str
    created_at: str
    response: str
    done: bool
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


class ErrorResponse(BaseModel):
    error: str
    status_code: int = Field(default=500)
