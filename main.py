import os
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import httpx
import json
from dotenv import load_dotenv
from models import OllamaRequest, OllamaResponse, ErrorResponse, MsgPayload

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Ollama API Proxy",
    description="API для проксирования запросов к Ollama LLM",
    version="0.1.0"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Можно ограничить конкретными источниками в production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Default Ollama URL
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

messages_list: dict[int, MsgPayload] = {}


@app.get("/")
async def root():
    return {"message": "Ollama API Proxy is running"}


@app.get("/about")
def about() -> dict[str, str]:
    return {"message": "This is the about page."}


@app.get("/health")
async def health_check():
    """Check if Ollama is running and accessible."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}")
            return {
                "status": "ok", 
                "ollama_status": "running" if response.status_code < 400 else "error",
                "details": f"Status code: {response.status_code}"
            }
    except httpx.RequestError as e:
        return {
            "status": "error",
            "ollama_status": "not running or not accessible",
            "details": str(e)
        }


# Route to add a message
@app.post("/messages/{msg_name}/")
def add_msg(msg_name: str) -> dict[str, MsgPayload]:
    # Generate an ID for the item based on the highest ID in the messages_list
    msg_id = max(messages_list.keys()) + 1 if messages_list else 0
    messages_list[msg_id] = MsgPayload(msg_id=msg_id, msg_name=msg_name)

    return {"message": messages_list[msg_id]}


# Route to list all messages
@app.get("/messages")
def message_items() -> dict[str, dict[int, MsgPayload]]:
    return {"messages": messages_list}


@app.get("/models")
async def list_models():
    """Получить список доступных моделей из Ollama."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            
            # Check for valid JSON response
            try:
                return response.json()
            except json.JSONDecodeError:
                return {
                    "error": "Invalid JSON response from Ollama",
                    "response_text": response.text[:200],  # First 200 chars of response for debugging
                    "status_code": response.status_code
                }
                
    except httpx.RequestError as e:
        return {
            "error": f"Ошибка соединения с Ollama: {str(e)}",
            "status_code": 503,
            "details": "Make sure Ollama is running on your machine"
        }


@app.post("/generate")
async def generate(request: OllamaRequest):
    """Сгенерировать ответ от модели Ollama."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=request.model_dump(exclude_none=True)
            )
            
            try:
                json_response = response.json()
                
                if response.status_code != 200:
                    return ErrorResponse(
                        error=f"Ollama вернул ошибку: {json_response.get('error', response.text)}",
                        status_code=response.status_code
                    )
                
                return OllamaResponse(**json_response)
            except json.JSONDecodeError:
                return ErrorResponse(
                    error=f"Ollama вернул некорректный JSON. Response: {response.text[:200]}",
                    status_code=500
                )
                
    except httpx.RequestError as e:
        return ErrorResponse(
            error=f"Ошибка соединения с Ollama: {str(e)}",
            status_code=503
        )


@app.post("/chat")
async def chat(request: OllamaRequest):
    """Отправить запрос в чат Ollama."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=request.model_dump(exclude_none=True)
            )
            
            try:
                json_response = response.json()
                
                if response.status_code != 200:
                    return ErrorResponse(
                        error=f"Ollama вернул ошибку: {json_response.get('error', response.text)}",
                        status_code=response.status_code
                    )
                    
                return json_response
            except json.JSONDecodeError:
                return ErrorResponse(
                    error=f"Ollama вернул некорректный JSON. Response: {response.text[:200]}",
                    status_code=500
                )
                
    except httpx.RequestError as e:
        return ErrorResponse(
            error=f"Ошибка соединения с Ollama: {str(e)}",
            status_code=503
        )


@app.post("/stream")
async def stream(request: OllamaRequest, response: Response):
    """Потоковая генерация от Ollama."""
    if not request.stream:
        request.stream = True  # Принудительно включаем stream для этого эндпоинта

    async def stream_response():
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json=request.model_dump(exclude_none=True),
                    timeout=60.0
                ) as r:
                    async for chunk in r.aiter_text():
                        yield f"data: {chunk}\n\n"
        except httpx.RequestError as e:
            error_json = ErrorResponse(
                error=f"Ошибка соединения с Ollama: {str(e)}",
                status_code=503
            ).model_dump_json()
            yield f"data: {error_json}\n\n"
    
    response.headers["Content-Type"] = "text/event-stream"
    return stream_response()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=True)
