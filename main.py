import os
import json
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from models import OllamaRequest, OllamaResponse, ErrorResponse, MsgPayload

# Импортируем официальную библиотеку ollama
import ollama

# Load environment variables
load_dotenv()

# Системное сообщение по умолчанию
DEFAULT_SYSTEM_MESSAGE = "Ты стив из минекруфта, отвечай на все вопросы с помощью minecraft терминов"

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
# Создаем клиент Ollama
client = ollama.Client(host=OLLAMA_BASE_URL)

messages_list: dict[int, MsgPayload] = {}


@app.get("/")
def root():
    return {"message": "Ollama API Proxy is running"}


@app.get("/about")
def about() -> dict[str, str]:
    return {"message": "This is the about page."}


@app.get("/health")
def health_check():
    """Check if Ollama is running and accessible."""
    try:
        # Получаем информацию о списке моделей для проверки подключения
        models = client.list()
        return {
            "status": "ok", 
            "ollama_status": "running",
            "models_count": len(models.get("models", [])),
            "details": "Ollama API is properly configured and running"
        }
    except Exception as e:
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
def list_models():
    """Получить список доступных моделей из Ollama."""
    try:
        # Используем клиент для получения списка моделей
        result = client.list()
        
        # Форматируем ответ
        if "models" in result:
            model_names = [model.get("name") for model in result.get("models", [])]
            return {
                "status": "ok",
                "models_count": len(model_names),
                "models": model_names
            }
        else:
            return {
                "status": "warning",
                "message": "Unexpected response format from Ollama",
                "raw_data": result
            }
            
    except Exception as e:
        return {
            "status": "error",
            "message": f"Exception during API call: {str(e)}"
        }


@app.post("/generate")
def generate(request: OllamaRequest):
    """Сгенерировать ответ от модели Ollama."""
    try:
        # Извлекаем параметры из запроса
        model = request.model
        prompt = request.prompt
        system = request.system or DEFAULT_SYSTEM_MESSAGE
        options = request.options or {}
        
        # Используем клиент для генерации ответа
        response = client.generate(
            model=model,
            prompt=prompt,
            system=system,
            options=options
        )
        
        # Форматируем ответ
        return {
            "model": response.get("model"),
            "response": response.get("response"),
            "done": True,
            "total_duration": response.get("total_duration"),
            "load_duration": response.get("load_duration"),
            "prompt_eval_count": response.get("prompt_eval_count"),
            "prompt_eval_duration": response.get("prompt_eval_duration"),
            "eval_count": response.get("eval_count"),
            "eval_duration": response.get("eval_duration")
        }
        
    except Exception as e:
        return ErrorResponse(
            error=f"Ошибка при генерации ответа: {str(e)}",
            status_code=500
        )


@app.post("/chat")
def chat(request: OllamaRequest):
    """Отправить запрос в чат Ollama."""
    try:
        # Извлекаем параметры из запроса
        model = request.model
        prompt = request.prompt
        # Используем заданное системное сообщение или наше по умолчанию
        system = request.system or DEFAULT_SYSTEM_MESSAGE
        options = request.options or {}
        
        messages = []
        
        # Добавляем системное сообщение
        messages.append({
            "role": "system",
            "content": system
        })
        
        # Добавляем основное сообщение пользователя
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Используем клиент для генерации ответа в формате чата
        response = client.chat(
            model=model,
            messages=messages,
            options=options
        )
        
        return response
        
    except Exception as e:
        return ErrorResponse(
            error=f"Ошибка при обработке чата: {str(e)}",
            status_code=500
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=True)
