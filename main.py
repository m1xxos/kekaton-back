import os
import json
import uuid
import datetime
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
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

# Словарь для хранения ответов
responses_storage = {}

# Настройка шаблонов
templates = Jinja2Templates(directory="templates")

# Настройка статических файлов (CSS, шрифты)
app.mount("/static", StaticFiles(directory="static"), name="static")

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
        
        # Генерируем уникальный ID для ответа
        response_id = str(uuid.uuid4())
        
        # Форматируем и сохраняем ответ
        response_data = {
            "model": model,
            "response": response.get("response"),
            "prompt": prompt,
            "done": True,
            "total_duration": response.get("total_duration"),
            "created_at": datetime.datetime.now().isoformat()
        }
        
        # Сохраняем ответ в хранилище
        responses_storage[response_id] = response_data
        
        # Возвращаем и URL для просмотра ответа, и оригинальные данные
        return {
            "response_id": response_id,
            "url": f"/minecraft-view/{response_id}",
            "preview": response_data["response"][:100] + "..." if len(response_data["response"]) > 100 else response_data["response"],
            # Оригинальные данные для обратной совместимости
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
        
        # Генерируем уникальный ID для ответа
        response_id = str(uuid.uuid4())
        
        # Подготавливаем данные для сохранения
        response_content = response.get("message", {}).get("content", "")
        
        # Форматируем ответ для сохранения
        response_data = {
            "model": model,
            "response": response_content,
            "prompt": prompt,
            "done": True,
            "created_at": datetime.datetime.now().isoformat()
        }
        
        # Сохраняем ответ в хранилище
        responses_storage[response_id] = response_data
        
        # Добавляем информацию о URL для просмотра ответа
        response["minecraft_view"] = {
            "response_id": response_id,
            "url": f"/minecraft-view/{response_id}"
        }
        
        return response
        
    except Exception as e:
        return ErrorResponse(
            error=f"Ошибка при обработке чата: {str(e)}",
            status_code=500
        )
    
@app.get("/minecraft-view/{response_id}", response_class=HTMLResponse)
def minecraft_view(request: Request, response_id: str):
    """Отобразить ответ на странице в стиле Minecraft."""
    if response_id not in responses_storage:
        return templates.TemplateResponse(
            "error.html", 
            {"request": request, "error": "Ответ не найден"}
        )
    
    response_data = responses_storage[response_id]
    return templates.TemplateResponse(
        "minecraft.html", 
        {
            "request": request, 
            "response": response_data["response"], 
            "prompt": response_data["prompt"],
            "model": response_data["model"],
            "total_duration": response_data.get("total_duration", 0) / 1000000 if response_data.get("total_duration") else 0,  # Конвертируем в секунды
            "title": "Стив из Майнкрафта"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=True)
