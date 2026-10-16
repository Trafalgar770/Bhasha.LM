
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import requests
import threading

app = FastAPI(title="BhashaLM")

BASE_DIR = Path(__file__).resolve().parent

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR),
    name="static"
)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b"

# Conversation memory for the current server session.
conversations = {}
conversation_lock = threading.Lock()

MAX_HISTORY_MESSAGES = 20

SUPPORTED_LANGUAGES = {
    "auto": "Automatically detect the user's language",
    "english": "English",
    "telugu": "Telugu (తెలుగు)",
    "hindi": "Hindi (हिन्दी)",
    "tamil": "Tamil (தமிழ்)",
    "kannada": "Kannada (ಕನ್ನಡ)",
    "malayalam": "Malayalam (മലയാളം)",
    "marathi": "Marathi (मराठी)",
    "bengali": "Bengali (বাংলা)",
    "gujarati": "Gujarati (ગુજરાતી)",
    "punjabi": "Punjabi (ਪੰਜਾਬੀ)",
    "odia": "Odia (ଓଡ଼ିଆ)",
}


class ChatRequest(BaseModel):
    message: str
    conversation_id: str = "default"
    language: str = "auto"


@app.get("/", response_class=HTMLResponse)
def home():
    index_file = BASE_DIR / "index.html"

    if not index_file.exists():
        return HTMLResponse(
            "<h1>BhashaLM</h1><p>index.html was not found.</p>",
            status_code=404
        )

    return HTMLResponse(
        index_file.read_text(encoding="utf-8")
    )


@app.get("/health")
def health():
    return {
        "status": "BhashaLM backend is running",
        "model": MODEL_NAME,
        "languages": list(SUPPORTED_LANGUAGES.keys())
    }


@app.get("/languages")
def languages():
    return {"languages": SUPPORTED_LANGUAGES}


@app.post("/chat")
def chat(request: ChatRequest):
    message = request.message.strip()
    conversation_id = request.conversation_id.strip()
    language = request.language.strip().lower()

    if not message:
        return {"reply": "Please enter a message."}

    if not conversation_id:
        return {"reply": "Please start a new conversation."}

    if language not in SUPPORTED_LANGUAGES:
        language = "auto"

    language_instruction = SUPPORTED_LANGUAGES[language]

    system_prompt = f"""
You are BhashaLM, a helpful multilingual AI assistant.

LANGUAGE INSTRUCTIONS:
- Language preference: {language_instruction}.
- If the preference is automatic, detect the language of the
  latest user message and answer in that language.
- If the user explicitly requests another language, follow
  that request.
- Support Telugu, Hindi, English, and other Indian languages
  to the best of your abilities.
- When replying in Telugu, use natural Telugu and తెలుగు script.
- When replying in Hindi, use natural Hindi and देवनागरी script.
- If the user writes a language using English letters, you may
  reply in its native script unless the user requests transliteration.
- For mixed-language messages, respond naturally in the user's
  preferred language.
- Do not translate everything into English unless requested.

CONVERSATION:
- Use the supplied conversation history to understand follow-up
  questions and maintain context.
- Do not invent memories or claim to know facts not supplied.
- If a question is ambiguous, ask a brief clarifying question.

ANSWER QUALITY:
- Be helpful, accurate, friendly, and clear.
- Use Markdown for headings, bold text, and lists when useful.
- Keep simple answers concise and explain complex topics clearly.
- Preserve names, numbers, and technical terms accurately.
"""

    # Read the existing history for this conversation.
    with conversation_lock:
        previous_history = conversations.get(
            conversation_id, []
        ).copy()

    prompt_parts = [system_prompt.strip()]

    for item in previous_history:
        role = "User" if item["role"] == "user" else "BhashaLM"
        prompt_parts.append(f'{role}: {item["content"]}')

    prompt_parts.append(f"User: {message}")
    prompt_parts.append("BhashaLM:")

    prompt = "\n\n".join(prompt_parts)

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.5
                }
            },
            timeout=180
        )

        response.raise_for_status()
        reply = response.json().get("response", "").strip()

        if not reply:
            reply = (
                "I couldn't generate a response. "
                "Please try asking in another way."
            )

        with conversation_lock:
            history = conversations.setdefault(
                conversation_id, []
            )

            history.extend([
                {"role": "user", "content": message},
                {"role": "assistant", "content": reply}
            ])

            conversations[conversation_id] = history[
                -MAX_HISTORY_MESSAGES:
            ]

        return {"reply": reply}

    except requests.exceptions.ConnectionError:
        return {
            "reply": (
                "I couldn't connect to Ollama. "
                "Please check that Ollama is running."
            )
        }

    except requests.exceptions.Timeout:
        return {
            "reply": (
                "The response took too long. "
                "Please try again."
            )
        }

    except requests.exceptions.RequestException:
        return {
            "reply": (
                "There was a problem contacting the AI model. "
                "Please try again."
            )
        }


@app.delete("/chat/{conversation_id}")
def clear_conversation(conversation_id: str):
    with conversation_lock:
        conversations.pop(conversation_id, None)

    return {"status": "Conversation cleared"}
