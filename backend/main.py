from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal
from threading import Lock

from bson import ObjectId
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel, Field
from pymongo import DESCENDING
from pymongo.errors import PyMongoError
from pymongo.asynchronous.mongo_client import AsyncMongoClient
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("chatzy")

Tone = Literal["professional", "casual", "concise"]
PromptCategory = Literal[
    "Programming", "Productivity", "Learning", "Writing", "Brainstorming"
]

TONE_INSTRUCTIONS: dict[Tone, str] = {
    "professional": "Use a professional, clear, and courteous tone.",
    "casual": "Use a warm, relaxed, conversational tone.",
    "concise": "Be concise and direct while still answering the question fully.",
}

ACTIVE_GENERATIONS: set[str] = set()
GENERATION_LOCK = Lock()


def cancel_generation(conversation_id: str) -> None:
    with GENERATION_LOCK:
        ACTIVE_GENERATIONS.add(conversation_id)


def start_generation(conversation_id: str) -> None:
    with GENERATION_LOCK:
        ACTIVE_GENERATIONS.discard(conversation_id)


def is_generation_cancelled(conversation_id: str) -> bool:
    with GENERATION_LOCK:
        return conversation_id in ACTIVE_GENERATIONS

TRANSFORM_INSTRUCTIONS = {
    "simpler": "Rewrite the response so it is easier to understand, using plain language. Preserve the important facts.",
    "expand": "Expand the response with useful detail and explain the reasoning behind the key points.",
    "examples": "Keep the key explanation and add clear, relevant examples.",
    "bullets": "Rewrite the response as a well-organized list of bullet points.",
    "professional": "Rewrite the response in a polished, professional tone without changing its meaning.",
}

BUILT_IN_PROMPTS = [
    {
        "id": "builtin:programming:debug",
        "category": "Programming",
        "title": "Debug my code",
        "prompt": "Help me debug this code. Explain the cause, then suggest a clear fix:\n\n",
    },
    {
        "id": "builtin:programming:review",
        "category": "Programming",
        "title": "Review my code",
        "prompt": "Review this code for correctness, clarity, and edge cases. Prioritize actionable improvements:\n\n",
    },
    {
        "id": "builtin:productivity:plan",
        "category": "Productivity",
        "title": "Plan my week",
        "prompt": "Help me make a realistic weekly plan. My priorities and commitments are:\n\n",
    },
    {
        "id": "builtin:productivity:breakdown",
        "category": "Productivity",
        "title": "Break down a task",
        "prompt": "Break this goal into small, practical steps and suggest what I should do first:\n\n",
    },
    {
        "id": "builtin:learning:explain",
        "category": "Learning",
        "title": "Teach me a concept",
        "prompt": "Teach me this concept step by step, starting with an intuitive explanation:\n\n",
    },
    {
        "id": "builtin:learning:quiz",
        "category": "Learning",
        "title": "Make a practice quiz",
        "prompt": "Create a short practice quiz about this topic, then provide an answer key:\n\n",
    },
    {
        "id": "builtin:writing:edit",
        "category": "Writing",
        "title": "Improve my writing",
        "prompt": "Improve the clarity and flow of this writing while preserving my meaning and voice:\n\n",
    },
    {
        "id": "builtin:writing:outline",
        "category": "Writing",
        "title": "Create an outline",
        "prompt": "Help me create a clear outline for a piece about this subject:\n\n",
    },
    {
        "id": "builtin:brainstorming:ideas",
        "category": "Brainstorming",
        "title": "Explore fresh ideas",
        "prompt": "Suggest a range of creative, practical ideas for this challenge:\n\n",
    },
    {
        "id": "builtin:brainstorming:names",
        "category": "Brainstorming",
        "title": "Find a name",
        "prompt": "Brainstorm memorable names for this project. Include a short explanation for each:\n\n",
    },
]


class ConversationCreate(BaseModel):
    title: str = Field(default="New chat", min_length=1, max_length=120)


class ConversationRename(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    tone: Tone | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20000)
    tone: Tone = "professional"


class TransformRequest(BaseModel):
    messageId: str
    action: Literal["simpler", "expand", "examples", "bullets", "professional"]


class PromptCreate(BaseModel):
    title: str = Field(min_length=1, max_length=100)
    prompt: str = Field(min_length=1, max_length=4000)
    category: PromptCategory


class ChatSendRequest(ChatRequest):
    conversationId: str | None = None


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def isoformat(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def serialize_conversation(document: dict) -> dict:
    return {
        "id": str(document["_id"]),
        "title": document["title"],
        "updatedAt": isoformat(document["updatedAt"]),
        "tone": document.get("tone", "professional"),
        "isStarred": document.get("isStarred", False),
    }


def serialize_message(document: dict) -> dict:
    message = {
        "id": str(document["_id"]),
        "role": document["role"],
        "content": document["content"],
        "createdAt": isoformat(document["createdAt"]),
        "tone": document.get("tone"),
        "status": document.get("status", "complete"),
    }
    if "transformation" in document:
        message["transformation"] = document["transformation"]
    return message


def parse_object_id(value: str, resource: str) -> ObjectId:
    if not ObjectId.is_valid(value):
        raise HTTPException(status_code=404, detail=f"{resource} not found.")
    return ObjectId(value)


def database(request: Request):
    db = request.app.state.db
    if db is None:
        raise HTTPException(
            status_code=503,
            detail="MongoDB is not configured. Set MONGODB_URI in backend/.env.",
        )
    return db


def openai_client(request: Request) -> AsyncOpenAI:
    client = request.app.state.openai
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="OpenAI is not configured. Set OPENAI_API_KEY in backend/.env.",
        )
    return client


def model_name() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db = None
    app.state.mongo_client = None
    app.state.openai = None

    mongo_uri = os.getenv("MONGODB_URI")
    if mongo_uri:
        try:
            client = AsyncMongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            await client.admin.command("ping")
            db = client[os.getenv("MONGODB_DATABASE", "chatzy_ai")]
            await db.conversations.create_index([("updatedAt", DESCENDING)])
            await db.messages.create_index(
                [("conversationId", 1), ("createdAt", 1)]
            )
            await db.prompts.create_index([("createdAt", DESCENDING)])
            app.state.mongo_client = client
            app.state.db = db
        except PyMongoError as error:
            logger.exception("MongoDB connection failed; starting in degraded mode")

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        app.state.openai = AsyncOpenAI(api_key=api_key)

    try:
        yield
    finally:
        if app.state.openai is not None:
            await app.state.openai.close()
        if app.state.mongo_client is not None:
            await app.state.mongo_client.close()


app = FastAPI(title="CHATZY AI API", version="1.0.0", lifespan=lifespan)
allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


RATE_WINDOW_SECONDS = 60
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "120"))
rate_requests: dict[str, list[float]] = {}
rate_lock = Lock()


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    from time import monotonic

    client_ip = request.client.host if request.client else "unknown"
    now = monotonic()
    with rate_lock:
        timestamps = [
            timestamp
            for timestamp in rate_requests.get(client_ip, [])
            if now - timestamp < RATE_WINDOW_SECONDS
        ]
        if len(timestamps) >= RATE_LIMIT_REQUESTS:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please try again shortly."},
            )
        timestamps.append(now)
        rate_requests[client_ip] = timestamps
    return await call_next(request)


@app.get("/api/health")
async def health(request: Request):
    db = request.app.state.db
    if db is None:
        return {"status": "degraded", "database": "not_configured"}
    try:
        await db.command("ping")
    except PyMongoError as error:
        logger.exception("MongoDB health check failed")
        raise HTTPException(status_code=503, detail="MongoDB is unavailable.") from error
    return {
        "status": "ok",
        "database": "connected",
        "openai": "configured" if request.app.state.openai else "not_configured",
    }


@app.get("/api/conversations")
async def list_conversations(request: Request):
    db = database(request)
    documents = await db.conversations.find().sort("updatedAt", DESCENDING).to_list(
        length=200
    )
    return [serialize_conversation(document) for document in documents]


@app.post("/api/conversations", status_code=201)
async def create_conversation(body: ConversationCreate, request: Request):
    db = database(request)
    created = now_utc()
    document = {
        "title": body.title.strip() or "New chat",
        "createdAt": created,
        "updatedAt": created,
        "tone": "professional",
        "isStarred": False,
        "messagesCount": 0,
    }
    result = await db.conversations.insert_one(document)
    document["_id"] = result.inserted_id
    return serialize_conversation(document)


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, request: Request):
    db = database(request)
    object_id = parse_object_id(conversation_id, "Conversation")
    document = await db.conversations.find_one({"_id": object_id})
    if document is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    messages = await db.messages.find({"conversationId": object_id}).sort(
        "createdAt", 1
    ).to_list(length=1000)
    return {
        **serialize_conversation(document),
        "messages": [serialize_message(message) for message in messages],
    }


@app.patch("/api/conversations/{conversation_id}")
async def rename_conversation(
    conversation_id: str, body: ConversationRename, request: Request
):
    db = database(request)
    object_id = parse_object_id(conversation_id, "Conversation")
    updates = {"title": body.title.strip(), "updatedAt": now_utc()}
    if body.tone is not None:
        updates["tone"] = body.tone
    document = await db.conversations.find_one_and_update(
        {"_id": object_id},
        {"$set": updates},
        return_document=True,
    )
    if document is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    document["title"] = body.title.strip()
    return serialize_conversation(document)


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, request: Request):
    db = database(request)
    object_id = parse_object_id(conversation_id, "Conversation")
    result = await db.conversations.delete_one({"_id": object_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    await db.messages.delete_many({"conversationId": object_id})
    return {"deleted": True}


@app.patch("/api/conversations/{conversation_id}/star")
async def toggle_star(conversation_id: str, request: Request, starred: bool = False):
    db = database(request)
    object_id = parse_object_id(conversation_id, "Conversation")
    document = await db.conversations.find_one_and_update(
        {"_id": object_id},
        {"$set": {"isStarred": starred, "updatedAt": now_utc()}},
        return_document=True,
    )
    if document is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return serialize_conversation(document)


def event_stream(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.post("/api/conversations/{conversation_id}/messages/stream")
async def stream_message(
    conversation_id: str, body: ChatRequest, request: Request
):
    db = database(request)
    client = openai_client(request)
    conversation_object_id = parse_object_id(conversation_id, "Conversation")
    conversation = await db.conversations.find_one({"_id": conversation_object_id})
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    user_content = body.message.strip()
    if not user_content:
        raise HTTPException(status_code=422, detail="Message cannot be blank.")
    created = now_utc()
    user_document = {
        "conversationId": conversation_object_id,
        "role": "user",
        "content": user_content,
        "createdAt": created,
        "tone": body.tone,
        "status": "complete",
    }
    user_result = await db.messages.insert_one(user_document)
    user_document["_id"] = user_result.inserted_id

    title = conversation["title"]
    if title == "New chat":
        title = user_content[:60].strip()
        if len(user_content) > 60:
            title = f"{title.rstrip()}…"
    await db.conversations.update_one(
        {"_id": conversation_object_id},
        {
            "$set": {"title": title, "tone": body.tone, "updatedAt": created},
            "$inc": {"messagesCount": 1},
        },
    )

    history = await db.messages.find(
        {"conversationId": conversation_object_id}
    ).sort("createdAt", DESCENDING).limit(40).to_list(length=40)
    history.reverse()
    model_messages = [
        {
            "role": "system",
            "content": (
                "You are CHATZY AI, a helpful, accurate assistant. "
                f"{TONE_INSTRUCTIONS[body.tone]} "
                "Use Markdown for structure and fenced code blocks for code."
            ),
        }
    ]
    model_messages.extend(
        {"role": item["role"], "content": item["content"]} for item in history
    )

    async def generate():
        response_parts: list[str] = []
        try:
            stream = await client.chat.completions.create(
                model=model_name(),
                messages=model_messages,
                stream=True,
            )
            async for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    response_parts.append(token)
                    yield event_stream({"token": token})

            if is_generation_cancelled(str(conversation_object_id)):
                yield event_stream({"cancelled": True})
                return
            content = "".join(response_parts)
            assistant_document = {
                "conversationId": conversation_object_id,
                "role": "assistant",
                "content": content,
                "createdAt": now_utc(),
                "tone": body.tone,
                "status": "complete",
            }
            result = await db.messages.insert_one(assistant_document)
            assistant_document["_id"] = result.inserted_id
            await db.conversations.update_one(
                {"_id": conversation_object_id},
                {
                    "$set": {"updatedAt": assistant_document["createdAt"]},
                    "$inc": {"messagesCount": 1},
                },
            )
            yield event_stream(
                {"done": True, "message": serialize_message(assistant_document)}
            )
        except (OpenAIError, PyMongoError) as error:
            logger.exception("Chat response failed")
            yield event_stream(
                {"error": "The response could not be completed. Please try again."}
            )
            if isinstance(error, PyMongoError):
                raise
        finally:
            with GENERATION_LOCK:
                ACTIVE_GENERATIONS.discard(str(conversation_object_id))

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/chat/{conversation_id}/stop")
async def stop_generation(conversation_id: str, request: Request):
    database(request)
    parse_object_id(conversation_id, "Conversation")
    cancel_generation(conversation_id)
    return {"cancelled": True}


@app.post("/api/chat/send")
async def send_chat(body: ChatSendRequest, request: Request):
    db = database(request)
    client = openai_client(request)
    created = now_utc()

    if body.conversationId:
        conversation_id = parse_object_id(body.conversationId, "Conversation")
        conversation = await db.conversations.find_one({"_id": conversation_id})
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
    else:
        conversation = {
            "title": "New chat",
            "createdAt": created,
            "updatedAt": created,
            "tone": "professional",
            "isStarred": False,
            "messagesCount": 0,
        }
        result = await db.conversations.insert_one(conversation)
        conversation_id = result.inserted_id

    user_content = body.message.strip()
    if not user_content:
        raise HTTPException(status_code=422, detail="Message cannot be blank.")
    user_document = {
        "conversationId": conversation_id,
        "role": "user",
        "content": user_content,
        "createdAt": created,
        "tone": body.tone,
        "status": "complete",
    }
    await db.messages.insert_one(user_document)

    history = await db.messages.find({"conversationId": conversation_id}).sort(
        "createdAt", DESCENDING
    ).limit(40).to_list(length=40)
    history.reverse()
    model_messages = [{
        "role": "system",
        "content": f"You are CHATZY AI, a helpful, accurate assistant. {TONE_INSTRUCTIONS[body.tone]} Use Markdown for structure.",
    }]
    model_messages.extend({"role": item["role"], "content": item["content"]} for item in history)

    try:
        completion = await client.chat.completions.create(
            model=model_name(),
            messages=model_messages,
        )
    except OpenAIError as error:
        logger.exception("Chat response failed")
        raise HTTPException(status_code=502, detail="The response could not be generated.") from error

    content = completion.choices[0].message.content
    if not content:
        raise HTTPException(status_code=502, detail="The AI returned an empty response.")
    assistant_document = {
        "conversationId": conversation_id,
        "role": "assistant",
        "content": content,
        "createdAt": now_utc(),
        "tone": body.tone,
        "status": "complete",
    }
    result = await db.messages.insert_one(assistant_document)
    assistant_document["_id"] = result.inserted_id
    title = conversation["title"]
    if title == "New chat":
        title = user_content[:60].strip()
        if len(user_content) > 60:
            title += "…"
    await db.conversations.update_one(
        {"_id": conversation_id},
        {
            "$set": {"title": title, "tone": body.tone, "updatedAt": assistant_document["createdAt"]},
            "$inc": {"messagesCount": 2},
        },
    )
    response = serialize_conversation({**conversation, "_id": conversation_id, "title": title})
    response["message"] = serialize_message(assistant_document)
    return response


@app.post("/api/conversations/{conversation_id}/transform")
async def transform_response(
    conversation_id: str, body: TransformRequest, request: Request
):
    db = database(request)
    client = openai_client(request)
    conversation_object_id = parse_object_id(conversation_id, "Conversation")
    if not ObjectId.is_valid(body.messageId):
        raise HTTPException(status_code=404, detail="Message not found.")
    original = await db.messages.find_one(
        {
            "_id": ObjectId(body.messageId),
            "conversationId": conversation_object_id,
            "role": "assistant",
        }
    )
    if original is None:
        raise HTTPException(status_code=404, detail="Message not found.")
    try:
        completion = await client.chat.completions.create(
            model=model_name(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Transform the supplied assistant response according to the "
                        "user's instruction. Return only the transformed response."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Instruction: {TRANSFORM_INSTRUCTIONS[body.action]}\n\n"
                        f"Response to transform:\n{original['content']}"
                    ),
                },
            ],
        )
    except OpenAIError as error:
        logger.exception("Response transformation failed")
        raise HTTPException(
            status_code=502, detail="The response could not be transformed. Please try again."
        ) from error

    content = completion.choices[0].message.content
    if not content:
        raise HTTPException(status_code=502, detail="The AI returned an empty response.")
    created = now_utc()
    document = {
        "conversationId": conversation_object_id,
        "role": "assistant",
        "content": content,
        "createdAt": created,
        "transformation": body.action,
    }
    result = await db.messages.insert_one(document)
    document["_id"] = result.inserted_id
    await db.conversations.update_one(
        {"_id": conversation_object_id}, {"$set": {"updatedAt": created}}
    )
    return serialize_message(document)


@app.get("/api/prompts")
async def list_prompts(request: Request):
    db = database(request)
    documents = await db.prompts.find().sort("createdAt", DESCENDING).to_list(
        length=200
    )
    saved = [
        {
            "id": str(document["_id"]),
            "title": document["title"],
            "prompt": document["prompt"],
            "category": document["category"],
            "saved": True,
        }
        for document in documents
    ]
    built_in = [
        {**prompt, "saved": False}
        for prompt in BUILT_IN_PROMPTS
    ]
    return saved + built_in


@app.post("/api/prompts", status_code=201)
async def save_prompt(body: PromptCreate, request: Request):
    db = database(request)
    document = {
        "title": body.title.strip(),
        "prompt": body.prompt.strip(),
        "category": body.category,
        "createdAt": now_utc(),
    }
    result = await db.prompts.insert_one(document)
    return {
        "id": str(result.inserted_id),
        "title": document["title"],
        "prompt": document["prompt"],
        "category": document["category"],
        "saved": True,
    }


@app.get("/api/prompts/saved")
async def list_saved_prompts(request: Request):
    db = database(request)
    documents = await db.prompts.find().sort("createdAt", DESCENDING).to_list(length=200)
    return [
        {
            "id": str(document["_id"]),
            "title": document["title"],
            "prompt": document["prompt"],
            "category": document["category"],
            "saved": True,
        }
        for document in documents
    ]


@app.delete("/api/prompts/{prompt_id}")
async def delete_prompt(prompt_id: str, request: Request):
    db = database(request)
    object_id = parse_object_id(prompt_id, "Prompt")
    result = await db.prompts.delete_one({"_id": object_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Prompt not found.")
    return {"deleted": True}
