import os

os.environ.pop("MONGODB_URI", None)
os.environ.pop("OPENAI_API_KEY", None)

from fastapi.testclient import TestClient

from main import (
    BUILT_IN_PROMPTS,
    TONE_INSTRUCTIONS,
    ChatRequest,
    ConversationCreate,
    app,
)


def test_health_is_degraded_when_services_are_not_configured():
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"


def test_tones_are_valid_and_distinct():
    assert set(TONE_INSTRUCTIONS) == {"professional", "casual", "concise"}
    assert len(set(TONE_INSTRUCTIONS.values())) == 3
    ChatRequest(message="Hello", tone="casual")


def test_conversation_and_chat_validation():
    assert ConversationCreate().title == "New chat"
    try:
        ChatRequest(message="")
    except Exception:
        pass
    else:
        raise AssertionError("Blank chat messages must fail validation")


def test_builtin_prompts_cover_expected_categories():
    categories = {prompt["category"] for prompt in BUILT_IN_PROMPTS}
    assert categories == {
        "Programming",
        "Productivity",
        "Learning",
        "Writing",
        "Brainstorming",
    }
