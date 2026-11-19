# CHATZY AI

A lightweight, single-page AI chat app with a React + TypeScript frontend and a FastAPI backend. Chat history and saved prompts are stored in MongoDB; OpenAI credentials stay on the backend.

## Requirements

- Node.js 18+
- Python 3.10+
- A MongoDB Atlas connection string
- An OpenAI API key

## Configure the backend

From the repository root, create `backend/.env` using `backend/.env.example` as a template. Set `OPENAI_API_KEY` and `MONGODB_URI`; the remaining values have sensible defaults. Never put the OpenAI key in the frontend.

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --env-file .env
```

The API is available at `http://localhost:8000`; its interactive docs are at `http://localhost:8000/docs`.

## Configure the frontend

In another terminal:

```powershell
cd frontend
npm install
Copy-Item .env.example .env
npm run dev
```

Open the Vite URL shown in the terminal (usually `http://localhost:5173`). Set `VITE_API_BASE_URL` in `frontend/.env` if the backend runs at a different address.

## Features

- Streaming AI chat, contextual conversation history, tone selection, and stop generation.
- Create, rename, and delete saved conversations.
- Markdown and syntax-highlighted code responses.
- Response Studio transformations saved to the active conversation.
- Categorized built-in prompts and user-created prompts saved in MongoDB.

If MongoDB or OpenAI is not configured, the backend starts with degraded health status and returns a clear configuration error when an affected feature is used.
