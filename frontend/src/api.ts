const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api").replace(/\/$/, "");

export type Tone = "professional" | "casual" | "concise";
export type PromptCategory = "Programming" | "Productivity" | "Learning" | "Writing" | "Brainstorming";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: string;
  transformation?: string;
}

export interface Conversation {
  id: string;
  title: string;
  updatedAt: string;
  messages?: ChatMessage[];
}

export interface LibraryPrompt {
  id: string;
  title: string;
  prompt: string;
  category: PromptCategory;
  saved: boolean;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: {
        ...(init?.body ? { "Content-Type": "application/json" } : {}),
        ...init?.headers,
      },
    });
  } catch {
    throw new Error("Cannot reach CHATZY AI. Check that the backend is running.");
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? `Request failed (${response.status}).`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  conversations: () => request<Conversation[]>("/conversations"),
  conversation: (id: string) => request<Conversation>(`/conversations/${id}`),
  createConversation: () =>
    request<Conversation>("/conversations", { method: "POST", body: JSON.stringify({}) }),
  renameConversation: (id: string, title: string) =>
    request<Conversation>(`/conversations/${id}`, {
      method: "PATCH",
      body: JSON.stringify({ title }),
    }),
  deleteConversation: (id: string) =>
    request<{ deleted: boolean }>(`/conversations/${id}`, { method: "DELETE" }),
  prompts: () => request<LibraryPrompt[]>("/prompts"),
  savePrompt: (prompt: Omit<LibraryPrompt, "id" | "saved">) =>
    request<LibraryPrompt>("/prompts", { method: "POST", body: JSON.stringify(prompt) }),
  deletePrompt: (id: string) => request<{ deleted: boolean }>(`/prompts/${id}`, { method: "DELETE" }),
  transform: (conversationId: string, messageId: string, action: string) =>
    request<ChatMessage>(`/conversations/${conversationId}/transform`, {
      method: "POST",
      body: JSON.stringify({ messageId, action }),
    }),
};

interface StreamCallbacks {
  onToken: (token: string) => void;
  signal: AbortSignal;
}

export async function streamMessage(
  conversationId: string,
  message: string,
  tone: Tone,
  { onToken, signal }: StreamCallbacks,
): Promise<ChatMessage> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/conversations/${conversationId}/messages/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, tone }),
      signal,
    });
  } catch (error) {
    if (signal.aborted) throw error;
    throw new Error("Cannot reach CHATZY AI. Check that the backend is running.");
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? `Request failed (${response.status}).`);
  }
  if (!response.body) throw new Error("The server did not start a response stream.");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result: ChatMessage | undefined;

  const consume = (frame: string) => {
    const data = frame
      .split(/\r?\n/)
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trim())
      .join("\n");
    if (!data) return;
    const event = JSON.parse(data) as { token?: string; done?: boolean; message?: ChatMessage; error?: string };
    if (event.token) onToken(event.token);
    if (event.error) throw new Error(event.error);
    if (event.done && event.message) result = event.message;
  };

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const frames = buffer.split(/\r?\n\r?\n/);
    buffer = frames.pop() ?? "";
    for (const frame of frames) consume(frame);
    if (done) break;
  }
  if (buffer.trim()) consume(buffer);
  if (!result && !signal.aborted) throw new Error("The response ended before it was saved.");
  return result as ChatMessage;
}
