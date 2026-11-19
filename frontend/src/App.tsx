import { FormEvent, KeyboardEvent, useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";
import {
  ArrowDown,
  ArrowUp,
  BookOpen,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  CircleStop,
  Copy,
  Ellipsis,
  Lightbulb,
  Menu,
  MessageSquare,
  Pencil,
  Plus,
  Sparkles,
  Trash2,
  WandSparkles,
  X,
} from "lucide-react";
import { api, ChatMessage, Conversation, LibraryPrompt, PromptCategory, streamMessage, Tone } from "./api";

const categories: PromptCategory[] = ["Programming", "Productivity", "Learning", "Writing", "Brainstorming"];
const tones: { value: Tone; label: string }[] = [
  { value: "professional", label: "Professional" },
  { value: "casual", label: "Casual" },
  { value: "concise", label: "Concise" },
];
const studioActions = [
  { value: "simpler", label: "Make simpler" },
  { value: "expand", label: "Expand explanation" },
  { value: "examples", label: "Give examples" },
  { value: "bullets", label: "Convert to bullet points" },
  { value: "professional", label: "Make professional" },
];

function App() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [tone, setTone] = useState<Tone>("professional");
  const [input, setInput] = useState("");
  const [prompts, setPrompts] = useState<LibraryPrompt[]>([]);
  const [category, setCategory] = useState<PromptCategory>("Programming");
  const [promptTitle, setPromptTitle] = useState("");
  const [promptText, setPromptText] = useState("");
  const [promptPanel, setPromptPanel] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [toneOpen, setToneOpen] = useState(false);
  const [menuId, setMenuId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [studioLoading, setStudioLoading] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const refreshConversations = useCallback(async () => {
    try {
      setConversations(await api.conversations());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not load conversation history.");
    }
  }, []);

  const refreshPrompts = useCallback(async () => {
    try {
      setPrompts(await api.prompts());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not load the prompt library.");
    }
  }, []);

  useEffect(() => {
    void refreshConversations();
    void refreshPrompts();
  }, [refreshConversations, refreshPrompts]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const newChat = () => {
    abortRef.current?.abort();
    setActiveId(null);
    setMessages([]);
    setInput("");
    setLoading(false);
    setError("");
    setSidebarOpen(false);
    textareaRef.current?.focus();
  };

  const openConversation = async (id: string) => {
    abortRef.current?.abort();
    setLoading(false);
    setError("");
    setSidebarOpen(false);
    try {
      const conversation = await api.conversation(id);
      setActiveId(conversation.id);
      setMessages(conversation.messages ?? []);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not open this conversation.");
    }
  };

  const sendMessage = async (event?: FormEvent) => {
    event?.preventDefault();
    const text = input.trim();
    if (!text || loading) return;
    setError("");
    setInput("");
    setLoading(true);
    const userMessage: ChatMessage = {
      id: `local-user-${Date.now()}`,
      role: "user",
      content: text,
      createdAt: new Date().toISOString(),
    };
    const pendingMessage: ChatMessage = {
      id: `local-assistant-${Date.now()}`,
      role: "assistant",
      content: "",
      createdAt: new Date().toISOString(),
    };
    setMessages((current) => [...current, userMessage, pendingMessage]);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      let conversationId = activeId;
      if (!conversationId) {
        const created = await api.createConversation();
        conversationId = created.id;
        setActiveId(created.id);
      }
      const response = await streamMessage(conversationId, text, tone, {
        signal: controller.signal,
        onToken: (token) =>
          setMessages((current) =>
            current.map((message) =>
              message.id === pendingMessage.id ? { ...message, content: message.content + token } : message,
            ),
          ),
      });
      if (response) {
        setMessages((current) =>
          current.map((message) => (message.id === pendingMessage.id ? response : message)),
        );
      } else {
        setMessages((current) => current.filter((message) => message.id !== pendingMessage.id));
      }
      await refreshConversations();
    } catch (reason) {
      if (!controller.signal.aborted) {
        setError(reason instanceof Error ? reason.message : "The response could not be generated.");
        setMessages((current) => current.filter((message) => message.id !== pendingMessage.id));
      }
    } finally {
      if (abortRef.current === controller) {
        setLoading(false);
        abortRef.current = null;
      }
    }
  };

  const stopGeneration = () => {
    abortRef.current?.abort();
    setLoading(false);
  };

  const transformMessage = async (message: ChatMessage, action: string) => {
    if (!activeId || studioLoading) return;
    setError("");
    setStudioLoading(`${message.id}:${action}`);
    try {
      const transformed = await api.transform(activeId, message.id, action);
      setMessages((current) => [...current, transformed]);
      await refreshConversations();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The response could not be transformed.");
    } finally {
      setStudioLoading(null);
    }
  };

  const copyMessage = async (message: ChatMessage) => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopiedId(message.id);
      window.setTimeout(() => setCopiedId(null), 1400);
    } catch {
      setError("Could not copy this message to the clipboard.");
    }
  };

  const renameConversation = async (conversation: Conversation) => {
    const title = window.prompt("Rename conversation", conversation.title)?.trim();
    if (!title || title === conversation.title) return;
    try {
      await api.renameConversation(conversation.id, title);
      await refreshConversations();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not rename this conversation.");
    }
    setMenuId(null);
  };

  const deleteConversation = async (conversation: Conversation) => {
    if (!window.confirm(`Delete “${conversation.title}”?`)) return;
    try {
      await api.deleteConversation(conversation.id);
      if (activeId === conversation.id) newChat();
      await refreshConversations();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not delete this conversation.");
    }
    setMenuId(null);
  };

  const savePrompt = async (event: FormEvent) => {
    event.preventDefault();
    if (!promptTitle.trim() || !promptText.trim()) return;
    try {
      const saved = await api.savePrompt({
        title: promptTitle.trim(),
        prompt: promptText.trim(),
        category,
      });
      setPrompts((current) => [saved, ...current]);
      setPromptTitle("");
      setPromptText("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not save this prompt.");
    }
  };

  const removePrompt = async (prompt: LibraryPrompt) => {
    if (!prompt.saved) return;
    try {
      await api.deletePrompt(prompt.id);
      setPrompts((current) => current.filter((item) => item.id !== prompt.id));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not delete this prompt.");
    }
  };

  const onComposerKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void sendMessage();
    }
  };

  const visiblePrompts = prompts.filter((prompt) => prompt.category === category);

  return (
    <div className="app-shell">
      {sidebarOpen && <button className="mobile-backdrop" aria-label="Close menu" onClick={() => setSidebarOpen(false)} />}
      <aside className={`sidebar ${sidebarOpen ? "sidebar-open" : ""}`}>
        <div className="brand">
          <div className="brand-mark"><Sparkles size={19} strokeWidth={2.2} /></div>
          <span>CHATZY <b>AI</b></span>
          <button className="icon-button mobile-close" aria-label="Close menu" onClick={() => setSidebarOpen(false)}><X size={18} /></button>
        </div>
        <button className="new-chat-button" onClick={newChat}>
          <Plus size={17} /> <span>New chat</span> <span className="shortcut">⌘ K</span>
        </button>
        <div className="history-heading">YOUR CHATS</div>
        <div className="conversation-list">
          {conversations.length === 0 && <p className="empty-history">Your conversations will appear here.</p>}
          {conversations.map((conversation) => (
            <div className={`conversation-row ${activeId === conversation.id ? "conversation-active" : ""}`} key={conversation.id}>
              <button className="conversation-select" onClick={() => void openConversation(conversation.id)}>
                <MessageSquare size={15} />
                <span>{conversation.title}</span>
              </button>
              <button className="icon-button conversation-menu-button" aria-label="Conversation options" onClick={() => setMenuId(menuId === conversation.id ? null : conversation.id)}>
                <Ellipsis size={17} />
              </button>
              {menuId === conversation.id && (
                <div className="conversation-menu">
                  <button onClick={() => void renameConversation(conversation)}><Pencil size={14} /> Rename</button>
                  <button className="danger-action" onClick={() => void deleteConversation(conversation)}><Trash2 size={14} /> Delete</button>
                </div>
              )}
            </div>
          ))}
        </div>
        <div className="sidebar-bottom">
          <div className="sidebar-tip"><Lightbulb size={17} /><span>Good ideas start with a conversation.</span></div>
          <span className="sidebar-version">Your thoughtful AI companion</span>
        </div>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <button className="icon-button mobile-menu" aria-label="Open menu" onClick={() => setSidebarOpen(true)}><Menu size={21} /></button>
          <button className="mobile-brand" onClick={newChat}><span className="brand-mark"><Sparkles size={16} /></span> CHATZY <b>AI</b></button>
          <div className="topbar-center"><span className="status-dot" /> Here to help</div>
          <button className={`library-toggle ${promptPanel ? "library-toggle-active" : ""}`} onClick={() => setPromptPanel(!promptPanel)}>
            <BookOpen size={16} /><span>Prompt library</span><span className="library-count">{prompts.length}</span>
          </button>
        </header>

        <section className="chat-scroll-area">
          <div className={`chat-content ${messages.length === 0 ? "chat-content-welcome" : ""}`}>
            {messages.length === 0 ? (
              <div className="welcome">
                <div className="welcome-orb"><Sparkles size={25} /></div>
                <span className="eyebrow">YOUR AI COMPANION</span>
                <h1>Hey there!<br /><span>How can I help you today?</span></h1>
                <p>Ask a question, explore an idea, or make something wonderful.</p>
                <div className="suggestion-grid">
                  {[
                    ["✦", "Explain a tricky concept", "in a simple way"],
                    ["↗", "Help me write", "something thoughtful"],
                    ["⌘", "Brainstorm an idea", "with me"],
                  ].map(([icon, title, caption]) => (
                    <button className="suggestion-card" key={title} onClick={() => setInput(`${title} ${caption}`)}>
                      <span className="suggestion-icon">{icon}</span><span><b>{title}</b><small>{caption}</small></span><ChevronRight size={16} />
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="message-list">
                {messages.map((message) => (
                  <article className={`message message-${message.role}`} key={message.id}>
                    {message.role === "assistant" && <div className="assistant-avatar"><Sparkles size={15} /></div>}
                    <div className="message-main">
                      {message.role === "assistant" && <div className="message-author">CHATZY <span>AI</span>{message.transformation && <small> · {message.transformation}</small>}</div>}
                      <div className={`message-bubble ${message.role === "user" ? "user-bubble" : ""}`}>
                        {message.role === "assistant" && !message.content ? <span className="typing-indicator"><i /><i /><i /></span> :
                          <div className="markdown-body"><ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>{message.content}</ReactMarkdown></div>}
                      </div>
                      {message.role === "assistant" && message.content && (
                        <>
                          <div className="message-tools">
                            <button onClick={() => void copyMessage(message)}>{copiedId === message.id ? <Check size={14} /> : <Copy size={14} />}{copiedId === message.id ? "Copied" : "Copy"}</button>
                          </div>
                          {!loading && !message.id.startsWith("local-") && (
                            <div className="studio-actions">
                              <span><WandSparkles size={13} /> Response studio</span>
                              {studioActions.map((action) => (
                                <button
                                  key={action.value}
                                  disabled={Boolean(studioLoading)}
                                  onClick={() => void transformMessage(message, action.value)}
                                >
                                  {studioLoading === `${message.id}:${action.value}` ? <span className="mini-spinner" /> : null}
                                  {action.label}
                                </button>
                              ))}
                            </div>
                          )}
                        </>
                      )}
                    </div>
                    {message.role === "user" && <div className="user-avatar">Y</div>}
                  </article>
                ))}
                <div ref={bottomRef} />
              </div>
            )}
          </div>
        </section>

        {error && <div className="error-banner" role="alert"><span>{error}</span><button aria-label="Dismiss error" onClick={() => setError("")}><X size={15} /></button></div>}
        <div className="composer-wrap">
          <form className="composer" onSubmit={(event) => void sendMessage(event)}>
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={onComposerKeyDown}
              placeholder="Message CHATZY AI..."
              rows={2}
              aria-label="Message"
            />
            <div className="composer-toolbar">
              <div className="tone-picker">
                <button type="button" className={`tone-button ${toneOpen ? "tone-open" : ""}`} onClick={() => setToneOpen(!toneOpen)}>
                  <span className="tone-sparkle"><Sparkles size={13} /></span>
                  <span>{tones.find((option) => option.value === tone)?.label}</span>
                  <ChevronDown size={14} />
                </button>
                {toneOpen && (
                  <div className="tone-menu">
                    {tones.map((option) => (
                      <button type="button" key={option.value} onClick={() => { setTone(option.value); setToneOpen(false); }}>
                        {option.label}{tone === option.value && <Check size={14} />}
                      </button>
                    ))}
                  </div>
                )}
              </div>
              <div className="composer-controls">
                {loading && <button type="button" className="stop-button" onClick={stopGeneration}><CircleStop size={16} /> Stop</button>}
                <span className="enter-hint">Shift + Enter for a new line</span>
                <button type="submit" className="send-button" disabled={!input.trim() || loading} aria-label="Send message">
                  <ArrowUp size={18} />
                </button>
              </div>
            </div>
          </form>
          <p className="disclaimer">CHATZY AI can make mistakes. Check important information.</p>
        </div>

        {promptPanel && (
          <aside className="prompt-drawer">
            <div className="drawer-heading"><div><span className="drawer-kicker">GET INSPIRED</span><h2>Prompt library</h2></div><button className="icon-button" aria-label="Close prompt library" onClick={() => setPromptPanel(false)}><X size={18} /></button></div>
            <div className="category-list">
              {categories.map((item) => <button key={item} className={category === item ? "category-active" : ""} onClick={() => setCategory(item)}>{item}</button>)}
            </div>
            <div className="prompt-list">
              {visiblePrompts.length === 0 && <p className="empty-prompts">No saved prompts in this category yet.</p>}
              {visiblePrompts.map((prompt) => (
                <article className="prompt-card" key={prompt.id}>
                  <div className="prompt-card-title"><b>{prompt.title}</b>{prompt.saved && <button className="icon-button prompt-delete" aria-label={`Delete ${prompt.title}`} onClick={() => void removePrompt(prompt)}><Trash2 size={14} /></button>}</div>
                  <p>{prompt.prompt}</p>
                  <button className="insert-prompt" onClick={() => { setInput(prompt.prompt); setPromptPanel(false); textareaRef.current?.focus(); }}>Use prompt <ArrowDown size={13} /></button>
                </article>
              ))}
            </div>
            <form className="save-prompt-form" onSubmit={(event) => void savePrompt(event)}>
              <span className="save-prompt-heading"><Plus size={14} /> Save your own</span>
              <input value={promptTitle} onChange={(event) => setPromptTitle(event.target.value)} placeholder="Prompt title" aria-label="Prompt title" maxLength={100} />
              <textarea value={promptText} onChange={(event) => setPromptText(event.target.value)} placeholder="Write a prompt to save..." aria-label="Prompt text" rows={3} maxLength={4000} />
              <button type="submit" disabled={!promptTitle.trim() || !promptText.trim()}>Save prompt</button>
            </form>
          </aside>
        )}
      </main>
      <button className="sidebar-edge-toggle" aria-label={sidebarOpen ? "Hide sidebar" : "Show sidebar"} onClick={() => setSidebarOpen(!sidebarOpen)}>
        {sidebarOpen ? <ChevronLeft size={15} /> : <ChevronRight size={15} />}
      </button>
    </div>
  );
}

export default App;
