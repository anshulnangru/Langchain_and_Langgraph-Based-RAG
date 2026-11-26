import { useState, useRef, useEffect, useCallback, Component } from 'react'
import ReactMarkdown from 'react-markdown'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const API_KEY = import.meta.env.VITE_API_KEY_NEW || ''

// Each visitor gets their own conversation memory instead of sharing one thread.
const newThreadId = () => `demo_${Math.random().toString(36).slice(2, 10)}`

const SUGGESTIONS = [
  'What is LangGraph and when should I use it?',
  'How do conditional edges work in LangGraph?',
  'How do I add memory with a checkpointer?',
  'LangChain vs LangGraph: what is the difference?',
]

// Chat history lives in this browser only. The server keeps memory in RAM,
// so an old chat still shows here after a backend restart but loses its context.
const STORAGE_KEY = 'rag_chats_v1'

function loadChats() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) || []
  } catch {
    return []
  }
}

function saveChats(chats) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(chats))
  } catch {
    // storage full or blocked: history just won't persist
  }
}

function formatTime(ts) {
  return new Date(ts).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
}

// ─── Error Boundary ───────────────────────────────────────────
class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, error: null }
  }
  static getDerivedStateFromError(error) {
    return { hasError: true, error }
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="message-bubble error-bubble">
          ⚠️ Failed to render response: {String(this.state.error?.message || 'Unknown error')}
        </div>
      )
    }
    return this.props.children
  }
}

// ─── Typing Indicator ─────────────────────────────────────────
function TypingIndicator() {
  return (
    <div className="message ai fade-in">
      <div className="message-avatar">🤖</div>
      <div className="message-content">
        <div className="typing-indicator">
          <div className="typing-dot" />
          <div className="typing-dot" />
          <div className="typing-dot" />
        </div>
      </div>
    </div>
  )
}

// ─── Thought Process accordion ────────────────────────────────
function ThoughtProcess({ steps }) {
  const [open, setOpen] = useState(false)
  if (!steps || steps.length === 0) return null

  return (
    <div className="thought-process">
      <button
        id="thought-toggle"
        className={`thought-toggle ${open ? 'open' : ''}`}
        onClick={() => setOpen(o => !o)}
      >
        ✨ Thought process ({steps.length} steps)
        <span className="chevron">▼</span>
      </button>
      {open && (
        <div className="thought-steps fade-in">
          {steps.map((step, i) => (
            <div className="thought-step" key={i}>
              <span className="step-index">{i + 1}</span>
              <span>{String(step)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ─── Source tags ──────────────────────────────────────────────
function Sources({ sources }) {
  if (!sources || sources.length === 0) return null

  const labels = sources.map((s, i) => {
    if (typeof s === 'string') return s
    if (typeof s === 'object' && s !== null) {
      return s.metadata?.source || s.page_content?.slice(0, 40) || `Source ${i + 1}`
    }
    return `Source ${i + 1}`
  })

  return (
    <div className="sources">
      {labels.slice(0, 5).map((label, i) => (
        <span className="source-tag" key={i}>
          📎 {label}
        </span>
      ))}
    </div>
  )
}

// ─── Single message ───────────────────────────────────────────
function Message({ msg }) {
  const isUser = msg.role === 'user'
  const text = msg.text ?? ''

  return (
    <div className={`message ${isUser ? 'user' : 'ai'}`}>
      <div className="message-avatar">
        {isUser ? '👤' : '🤖'}
      </div>
      <div className="message-content">
        <div className={`message-bubble ${msg.error ? 'error-bubble' : ''}`}>
          {isUser ? (
            text
          ) : (
            <ErrorBoundary>
              <div className="md-content">
                <ReactMarkdown>{text}</ReactMarkdown>
              </div>
            </ErrorBoundary>
          )}
        </div>
        {!isUser && msg.thoughtProcess && (
          <ThoughtProcess steps={msg.thoughtProcess} />
        )}
        {!isUser && msg.sources && (
          <Sources sources={msg.sources} />
        )}
        <span className="message-meta">{formatTime(msg.timestamp)}</span>
      </div>
    </div>
  )
}

// ─── Main App ─────────────────────────────────────────────────
export default function App() {
  const [messages, setMessages] = useState([])
  const [query, setQuery] = useState('')
  const [threadId, setThreadId] = useState(newThreadId)
  const [loading, setLoading] = useState(false)
  const [apiStatus, setApiStatus] = useState('checking')
  const [chats, setChats] = useState(loadChats)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const messagesEndRef = useRef(null)
  const textareaRef = useRef(null)

  // Save the open conversation into history whenever it changes.
  useEffect(() => {
    if (messages.length === 0) return
    setChats(prev => {
      const title = messages.find(m => m.role === 'user')?.text.slice(0, 60) || 'New chat'
      const chat = { id: threadId, title, messages, updatedAt: Date.now() }
      const next = [chat, ...prev.filter(c => c.id !== threadId)]
      saveChats(next)
      return next
    })
  }, [messages, threadId])

  const openChat = (chat) => {
    if (loading) return
    setThreadId(chat.id)
    setMessages(chat.messages)
    setSidebarOpen(false)
  }

  const newChat = () => {
    if (loading) return
    setThreadId(newThreadId())
    setMessages([])
    setSidebarOpen(false)
  }

  const deleteChat = (id) => {
    if (loading) return
    setChats(prev => {
      const next = prev.filter(c => c.id !== id)
      saveChats(next)
      return next
    })
    if (id === threadId) {
      setThreadId(newThreadId())
      setMessages([])
    }
  }

  // Check API health. A free-tier host sleeps when idle, so if the first
  // check is slow, tell the visitor the server is waking up.
  useEffect(() => {
    let cancelled = false
    const wakeTimer = setTimeout(() => {
      if (!cancelled) setApiStatus(s => (s === 'checking' ? 'waking' : s))
    }, 4000)

    fetch(`${API_BASE}/`)
      .then(r => {
        if (!cancelled) setApiStatus(r.ok ? 'online' : 'error')
      })
      .catch(() => {
        if (!cancelled) setApiStatus('error')
      })

    return () => {
      cancelled = true
      clearTimeout(wakeTimer)
    }
  }, [])

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  // Auto-resize textarea
  useEffect(() => {
    const ta = textareaRef.current
    if (ta) {
      ta.style.height = 'auto'
      ta.style.height = Math.min(ta.scrollHeight, 140) + 'px'
    }
  }, [query])

  const sendQuery = useCallback(async (text = query) => {
    const q = typeof text === 'string' ? text.trim() : ''
    if (!q || loading) return

    setQuery('')
    setMessages(prev => [...prev, {
      role: 'user',
      text: q,
      timestamp: Date.now(),
    }])
    setLoading(true)

    // Add a blank AI message that we'll fill token-by-token
    const aiMsgTimestamp = Date.now()
    setMessages(prev => [...prev, {
      role: 'ai',
      text: '',
      thoughtProcess: [],
      sources: [],
      timestamp: aiMsgTimestamp,
      streaming: true,
    }])

    try {
      const res = await fetch(`${API_BASE}/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': API_KEY,
        },
        body: JSON.stringify({ q, thread_id: threadId }),
      })

      if (res.status === 429) throw new Error('RATE_LIMIT')
      if (res.status === 401 || res.status === 403) throw new Error('AUTH')
      if (!res.ok) throw new Error(`HTTP ${res.status}`)

      setApiStatus('online')

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { value, done } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() // keep incomplete line in buffer

        for (const line of lines) {
          if (!line.startsWith('data:')) continue
          const payload = line.slice(5).trim()
          if (payload === '[DONE]') break

          try {
            const parsed = JSON.parse(payload)
            if (parsed.token) {
              // Append token to the last AI message
              setMessages(prev => {
                const updated = [...prev]
                const last = { ...updated[updated.length - 1] }
                last.text = (last.text ?? '') + parsed.token
                updated[updated.length - 1] = last
                return updated
              })
            }
            if (parsed.error) {
              setMessages(prev => {
                const updated = [...prev]
                const last = { ...updated[updated.length - 1] }
                last.text = `⚠️ Stream error: ${parsed.error}`
                last.error = true
                updated[updated.length - 1] = last
                return updated
              })
            }
          } catch {
            // ignore malformed SSE lines
          }
        }
      }

      // Mark streaming done
      setMessages(prev => {
        const updated = [...prev]
        const last = { ...updated[updated.length - 1] }
        last.streaming = false
        updated[updated.length - 1] = last
        return updated
      })
    } catch (err) {
      let message
      if (err.message === 'RATE_LIMIT') {
        message = '⚠️ Too many requests. Wait a minute and try again.'
      } else if (err.message === 'AUTH') {
        message = '⚠️ The server rejected this request (invalid or missing API key).'
      } else {
        message = `⚠️ Could not reach the backend API. If it has been idle, it can take up to a minute to wake up. Try again shortly.\n\nError: ${err.message}`
      }
      setMessages(prev => {
        const updated = [...prev]
        const last = { ...updated[updated.length - 1] }
        last.text = message
        last.error = true
        last.streaming = false
        updated[updated.length - 1] = last
        return updated
      })
    } finally {
      setLoading(false)
    }
  }, [query, threadId, loading])

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendQuery()
    }
  }

  const statusColor = {
    online: '#34d399',
    error: '#f87171',
    checking: '#fbbf24',
    waking: '#fbbf24',
  }

  const statusLabel = {
    online: 'API Online',
    error: 'API Offline',
    checking: 'Connecting...',
    waking: 'Waking up server...',
  }

  return (
    <div className="shell">
      {/* Chat history sidebar */}
      <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`} aria-label="Chat history">
        <button className="new-chat-btn" onClick={newChat} disabled={loading}>
          + New chat
        </button>
        <div className="history-label">History</div>
        <nav className="history-list">
          {chats.length === 0 && <div className="history-empty">No conversations yet</div>}
          {chats.map(chat => (
            <div key={chat.id} className={`history-item ${chat.id === threadId ? 'active' : ''}`}>
              <button
                className="history-open"
                onClick={() => openChat(chat)}
                disabled={loading}
                title={chat.title}
              >
                {chat.title}
              </button>
              <button
                className="history-delete"
                onClick={() => deleteChat(chat.id)}
                disabled={loading}
                aria-label={`Delete chat: ${chat.title}`}
              >
                ×
              </button>
            </div>
          ))}
        </nav>
      </aside>
      {sidebarOpen && <div className="sidebar-backdrop" onClick={() => setSidebarOpen(false)} />}

    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-brand">
          <button
            className="menu-btn"
            onClick={() => setSidebarOpen(o => !o)}
            aria-label="Show chat history"
            aria-expanded={sidebarOpen}
          >
            ☰
          </button>
          <div className="header-logo">🧠</div>
          <div>
            <div className="header-title">RAG Assistant</div>
            <div className="header-subtitle">Agentic RAG · LangGraph</div>
          </div>
        </div>
        <div className="status-pill">
          <div className="status-dot" style={{ background: statusColor[apiStatus] }} />
          {statusLabel[apiStatus]}
        </div>
      </header>

      {/* Wake-up notice (free-tier hosts sleep when idle) */}
      {apiStatus === 'waking' && (
        <div
          role="status"
          style={{
            padding: '10px 20px',
            fontSize: '0.875rem',
            background: 'rgba(251, 191, 36, 0.12)',
            borderBottom: '1px solid rgba(251, 191, 36, 0.35)',
          }}
        >
          The server was idle and is starting up. This can take up to a minute.
          You can type your question now, and it will send once the server is ready.
        </div>
      )}

      {/* Messages */}
      <main className="messages-area" id="messages-area" role="log" aria-live="polite">
        {messages.length === 0 && !loading ? (
          <div className="empty-state">
            <div className="empty-icon">🧠</div>
            <h1 className="empty-title">Ask about LangChain & LangGraph</h1>
            <p className="empty-desc">
              Answers grounded in the official LangChain and LangGraph documentation,
              retrieved and reranked by an agentic RAG pipeline.
            </p>
            <div className="suggestions">
              {SUGGESTIONS.map((s, i) => (
                <button
                  key={i}
                  className="suggestion-chip"
                  onClick={() => sendQuery(s)}
                  id={`suggestion-${i}`}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {messages.map((msg, i) => (
              <Message key={i} msg={msg} />
            ))}
            {loading && <TypingIndicator />}
          </>
        )}
        <div ref={messagesEndRef} />
      </main>

      {/* Input */}
      <footer className="input-area">
        <div className="input-wrapper">
          <textarea
            ref={textareaRef}
            id="query-input"
            className="query-input"
            placeholder="Ask about LangChain or LangGraph…"
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
            disabled={loading}
          />
          <div className="input-actions">
            <button
              id="send-btn"
              className="send-btn"
              onClick={() => sendQuery()}
              disabled={!query.trim() || loading}
              aria-label="Send message"
            >
              <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
              </svg>
            </button>
          </div>
        </div>
        <div className="input-hint">
          <span>
            <kbd>↵</kbd> send · <kbd>⇧↵</kbd> newline
          </span>
        </div>
      </footer>
    </div>
    </div>
  )
}