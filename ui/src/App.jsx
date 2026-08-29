import { useState, useRef, useEffect, useCallback, Component } from 'react'
import ReactMarkdown from 'react-markdown'

const API_BASE = 'http://localhost:8000'

const SUGGESTIONS = [
  '🧠 What topics are in my knowledge base?',
  '📄 Summarize the key concepts',
  '🔍 Find information about ...',
  '💡 Explain the main ideas',
]

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
  const [threadId, setThreadId] = useState('default_user')
  const [loading, setLoading] = useState(false)
  const [apiStatus, setApiStatus] = useState('checking')
  const messagesEndRef = useRef(null)
  const textareaRef = useRef(null)

  // Check API health
  useEffect(() => {
    fetch(`${API_BASE}/`)
      .then(r => r.ok ? setApiStatus('online') : setApiStatus('error'))
      .catch(() => setApiStatus('error'))
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
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ q, thread_id: threadId }),
      })

      if (!res.ok) throw new Error(`HTTP ${res.status}`)

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
      setMessages(prev => {
        const updated = [...prev]
        const last = { ...updated[updated.length - 1] }
        last.text = `⚠️ Could not reach the backend API. Make sure the FastAPI server is running on port 8000.\n\nError: ${err.message}`
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
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-brand">
          <div className="header-logo">🧠</div>
          <div>
            <div className="header-title">Second Brain</div>
            <div className="header-subtitle">Agentic RAG · LangGraph</div>
          </div>
        </div>
        <div className="status-pill">
          <div className="status-dot" style={{ background: statusColor[apiStatus] }} />
          {apiStatus === 'online' ? 'API Online' : apiStatus === 'error' ? 'API Offline' : 'Connecting...'}
        </div>
      </header>

      {/* Messages */}
      <main className="messages-area" id="messages-area" role="log" aria-live="polite">
        {messages.length === 0 && !loading ? (
          <div className="empty-state">
            <div className="empty-icon">🧠</div>
            <h1 className="empty-title">Ask your Second Brain</h1>
            <p className="empty-desc">
              Powered by an agentic RAG pipeline — ask anything about your knowledge base
              and get grounded, reasoned answers.
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
            placeholder="Ask anything about your knowledge base…"
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
            disabled={loading}
          />
          <div className="input-actions">
            <input
              id="thread-id-input"
              className="thread-input"
              type="text"
              placeholder="Thread ID"
              value={threadId}
              onChange={e => setThreadId(e.target.value)}
              title="Conversation thread ID (for memory)"
            />
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
            Thread: <kbd>{threadId}</kbd> controls conversation memory
          </span>
          <span>
            <kbd>↵</kbd> send · <kbd>⇧↵</kbd> newline
          </span>
        </div>
      </footer>
    </div>
  )
}
