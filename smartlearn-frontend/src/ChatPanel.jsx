/* ChatPanel.jsx — multi-turn chat with citation buttons and page-jump callback */
import { useState, useEffect, useRef } from 'react'
import { marked } from 'marked'
import katex from 'katex'
import 'katex/dist/katex.min.css'
import { askQuestion } from './api.js'

/* ── Render Markdown + LaTeX ── */
function renderContent(text) {
  const blocks = []

  // 1. Protect display math: \[...\] or $$...$$
  let protected_ = text
    .replace(/(^|\n)\s*\\\[([\s\S]*?)\\\]\s*(?=\n|$)/g, (_, nl, formula) => {
      blocks.push({ type: 'display', formula: formula.trim() })
      return `${nl}%%LATEX${blocks.length - 1}%%`
    })
    .replace(/(^|\n)\s*\$\$([\s\S]*?)\$\$\s*(?=\n|$)/g, (_, nl, formula) => {
      blocks.push({ type: 'display', formula: formula.trim() })
      return `${nl}%%LATEX${blocks.length - 1}%%`
    })

  // 2. Protect inline math: \(...\) or $...$
  protected_ = protected_
    .replace(/\\\(([\s\S]*?)\\\)/g, (_, formula) => {
      blocks.push({ type: 'inline', formula: formula.trim() })
      return `%%LATEX${blocks.length - 1}%%`
    })
    .replace(/\$((?!\$)[\s\S]*?[^$])\$(?!\$)/g, (_, formula) => {
      blocks.push({ type: 'inline', formula: formula.trim() })
      return `%%LATEX${blocks.length - 1}%%`
    })

  // 3. Run markdown
  let html = marked(protected_)

  // 4. Restore math blocks as KaTeX
  html = html.replace(/%%LATEX(\d+)%%/g, (_, idx) => {
    const block = blocks[parseInt(idx)]
    try {
      return katex.renderToString(block.formula, {
        displayMode: block.type === 'display',
        throwOnError: false,
      })
    } catch {
      return block.formula
    }
  })

  return html
}

export default function ChatPanel({ enabled, onJumpToPage, uploadKey, chatId, initialMessages }) {
  const [message, setMessage] = useState('')
  const [messages, setMessages] = useState([])
  const [status, setStatus] = useState('idle')
  const [error, setError] = useState('')
  const [expanded, setExpanded] = useState(false)
  const listEndRef = useRef(null)

  /* ── clear message list when a new upload remounts ── */
  useEffect(() => {
    setMessages([])
    setError('')
  }, [uploadKey])

  /* ── load history when session is restored ── */
  useEffect(() => {
    if (initialMessages && initialMessages.length > 0) {
      setMessages(initialMessages)
    }
  }, [initialMessages])

  /* ── auto-scroll to latest message ── */
  useEffect(() => {
    listEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function handleAsk(event) {
    event.preventDefault()
    const trimmed = message.trim()
    if (!trimmed) return
    try {
      setError('')
      setStatus('asking')

      const userMsg = { role: 'user', content: trimmed }
      setMessages(prev => [...prev, userMsg])
      setMessage('')

      const result = await askQuestion(trimmed, chatId)

      const assistantMsg = {
        role: 'assistant',
        content: result.answer,
        citations: result.citations || [],
        sources: result.sources || [],
      }
      setMessages(prev => [...prev, assistantMsg])
    } catch (err) {
      setError(err.message || 'Chat failed')
    } finally {
      setStatus('idle')
    }
  }

  function renderMessage(msg, idx) {
    const isUser = msg.role === 'user'
    return (
      <div key={idx} className={`msg ${isUser ? 'msg-user' : 'msg-assistant'}`}>
        <div className="msg-role">{isUser ? 'You' : 'Assistant'}</div>
        <div
          className="msg-content markdown-body"
          dangerouslySetInnerHTML={{ __html: renderContent(msg.content) }}
        />
        {!isUser && msg.citations && msg.citations.length > 0 && (
          <div className="chips">
            {msg.citations.map((page) => (
              <button
                key={page}
                type="button"
                className="chip chip-btn"
                onClick={() => onJumpToPage && onJumpToPage(page)}
              >
                Page {page}
              </button>
            ))}
          </div>
        )}
      </div>
    )
  }

  /* ── shared chat body ── */
  function chatBody() {
    return (
      <>
        <div className="chat-messages">
          {messages.length === 0 && (
            <p className="chat-placeholder">Ask a question about the uploaded PDF.</p>
          )}
          {messages.map(renderMessage)}
          <div ref={listEndRef} />
        </div>

        {error && <p role="alert" className="error">{error}</p>}

        <form onSubmit={handleAsk} className="chat-form">
          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            rows={2}
            placeholder="Type your question…"
          />
          <button
            type="submit"
            disabled={!message.trim() || status !== 'idle'}
          >
            {status === 'asking' ? 'Asking…' : 'Send'}
          </button>
        </form>
      </>
    )
  }

  if (!enabled) {
    return (
      <section className="card chat-panel">
        <p className="chat-placeholder">Upload a PDF to start asking questions.</p>
      </section>
    )
  }

  return (
    <>
      {/* ── Inline chat ── */}
      <section className="card chat-panel">
        <div className="chat-header">
          <h2>Chat</h2>
          <button
            type="button"
            className="chat-expand-btn"
            onClick={() => setExpanded(true)}
            title="Expand chat"
          >
            ⛶
          </button>
        </div>
        {chatBody()}
      </section>

      {/* ── Floating overlay ── */}
      {expanded && (
        <div className="chat-overlay" onClick={() => setExpanded(false)}>
          <div className="chat-overlay-panel" onClick={(e) => e.stopPropagation()}>
            <div className="chat-header">
              <h2>Chat</h2>
              <button
                type="button"
                className="chat-expand-btn"
                onClick={() => setExpanded(false)}
                title="Minimize"
              >
                ✕
              </button>
            </div>
            {chatBody()}
          </div>
        </div>
      )}
    </>
  )
}
