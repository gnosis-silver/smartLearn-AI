/* ChatPanel.jsx — handles message input and chat with AI */
import { useState } from 'react'
import { askQuestion } from './api.js'

export default function ChatPanel() {
  const [message, setMessage] = useState('')    // controlled chat input
  const [answer, setAnswer] = useState(null)    // /chat response
  const [status, setStatus] = useState('idle')  // idle | asking
  const [error, setError] = useState('')        // visible error text

  async function handleAsk(event) {
    event.preventDefault()
    const trimmed = message.trim()
    if (!trimmed) return
    try {
      setError('')
      setStatus('asking')
      const result = await askQuestion(trimmed)
      setAnswer(result)
    } catch (err) {
      setError(err.message || 'Chat failed')
    } finally {
      setStatus('idle')
    }
  }

  return (
    <section className="card">
      <h2>Ask a Question</h2>
      <form onSubmit={handleAsk}>
        <label htmlFor="message">Message</label>
        <textarea
          id="message"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          rows={3}
        />
        <button
          type="submit"
          disabled={!message.trim() || status !== 'idle'}
        >
          {status === 'asking' ? 'Asking…' : 'Ask'}
        </button>
      </form>

      {error && <p role="alert" className="error">{error}</p>}

      {answer && (
        <div>
          <h3>Answer</h3>
          <p>{answer.answer}</p>
          {answer.citations && answer.citations.length > 0 && (
            <div className="chips">
              {answer.citations.map((page) => (
                <span key={page} className="chip">Page {page}</span>
              ))}
            </div>
          )}
        </div>
      )}
    </section>
  )
}
