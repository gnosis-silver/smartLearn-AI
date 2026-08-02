/* App.jsx — the main component: orchestrates upload + chat with Markdown + LaTeX rendering */
import { useState } from 'react'
import { marked } from 'marked'
import katex from 'katex'
import 'katex/dist/katex.min.css'
import PdfUploader from './PdfUploader.jsx'
import { askQuestion } from './api.js'

/* ── Render Markdown + LaTeX: protect math → markdown → restore math ── */
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

export default function App() {
  const [file, setFile] = useState(null)        // selected PDF File object
  const [upload, setUpload] = useState(null)     // successful upload metadata
  const [uploadStatus, setUploadStatus] = useState('idle')  // idle | uploading
  const [error, setError] = useState('')         // visible error text

  /* ── Chat state ── */
  const [message, setMessage] = useState('')     // controlled chat input
  const [answer, setAnswer] = useState(null)     // /chat response
  const [chatStatus, setChatStatus] = useState('idle')  // idle | asking

  const isBusy = uploadStatus !== 'idle'

  /* ── Chat handler ── */
  async function handleAsk(event) {
    event.preventDefault()
    const trimmed = message.trim()
    if (!trimmed) return
    try {
      setError('')
      setChatStatus('asking')
      const result = await askQuestion(trimmed)
      setAnswer(result)
    } catch (err) {
      setError(err.message || 'Chat failed')
    } finally {
      setChatStatus('idle')
    }
  }

  return (
    <main>
      <h1>SmartLearn Lite</h1>

      <PdfUploader
        file={file}
        onFileChange={setFile}
        upload={upload}
        setUpload={setUpload}
        status={uploadStatus}
        setStatus={setUploadStatus}
        setError={setError}
        isBusy={isBusy}
      />

      {upload && (
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
              disabled={!message.trim() || chatStatus !== 'idle'}
            >
              {chatStatus === 'asking' ? 'Asking…' : 'Ask'}
            </button>
          </form>

          {error && <p role="alert" className="error">{error}</p>}

          {answer && (
            <div>
              <h3>Answer</h3>
              <div
                className="markdown-body"
                dangerouslySetInnerHTML={{ __html: renderContent(answer.answer) }}
              />
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
      )}
    </main>
  )
}
