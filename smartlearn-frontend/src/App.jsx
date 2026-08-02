/* App.jsx — the main component: all visible UI starts here */
import { useState } from 'react'
import { uploadPDF, askQuestion } from './api.js'

export default function App() {
  const [file, setFile] = useState(null)       // selected PDF File object
  const [upload, setUpload] = useState(null)    // successful upload metadata
  const [message, setMessage] = useState('')    // controlled chat input
  const [answer, setAnswer] = useState(null)    // /chat response { answer, citations }
  const [status, setStatus] = useState('idle')  // idle | uploading | asking
  const [error, setError] = useState('')        // visible error text

  const isBusy = status !== 'idle'

  async function handleUpload() {
    if (!file) return
    try {
      setError('')
      setStatus('uploading')
      const result = await uploadPDF(file)
      setUpload(result)
    } catch (err) {
      setError(err.message || 'Upload failed')
    } finally {
      setStatus('idle')
    }
  }

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
    <main>
      <h1>SmartLearn Lite</h1>

      {/* ── Upload section ── */}
      <section>
        <h2>Upload PDF</h2>
        <label htmlFor="pdf-file">Choose a PDF file</label>
        <input
          id="pdf-file"
          type="file"
          accept=".pdf"
          onChange={(e) => setFile(e.target.files[0])}
        />
        <button
          type="button"
          disabled={!file || isBusy}
          onClick={handleUpload}
        >
          {status === 'uploading' ? 'Uploading…' : 'Upload'}
        </button>
        {upload && (
          <p>Uploaded: {upload.filename} ({upload.pages} pages, {upload.characters} characters)</p>
        )}
      </section>

      {/* ── Chat section (only visible after upload) ── */}
      {upload && (
        <section>
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
              disabled={!message.trim() || isBusy}
            >
              {status === 'asking' ? 'Asking…' : 'Ask'}
            </button>
          </form>

          {error && <p role="alert">{error}</p>}

          {answer && (
            <div>
              <h3>Answer</h3>
              <p>{answer.answer}</p>
              {answer.citations && answer.citations.length > 0 && (
                <div>
                  {answer.citations.map((page) => (
                    <span key={page}>Page {page}</span>
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
