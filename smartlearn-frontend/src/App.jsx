/* App.jsx — coordinates upload, PDF preview, and multi-turn chat */
import { useState } from 'react'
import PdfUploader from './PdfUploader.jsx'
import PdfPreview from './PdfPreview.jsx'
import ChatPanel from './ChatPanel.jsx'
import { CHAT_ID, loadSession } from './api.js'

export default function App() {
  const [chatId, setChatId] = useState(CHAT_ID)
  const [file, setFile] = useState(null)
  const [upload, setUpload] = useState(null)
  const [uploadStatus, setUploadStatus] = useState('idle')
  const [error, setError] = useState('')
  const [sessionLoading, setSessionLoading] = useState(false)

  /* ── shared state ── */
  const [activePage, setActivePage] = useState(1)
  const [uploadKey, setUploadKey] = useState(0)
  const [initialMessages, setInitialMessages] = useState(null)

  const isBusy = uploadStatus !== 'idle' || sessionLoading

  function handleJumpToPage(page) {
    setActivePage(page)
  }

  function handleSetUpload(result) {
    setUpload(result)
    setActivePage(1)
    setUploadKey(prev => prev + 1)
    setInitialMessages(null)  // clear loaded history on fresh upload
  }

  /* ── load existing session ── */
  async function handleLoadSession() {
    if (!chatId) return
    try {
      setError('')
      setSessionLoading(true)
      const session = await loadSession(chatId)
      // Restore upload state so PdfPreview can show the PDF
      setUpload({
        ...session,
        chat_id: chatId,
      })
      setActivePage(1)
      setUploadKey(prev => prev + 1)
      // Restore chat history
      const msgs = []
      for (const turn of (session.history || [])) {
        msgs.push({ role: 'user', content: turn.question })
        msgs.push({
          role: 'assistant',
          content: turn.answer,
          citations: turn.citations || [],
          sources: [],
        })
      }
      setInitialMessages(msgs)
    } catch (err) {
      setError(err.message || 'Session not found')
    } finally {
      setSessionLoading(false)
    }
  }

  return (
    <main>
      <div className="top-bar">
        <h1>SmartLearn Lite</h1>
        <label className="chat-id-label">
          Session
          <input
            type="text"
            className="chat-id-input"
            value={chatId}
            onChange={(e) => setChatId(e.target.value.trim() || CHAT_ID)}
          />
          <button
            type="button"
            className="session-confirm-btn"
            onClick={handleLoadSession}
            disabled={isBusy || !chatId}
          >
            {sessionLoading ? 'Loading…' : 'Confirm'}
          </button>
        </label>
      </div>

      {error && <p role="alert" className="error">{error}</p>}

      <div className="workspace">
        <div className="left-col">
          <PdfUploader
            file={file}
            onFileChange={setFile}
            upload={upload}
            setUpload={handleSetUpload}
            status={uploadStatus}
            setStatus={setUploadStatus}
            setError={setError}
            isBusy={isBusy}
            chatId={chatId}
          />
          <ChatPanel
            enabled={!!upload}
            onJumpToPage={handleJumpToPage}
            uploadKey={uploadKey}
            chatId={chatId}
            initialMessages={initialMessages}
          />
        </div>
        <div className="right-col">
          <PdfPreview upload={upload} activePage={activePage} uploadKey={uploadKey} />
        </div>
      </div>
    </main>
  )
}
