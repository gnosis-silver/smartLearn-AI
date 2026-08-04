/* App.jsx — coordinates upload, PDF preview, and multi-turn chat */
import { useState } from 'react'
import PdfUploader from './PdfUploader.jsx'
import PdfPreview from './PdfPreview.jsx'
import ChatPanel from './ChatPanel.jsx'

export default function App() {
  const [file, setFile] = useState(null)
  const [upload, setUpload] = useState(null)
  const [uploadStatus, setUploadStatus] = useState('idle')
  const [error, setError] = useState('')

  /* ── shared state ── */
  const [activePage, setActivePage] = useState(1)
  const [uploadKey, setUploadKey] = useState(0)

  const isBusy = uploadStatus !== 'idle'

  function handleJumpToPage(page) {
    setActivePage(page)
  }

  /* ── PdfUploader setUpload wrapper: also reset page + bump remount key ── */
  function handleSetUpload(result) {
    setUpload(result)
    setActivePage(1)
    setUploadKey(prev => prev + 1)
  }

  return (
    <main>
      <h1>SmartLearn Lite</h1>

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
          />
          <ChatPanel
            enabled={!!upload}
            onJumpToPage={handleJumpToPage}
            uploadKey={uploadKey}
          />
        </div>
        <div className="right-col">
          <PdfPreview upload={upload} activePage={activePage} uploadKey={uploadKey} />
        </div>
      </div>
    </main>
  )
}
