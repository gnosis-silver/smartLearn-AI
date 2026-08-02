/* App.jsx — the main component: orchestrates upload + chat */
import { useState } from 'react'
import PdfUploader from './PdfUploader.jsx'
import ChatPanel from './ChatPanel.jsx'

export default function App() {
  const [file, setFile] = useState(null)       // selected PDF File object
  const [upload, setUpload] = useState(null)    // successful upload metadata
  const [status, setStatus] = useState('idle')  // idle | uploading (global for PdfUploader)
  const [error, setError] = useState('')        // visible error text

  const isBusy = status !== 'idle'

  return (
    <main>
      <h1>SmartLearn Lite</h1>

      <PdfUploader
        file={file}
        onFileChange={setFile}
        upload={upload}
        setUpload={setUpload}
        status={status}
        setStatus={setStatus}
        setError={setError}
        isBusy={isBusy}
      />

      {upload && <ChatPanel />}
    </main>
  )
}
