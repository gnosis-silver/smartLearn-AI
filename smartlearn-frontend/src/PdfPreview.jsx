/* PdfPreview.jsx — shows the uploaded PDF and jumps to a cited page */
import { API } from './api.js'

export const CHAT_ID = 'day2-demo'

export function getDocumentFileURL(chatId, page = 1) {
  return `${API}/documents/${encodeURIComponent(chatId)}/file#page=${page}`
}

export default function PdfPreview({ upload, activePage, uploadKey }) {
  if (!upload) {
    return (
      <section className="card preview-placeholder">
        <p>Upload a PDF to preview it here.</p>
      </section>
    )
  }

  const chatId = upload.chat_id || CHAT_ID
  const url = getDocumentFileURL(chatId, activePage || 1)

  return (
    <section className="card preview-panel">
      <div className="preview-header">
        <span className="preview-label">
          {upload.filename} — page {activePage || 1}
        </span>
      </div>
      <iframe
        key={`${uploadKey}-${activePage}`}
        src={url}
        className="preview-frame"
        title="PDF Preview"
      />
    </section>
  )
}
