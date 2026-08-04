/* PdfUploader.jsx — handles PDF file selection and upload */
import { uploadPDF } from './api.js'

export default function PdfUploader({ file, onFileChange, upload, setUpload, status, setStatus, setError, isBusy, chatId }) {
  async function handleUpload() {
    if (!file) return
    try {
      setError('')
      setStatus('uploading')
      const result = await uploadPDF(file, chatId)
      setUpload(result)
    } catch (err) {
      setError(err.message || 'Upload failed')
    } finally {
      setStatus('idle')
    }
  }

  return (
    <section className="card">
      <h2>Upload PDF</h2>
      <label htmlFor="pdf-file">Choose a PDF file</label>
      <input
        id="pdf-file"
        type="file"
        accept=".pdf"
        onChange={(e) => onFileChange(e.target.files[0])}
      />
      <button
        type="button"
        disabled={!file || isBusy}
        onClick={handleUpload}
      >
        {status === 'uploading' ? 'Uploading…' : 'Upload'}
      </button>
      {upload && (
        <p className="upload-info">Uploaded: {upload.filename} ({upload.pages} pages, {upload.characters} characters)</p>
      )}
    </section>
  )
}
