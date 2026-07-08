import { useState, useEffect, useRef } from 'react';
import { fetchDocuments, uploadDocument, deleteDocument } from '../hooks/useApi';

const FILE_ICONS = {
  pdf: '📕',
  md: '📝',
  txt: '📄',
  docx: '📘',
  html: '🌐',
};

const ALLOWED_EXTS = ['.pdf', '.md', '.txt', '.docx', '.html'];

export default function KnowledgeBase({ onAddToast }) {
  const [documents, setDocuments] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadFileName, setUploadFileName] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);

  useEffect(() => {
    loadDocuments();
  }, []);

  const loadDocuments = async () => {
    try {
      const docs = await fetchDocuments();
      setDocuments(docs);
    } catch (err) {
      onAddToast('Failed to load documents', 'error');
    }
  };

  const handleUpload = async (file) => {
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!ALLOWED_EXTS.includes(ext)) {
      onAddToast(`Unsupported file type: ${ext}`, 'warning');
      return;
    }

    setUploading(true);
    setUploadFileName(file.name);
    setUploadProgress(10);

    // Simulate progress while uploading
    const progressInterval = setInterval(() => {
      setUploadProgress((p) => Math.min(p + 8, 90));
    }, 500);

    try {
      await uploadDocument(file);
      clearInterval(progressInterval);
      setUploadProgress(100);
      onAddToast(`"${file.name}" ingested successfully`, 'success');
      await loadDocuments();
    } catch (err) {
      clearInterval(progressInterval);
      onAddToast(`Ingestion failed: ${err.message}`, 'error');
    } finally {
      setTimeout(() => {
        setUploading(false);
        setUploadProgress(0);
        setUploadFileName('');
      }, 1000);
    }
  };

  const handleDelete = async (name) => {
    if (!confirm(`Delete "${name}" from the knowledge base?`)) return;
    try {
      await deleteDocument(name);
      onAddToast(`"${name}" deleted`, 'info');
      await loadDocuments();
    } catch (err) {
      onAddToast(`Failed to delete: ${err.message}`, 'error');
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) handleUpload(files[0]);
  };

  const handleFileSelect = (e) => {
    const files = Array.from(e.target.files);
    if (files.length > 0) handleUpload(files[0]);
    e.target.value = '';
  };

  return (
    <div className="kb-container">
      <div className="kb-header">
        <h2>📚 Knowledge Base</h2>
        <p>Upload documents to build your searchable knowledge base</p>
      </div>

      <div className="kb-content">
        {/* Upload Zone */}
        <div
          className={`upload-zone ${dragOver ? 'dragover' : ''}`}
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
        >
          <div className="upload-zone-icon">📁</div>
          <div className="upload-zone-text">
            Drag & drop a file here, or <strong>browse</strong>
          </div>
          <div className="upload-zone-formats">
            Supported: PDF, Markdown, Text, DOCX, HTML
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.md,.txt,.docx,.html"
            onChange={handleFileSelect}
            style={{ display: 'none' }}
          />
        </div>

        {/* Upload Progress */}
        {uploading && (
          <div className="upload-progress animate-in">
            <div className="upload-progress-bar">
              <div
                className="upload-progress-fill"
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
            <div className="upload-progress-text">
              {uploadProgress < 100
                ? `Ingesting ${uploadFileName}…`
                : 'Complete!'}
            </div>
          </div>
        )}

        {/* Document List */}
        <div>
          <div className="doc-list-header">
            Indexed Documents ({documents.length})
          </div>

          {documents.length === 0 && (
            <div className="doc-empty">
              <div className="doc-empty-icon">📭</div>
              <div>No documents ingested yet</div>
            </div>
          )}

          {documents.map((doc, i) => {
            const ext = doc.type || doc.name?.split('.').pop() || 'txt';
            return (
              <div key={doc.name || i} className="doc-item">
                <div className="doc-item-icon">
                  {FILE_ICONS[ext] || '📄'}
                </div>
                <div className="doc-item-info">
                  <div className="doc-item-name">{doc.name}</div>
                  <div className="doc-item-meta">
                    <span>{doc.chunks || 0} chunks</span>
                    <span>{doc.episodes || 0} episodes</span>
                    <span>{doc.triples || 0} triples</span>
                  </div>
                </div>
                <span className="doc-item-badge">{doc.status || 'ready'}</span>
                <button
                  className="doc-item-delete"
                  title="Delete document"
                  onClick={() => handleDelete(doc.name)}
                >
                  🗑
                </button>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
