const API_BASE = '';

async function apiRequest(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export async function fetchHealth() {
  return apiRequest('/api/health');
}

export async function fetchSessions() {
  return apiRequest('/api/sessions');
}

export async function createSession(name) {
  return apiRequest('/api/sessions', {
    method: 'POST',
    body: JSON.stringify({ name }),
  });
}

export async function deleteSession(id) {
  return apiRequest(`/api/sessions/${id}`, { method: 'DELETE' });
}

export async function sendChat(sessionId, query) {
  return apiRequest('/api/chat', {
    method: 'POST',
    body: JSON.stringify({ session_id: sessionId, query }),
  });
}

export async function fetchDocuments() {
  return apiRequest('/api/documents');
}

export async function uploadDocument(file) {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE}/api/ingest`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export async function deleteDocument(name) {
  return apiRequest(`/api/documents/${encodeURIComponent(name)}`, {
    method: 'DELETE',
  });
}
