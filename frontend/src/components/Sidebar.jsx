export default function Sidebar({
  sessions,
  activeSessionId,
  activeView,
  isHealthy,
  onSelectSession,
  onNewChat,
  onDeleteSession,
  onNavigate,
}) {
  return (
    <aside className="sidebar">
      {/* Brand */}
      <div className="sidebar-header">
        <div className="sidebar-brand">
          <div className="sidebar-logo">🧠</div>
          <div>
            <div className="sidebar-title">Agentic RAG</div>
            <div className="sidebar-subtitle">Document Intelligence</div>
          </div>
        </div>
        <button className="new-chat-btn" onClick={onNewChat}>
          ＋ New Chat
        </button>
      </div>

      {/* Session list */}
      <div className="session-list">
        <div className="session-group-label">Conversations</div>
        {sessions.length === 0 && (
          <div style={{ padding: '16px 8px', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            No sessions yet. Start a new chat!
          </div>
        )}
        {sessions.map((s) => (
          <div
            key={s.id}
            className={`session-item ${s.id === activeSessionId ? 'active' : ''}`}
            onClick={() => onSelectSession(s.id)}
          >
            <span className="session-item-icon">💬</span>
            <div className="session-item-info">
              <div className="session-item-name">{s.name}</div>
              <div className="session-item-meta">{s.turns || 0} turns</div>
            </div>
            <button
              className="session-item-delete"
              title="Delete session"
              onClick={(e) => {
                e.stopPropagation();
                onDeleteSession(s.id);
              }}
            >
              🗑
            </button>
          </div>
        ))}
      </div>

      {/* Navigation */}
      <nav className="sidebar-nav">
        <div
          className={`sidebar-nav-item ${activeView === 'chat' ? 'active' : ''}`}
          onClick={() => onNavigate('chat')}
        >
          <span>💬</span> Chat
        </div>
        <div
          className={`sidebar-nav-item ${activeView === 'knowledge' ? 'active' : ''}`}
          onClick={() => onNavigate('knowledge')}
        >
          <span>📚</span> Knowledge Base
        </div>
      </nav>

      {/* Status */}
      <div className="sidebar-status">
        <div className={`status-dot ${isHealthy ? '' : 'offline'}`} />
        <span>{isHealthy ? 'All services online' : 'Services offline'}</span>
      </div>
    </aside>
  );
}
