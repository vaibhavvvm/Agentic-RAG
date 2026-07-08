import { useState, useEffect, useCallback } from 'react';
import Sidebar from './components/Sidebar';
import ChatView from './components/ChatView';
import KnowledgeBase from './components/KnowledgeBase';
import ToastContainer, { useToast } from './components/Toast';
import { fetchHealth, createSession, deleteSession } from './hooks/useApi';

export default function App() {
  const [activeView, setActiveView] = useState('chat');
  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [chatHistories, setChatHistories] = useState({}); // { sessionId: [msgs] }
  const [isHealthy, setIsHealthy] = useState(false);
  const { toasts, addToast, removeToast } = useToast();

  // Health check on mount and every 30s
  useEffect(() => {
    const check = async () => {
      try {
        const data = await fetchHealth();
        setIsHealthy(data.status === 'online');
      } catch {
        setIsHealthy(false);
      }
    };
    check();
    const interval = setInterval(check, 30000);
    return () => clearInterval(interval);
  }, []);

  // Auto-create first session on mount
  useEffect(() => {
    if (sessions.length === 0) {
      handleNewChat();
    }
  }, []);

  const handleNewChat = useCallback(async () => {
    try {
      const name = `Chat ${sessions.length + 1}`;
      const data = await createSession(name);
      const newSession = {
        id: data.session_id,
        name: data.name || name,
        turns: 0,
      };
      setSessions((prev) => [newSession, ...prev]);
      setActiveSessionId(data.session_id);
      setChatHistories((prev) => ({ ...prev, [data.session_id]: [] }));
      setActiveView('chat');
    } catch (err) {
      addToast('Failed to create session: ' + err.message, 'error');
    }
  }, [sessions.length, addToast]);

  const handleSelectSession = useCallback((id) => {
    setActiveSessionId(id);
    setActiveView('chat');
  }, []);

  const handleDeleteSession = useCallback(async (id) => {
    try {
      await deleteSession(id);
      setSessions((prev) => prev.filter((s) => s.id !== id));
      setChatHistories((prev) => {
        const copy = { ...prev };
        delete copy[id];
        return copy;
      });
      if (activeSessionId === id) {
        const remaining = sessions.filter((s) => s.id !== id);
        if (remaining.length > 0) {
          setActiveSessionId(remaining[0].id);
        } else {
          setActiveSessionId(null);
          // Auto-create new session
          handleNewChat();
        }
      }
      addToast('Session deleted', 'info');
    } catch (err) {
      addToast('Failed to delete session: ' + err.message, 'error');
    }
  }, [activeSessionId, sessions, addToast, handleNewChat]);

  const handleMessageSent = useCallback((msg) => {
    setChatHistories((prev) => {
      const history = prev[activeSessionId] || [];
      return { ...prev, [activeSessionId]: [...history, msg] };
    });
    // Update turns count
    if (msg.role === 'user') {
      setSessions((prev) =>
        prev.map((s) =>
          s.id === activeSessionId ? { ...s, turns: (s.turns || 0) + 1 } : s
        )
      );
    }
  }, [activeSessionId]);

  const currentHistory = chatHistories[activeSessionId] || [];

  return (
    <>
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        activeView={activeView}
        isHealthy={isHealthy}
        onSelectSession={handleSelectSession}
        onNewChat={handleNewChat}
        onDeleteSession={handleDeleteSession}
        onNavigate={setActiveView}
      />

      <main className="main-content">
        {activeView === 'chat' && (
          <ChatView
            sessionId={activeSessionId}
            chatHistory={currentHistory}
            onMessageSent={handleMessageSent}
            onAddToast={addToast}
          />
        )}
        {activeView === 'knowledge' && (
          <KnowledgeBase onAddToast={addToast} />
        )}
      </main>

      <ToastContainer toasts={toasts} onRemove={removeToast} />
    </>
  );
}
