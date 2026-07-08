import { useState, useRef, useEffect } from 'react';
import SourceCard from './SourceCard';
import { sendChat } from '../hooks/useApi';

const SUGGESTIONS = [
  '📖 What components are in the ingestion pipeline?',
  '🔗 How does hybrid search work?',
  '📊 What evaluation metrics are supported?',
  '🧠 Explain the memory system architecture',
];

function formatMarkdown(text) {
  if (!text) return '';
  let html = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  // Code blocks
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  // Bold
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // Italic
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
  // Newlines to <br> (but not inside <pre>)
  html = html.replace(/\n/g, '<br/>');
  // Bullet lists
  html = html.replace(/((?:<br\/>)?- .+(?:<br\/>)?)+/g, (match) => {
    const items = match
      .split('<br/>')
      .filter((l) => l.trim().startsWith('- '))
      .map((l) => `<li>${l.trim().slice(2)}</li>`)
      .join('');
    return `<ul>${items}</ul>`;
  });
  // Numbered lists
  html = html.replace(/((?:<br\/>)?\d+\. .+(?:<br\/>)?)+/g, (match) => {
    const items = match
      .split('<br/>')
      .filter((l) => /^\d+\.\s/.test(l.trim()))
      .map((l) => `<li>${l.trim().replace(/^\d+\.\s/, '')}</li>`)
      .join('');
    return `<ol>${items}</ol>`;
  });

  return html;
}

function getRoutingTag(metadata) {
  if (!metadata) return null;
  const intent = metadata.intent || '';
  const tier = metadata.router_tier || metadata.tier || '';
  if (intent.includes('vector') || intent === 'vector') return 'vector';
  if (intent.includes('graph') || intent === 'graph') return 'graph';
  if (intent.includes('hybrid') || intent === 'hybrid') return 'hybrid';
  if (intent.includes('general') || intent === 'general') return 'general';
  if (tier) return tier;
  return null;
}

export default function ChatView({ sessionId, chatHistory, onMessageSent, onAddToast }) {
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    if (messagesRef.current) {
      messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
    }
  }, [chatHistory, isTyping]);

  const handleSend = async () => {
    const text = inputText.trim();
    if (!text || !sessionId) return;

    setInputText('');
    if (textareaRef.current) textareaRef.current.style.height = 'auto';

    // Add user message immediately
    onMessageSent({ role: 'user', content: text });

    setIsTyping(true);
    try {
      const data = await sendChat(sessionId, text);
      const answer = data.answer || '_(no answer)_';
      const metadata = data.metadata || {};
      const sources = data.sources || [];
      onMessageSent({
        role: 'assistant',
        content: answer,
        meta: { ...metadata, sources },
      });
    } catch (err) {
      onAddToast('Failed to get response: ' + err.message, 'error');
      onMessageSent({
        role: 'assistant',
        content: 'Sorry, I could not connect to the RAG server. Please check if the backend is running.',
      });
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleTextareaInput = (e) => {
    setInputText(e.target.value);
    e.target.style.height = 'auto';
    e.target.style.height = Math.min(e.target.scrollHeight, 150) + 'px';
  };

  const useSuggestion = (text) => {
    const cleaned = text.replace(/^[^\s]+\s/, '');
    setInputText(cleaned);
    // Auto-send after a tiny delay for UX
    setTimeout(() => {
      onMessageSent({ role: 'user', content: cleaned });
      setIsTyping(true);
      sendChat(sessionId, cleaned)
        .then((data) => {
          onMessageSent({
            role: 'assistant',
            content: data.answer || '_(no answer)_',
            meta: { ...(data.metadata || {}), sources: data.sources || [] },
          });
        })
        .catch((err) => {
          onAddToast('Failed to get response', 'error');
          onMessageSent({
            role: 'assistant',
            content: 'Sorry, I could not connect to the backend.',
          });
        })
        .finally(() => {
          setIsTyping(false);
          setInputText('');
        });
    }, 50);
  };

  const hasMessages = chatHistory && chatHistory.length > 0;

  return (
    <div className="chat-container">
      {!hasMessages && (
        <div className="chat-welcome">
          <div className="welcome-icon">🧠</div>
          <div className="welcome-title">Agentic RAG</div>
          <div className="welcome-subtitle">
            Ask questions about your documents. I use vector search, graph
            reasoning, and conversation memory to give you accurate answers.
          </div>
          <div className="suggestion-chips">
            {SUGGESTIONS.map((s, i) => (
              <button
                key={i}
                className="suggestion-chip"
                onClick={() => useSuggestion(s)}
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      )}

      {hasMessages && (
        <div className="chat-messages" ref={messagesRef}>
          {chatHistory.map((msg, i) => (
            <Message key={i} msg={msg} />
          ))}
        </div>
      )}

      {/* Typing indicator */}
      <div className={`typing-indicator ${isTyping ? 'visible' : ''}`}>
        <div style={{
          width: 36, height: 36, borderRadius: 'var(--radius-sm)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: '1rem', background: 'var(--bg-card)', border: '1px solid var(--border-subtle)'
        }}>
          🧠
        </div>
        <div className="typing-dots">
          <span /><span /><span />
        </div>
      </div>

      {/* Input */}
      <div className="chat-input-area">
        <div className="chat-input-wrapper">
          <textarea
            ref={textareaRef}
            value={inputText}
            onChange={handleTextareaInput}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your documents…"
            rows={1}
          />
          <button
            className="send-btn"
            onClick={handleSend}
            disabled={!inputText.trim() || isTyping}
          >
            ▲
          </button>
        </div>
      </div>
    </div>
  );
}

function Message({ msg }) {
  const { role, content, meta } = msg;
  const isUser = role === 'user';
  const tag = isUser ? null : getRoutingTag(meta);
  const sources = meta?.sources || [];
  const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const [sourcesOpen, setSourcesOpen] = useState(false);

  return (
    <div className={`message ${role} animate-in`}>
      <div className="message-avatar">{isUser ? '👤' : '🧠'}</div>
      <div className="message-body">
        <div
          className="message-content"
          dangerouslySetInnerHTML={{ __html: formatMarkdown(content) }}
        />

        {/* Sources */}
        {sources.length > 0 && (
          <div className="sources-section">
            <button
              className={`sources-toggle ${sourcesOpen ? 'open' : ''}`}
              onClick={() => setSourcesOpen(!sourcesOpen)}
            >
              <span>📎 {sources.length} source{sources.length > 1 ? 's' : ''} referenced</span>
              <span className="chevron">▾</span>
            </button>
            <div className={`sources-list ${sourcesOpen ? 'open' : ''}`}>
              {sources.map((s, i) => (
                <SourceCard key={i} source={s} />
              ))}
            </div>
          </div>
        )}

        <div className="message-meta">
          <span>{time}</span>
          {tag && <span className={`routing-tag ${tag}`}>{tag}</span>}
        </div>
      </div>
    </div>
  );
}
