export default function SourceCard({ source }) {
  const name = source.source || source.kind || 'unknown';
  const snippet = source.snippet || '';
  const score = source.score;

  return (
    <div className="source-card">
      <div className="source-card-header">
        <span className="source-card-name">📄 {name}</span>
        {score != null && (
          <span className="source-card-score">
            {(typeof score === 'number' ? score : parseFloat(score) || 0).toFixed(2)}
          </span>
        )}
      </div>
      {snippet && <div className="source-card-snippet">{snippet}</div>}
    </div>
  );
}
