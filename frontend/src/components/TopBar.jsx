import { useRef } from "react";

function ModelChips({ health }) {
  if (!health) return null;
  const untrained = (name) => /heuristic/.test(name);
  return (
    <div className="chips">
      {health.demo && <span className="chip warn">demo data</span>}
      <span className={`chip ${untrained(health.models.intent) ? "warn" : "ok"}`} title={health.models.intent}>
        M1 intent: {untrained(health.models.intent) ? "untrained" : "trained"}
      </span>
      <span className={`chip ${untrained(health.models.influence) ? "warn" : "ok"}`} title={health.models.influence}>
        M2 influence: {untrained(health.models.influence) ? "untrained" : "trained"}
      </span>
      {!health.demo && !health.s2_key && (
        <span className="chip warn" title="Set S2_API_KEY for reliable access">no S2 key</span>
      )}
      {health.llm && <span className="chip ok">LLM summary on</span>}
    </div>
  );
}

export default function TopBar({ query, setQuery, file, setFile, busy, onSubmit, health }) {
  const fileInput = useRef(null);
  return (
    <header className="topbar">
      <div className="brand"><span className="logo">◎</span> PaperLens</div>
      <form className="search" autoComplete="off" onSubmit={(e) => { e.preventDefault(); onSubmit(); }}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="arXiv ID or URL, DOI, Semantic Scholar link, or paper title"
        />
        <label className={`upload ${file ? "has" : ""}`} title={file ? "Click to change the PDF" : "Upload a PDF"}>
          <input
            ref={fileInput}
            type="file"
            accept="application/pdf"
            hidden
            onChange={(e) => setFile(e.target.files[0] ?? null)}
          />
          <span>{file ? file.name : "PDF"}</span>
        </label>
        {file && (
          <button type="button" className="clear" title="Remove PDF"
            onClick={() => { setFile(null); fileInput.current.value = ""; }}>×</button>
        )}
        <button type="submit" id="go" disabled={busy}>{busy ? "Analyzing…" : "Analyze"}</button>
      </form>
      <ModelChips health={health} />
    </header>
  );
}
