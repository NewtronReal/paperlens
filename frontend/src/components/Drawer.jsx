import { useEffect } from "react";
import { Badge, ExtLink } from "./bits.jsx";
import { fmt, intentColor, num } from "../lib/format.js";

function IntentProbs({ probs }) {
  return (
    <>
      <div className="sub">Model 1: intent probabilities (mean over sentences)</div>
      <div className="probs">
        {["background", "method", "result"].map((k) => [
          <span key={`${k}l`}>{k}</span>,
          <div key={`${k}t`} className="track">
            <div className="fill" style={{ width: `${Math.round(probs[k] * 100)}%`, background: intentColor(k) }} />
          </div>,
          <span key={`${k}v`}>{fmt(probs[k])}</span>,
        ])}
      </div>
    </>
  );
}

/** Side panel with everything known about one paper; `onAnalyze(query)` re-runs the app on it. */
export default function Drawer({ paperId, result, byId, onClose, onAnalyze }) {
  useEffect(() => {
    const onKey = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!paperId) return null;
  const r = byId.get(paperId);
  const ln = result.lineage.nodes.find((n) => n.paperId === paperId);
  const evo = result.evolution.nodes?.find((n) => n.paperId === paperId);
  const base = r ?? ln ?? evo;
  if (!base) return null;

  const isFocus = paperId === result.paper.paperId;
  const role = isFocus ? "this paper" : r ? (r.role === "reference" ? "cited by this paper" : "cites this paper") : "ancestor (lineage)";
  const analyzeLink = (q) => (
    <a href="#" onClick={(e) => { e.preventDefault(); onAnalyze(q); }}>Analyse this paper →</a>
  );

  return (
    <aside className="drawer">
      <button className="close" onClick={onClose} aria-label="Close">×</button>
      <div className="meta-line">{role}</div>
      <h2>{base.title}</h2>
      <div className="pm">
        {[base.year, r?.venue].filter(Boolean).join(" · ")}
        {r?.authors?.length ? ` · ${r.authors.join(", ")}` : ""}
      </div>

      {r && (
        <>
          <p>
            <Badge intent={r.intent} /> &nbsp;Influence <b>{fmt(r.influence)}</b>
            {r.s2_influential && " · S2 marks influential"} · {num(r.citationCount)} citations
          </p>
          {r.intent_probs && <IntentProbs probs={r.intent_probs} />}
          <div className="sub">Citation sentences{r.context_source === "pdf" && " (recovered from your PDF)"}</div>
          {r.contexts.length ? r.contexts.map((c, i) => (
            <div key={i} className={`ctx ${c.intent}`}>
              {c.text}
              <div className="pm"><Badge intent={c.intent} /> confidence {fmt(c.confidence)}</div>
            </div>
          )) : (
            <p className="muted">Semantic Scholar has no citation sentence for this edge. Upload the PDF to recover them.</p>
          )}
          <p className="links">
            {r.arxiv && <ExtLink href={`https://arxiv.org/abs/${r.arxiv}`}>arXiv</ExtLink>}
            {!result.demo && <ExtLink href={`https://www.semanticscholar.org/paper/${r.paperId}`}>Semantic Scholar</ExtLink>}
            {analyzeLink(result.demo ? r.paperId : r.arxiv || r.paperId)}
          </p>
        </>
      )}
      {!r && ln?.level > 0 && (
        <>
          <p><Badge intent={ln.intent} /> Lineage level {ln.level} · influence <b>{fmt(ln.influence)}</b></p>
          <p>{analyzeLink(paperId)}</p>
        </>
      )}
    </aside>
  );
}
