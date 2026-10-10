import { Badge, ExtLink, Meter, Swatch } from "./bits.jsx";
import { INTENTS, INTENT_TEXT, fmt, intentColor, num } from "../lib/format.js";

function Summary({ summary }) {
  const rows = [["TL;DR", summary?.tldr, "tldr"], ["Problem", summary?.problem], ["Approach", summary?.approach],
    ["Results", summary?.results], ["Limitations", summary?.limitations], ["Read if", summary?.read_if]].filter((r) => r[1]);
  return (
    <div className="card">
      <h3>At a glance <span className="src">{summary?.source}</span></h3>
      <dl className="summary">
        {rows.length ? rows.map(([k, v, cls]) => [<dt key={`${k}t`}>{k}</dt>, <dd key={`${k}d`} className={cls}>{v}</dd>])
          : <dd className="muted">No abstract available.</dd>}
      </dl>
    </div>
  );
}

function Storyline({ storyline }) {
  return (
    <div className="card">
      <h3>Where the idea comes from and where it went</h3>
      {storyline?.length ? (
        <div className="storyline">
          {storyline.map((c) => (
            <div key={c.span + c.label} className={`story ${c.focus ? "focus" : ""}`}>
              <div className="span">{c.span}</div>
              <div className="dot" />
              <div>
                <div className="lab">{c.label}{c.focus && <> <span className="tag">this paper</span></>}</div>
                <div className="kp">key paper: {c.key_paper}</div>
              </div>
            </div>
          ))}
        </div>
      ) : <p className="muted">Not enough connected papers to map.</p>}
    </div>
  );
}

function IntentBreakdown({ counts }) {
  const total = Object.values(counts).reduce((a, b) => a + b, 0) || 1;
  return (
    <div className="card">
      <h3>Why the references are cited <span className="src">Model 1</span></h3>
      <div className="intentbar">
        {INTENTS.filter((i) => counts[i]).map((i) => (
          <div key={i} style={{ flex: counts[i], background: intentColor(i) }} title={`${INTENT_TEXT[i]}: ${counts[i]}`}>
            {Math.round((100 * counts[i]) / total)}%
          </div>
        ))}
      </div>
      <div className="legend">
        {INTENTS.map((i) => <span key={i}><Swatch color={intentColor(i)} />{INTENT_TEXT[i]} ({counts[i] || 0})</span>)}
      </div>
    </div>
  );
}

function PaperList({ title, src, items, empty, onOpen }) {
  return (
    <div className="card">
      <h3>{title} {src && <span className="src">{src}</span>}</h3>
      {items.length ? (
        <ol className="plist">
          {items.map((r) => (
            <li key={r.paperId} onClick={() => onOpen(r.paperId)}>
              <div>
                <div className="pt">{r.title}</div>
                <div className="pm">{r.year} · <Badge intent={r.intent} /> · {r.why}</div>
              </div>
              <Meter value={r.influence} />
            </li>
          ))}
        </ol>
      ) : <p className="muted">{empty}</p>}
    </div>
  );
}

function RepoRow({ r }) {
  const label = r.source === "paper" ? "linked in paper" : r.source === "search" ? "found by search" : r.source;
  const tip = r.source === "paper" ? "Link written in the paper" : r.source === "search" ? "README mentions the arXiv ID" : "synthetic";
  return (
    <div className="repo">
      <div>
        {r.url ? <ExtLink href={r.url}>{r.name || r.url}</ExtLink> : r.name}
        {r.description && <div className="d">{r.description}</div>}
      </div>
      <div>
        {r.stars !== null && r.stars !== undefined && `★ ${num(r.stars)} `}
        <span className="tag" title={tip}>{label}</span>
      </div>
    </div>
  );
}

function Code({ repos }) {
  const rp = repos ?? { paper: [], references: [] };
  return (
    <div className="card">
      <h3>Code</h3>
      {rp.paper.length ? rp.paper.map((r) => <RepoRow key={r.name || r.url} r={r} />)
        : <p className="muted">No repository found for this paper.</p>}
      {rp.references.length > 0 && (
        <>
          <div className="sub">Code of the papers it builds on</div>
          {rp.references.map((x) => (
            <div key={x.paperId}>
              <div className="pm" style={{ marginTop: 8 }}>{x.title} · influence {fmt(x.influence)}</div>
              {x.repos.map((r) => <RepoRow key={r.name || r.url} r={r} />)}
            </div>
          ))}
        </>
      )}
    </div>
  );
}

export default function Overview({ result, onOpen }) {
  return (
    <div className="grid">
      <div className="col">
        <Summary summary={result.summary} />
        <Storyline storyline={result.evolution.storyline} />
        <IntentBreakdown counts={result.stats.intent_counts} />
      </div>
      <div className="col">
        <PaperList title="Read these first" src="Model 2 influence" items={result.references.slice(0, 6)}
          empty="No references." onOpen={onOpen} />
        <PaperList title="Built on this paper" items={result.citations.slice(0, 5)}
          empty="No citing papers yet." onOpen={onOpen} />
        <Code repos={result.repos} />
      </div>
    </div>
  );
}
