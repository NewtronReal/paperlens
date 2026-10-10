import { STAGES } from "../lib/format.js";

export function Empty({ examples, onPick }) {
  return (
    <section className="empty">
      <h1>Understand a paper at a glance.</h1>
      <p>
        PaperLens reads every sentence where a paper cites prior work, classifies <em>why</em> it was cited
        (Model 1, SciBERT), predicts which references the paper truly builds on (Model 2, graph neural network),
        and turns that into a lineage tree, an idea-evolution map and a reading list.
      </p>
      <div className="examples">
        {examples.map((e) => (
          <button key={e.query} onClick={() => onPick(e.query)}>{e.label}</button>
        ))}
      </div>
    </section>
  );
}

export function Loading({ job }) {
  const idx = STAGES.indexOf(job?.stage);
  return (
    <section className="loading">
      <div className="stage">{job?.stage ?? "Starting…"}</div>
      <div className="bar"><div id="barFill" style={{ width: `${Math.round((job?.progress ?? 0) * 100)}%` }} /></div>
      <ol className="stages">
        {STAGES.map((s, i) => (
          <li key={s} className={(idx < 0 ? job?.status === "done" : i < idx) ? "done" : ""}>{s}</li>
        ))}
      </ol>
    </section>
  );
}

export function ErrorBox({ message }) {
  return <section className="error">Could not analyse this paper: {message}</section>;
}
