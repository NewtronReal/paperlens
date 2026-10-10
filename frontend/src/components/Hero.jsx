import { ExtLink } from "./bits.jsx";
import { fmt, num } from "../lib/format.js";

export default function Hero({ result }) {
  const p = result.paper;
  const s = result.stats;
  const extraAuthors = (p.n_authors ?? 0) - (p.authors ?? []).length;
  const stats = [
    [num(p.citationCount), "citations"],
    [num(p.influentialCitationCount), "influential citations (S2)"],
    [num(s.n_references), "references resolved"],
    [num(s.n_core_references), "core references (Model 2 ≥ 0.5)"],
    [s.median_reference_age === null ? "–" : `${fmt(s.median_reference_age, 0)} y`, "median age of cited work"],
  ];
  return (
    <>
      <div>
        {result.warnings.map((w) => (
          <div key={w} className={`warn-banner ${/DEMO/.test(w) ? "demo" : ""}`}>{w}</div>
        ))}
      </div>
      <div className="hero">
        <div className="meta-line">{[p.venue, p.year, ...(p.fields ?? [])].filter(Boolean).join(" · ")}</div>
        <h1>{p.title || "Untitled"}</h1>
        <div className="authors">
          {(p.authors ?? []).join(", ")}
          {extraAuthors > 0 && ` +${extraAuthors} more`}
        </div>
        <div className="links">
          {p.arxiv && <ExtLink href={`https://arxiv.org/abs/${p.arxiv}`}>arXiv {p.arxiv}</ExtLink>}
          {p.pdf_url && <ExtLink href={p.pdf_url}>PDF</ExtLink>}
          {p.doi && <ExtLink href={`https://doi.org/${p.doi}`}>DOI</ExtLink>}
          {p.url && <ExtLink href={p.url}>Semantic Scholar</ExtLink>}
        </div>
        <div className="stats">
          {stats.map(([v, k]) => (
            <div className="stat" key={k}><div className="v">{v}</div><div className="k">{k}</div></div>
          ))}
        </div>
      </div>
    </>
  );
}
