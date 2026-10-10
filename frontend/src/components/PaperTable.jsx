import { useMemo, useState } from "react";
import { Badge, Meter } from "./bits.jsx";
import { num } from "../lib/format.js";

const COLS = [
  ["title", "Title"], ["year", "Year"], ["intent", "Intent"], ["influence", "Influence"],
  ["n_contexts", "Mentions"], ["citationCount", "Citations"], ["s2_influential", "S2 infl."],
];

export default function PaperTable({ rows, emptyText, onOpen }) {
  const [sort, setSort] = useState({ key: "influence", asc: false });

  const sorted = useMemo(() => [...rows].sort((a, b) => {
    const x = a[sort.key] ?? -1;
    const y = b[sort.key] ?? -1;
    return (x > y ? 1 : x < y ? -1 : 0) * (sort.asc ? 1 : -1);
  }), [rows, sort]);

  const clickHeader = (key) =>
    setSort((s) => ({ key, asc: s.key === key ? !s.asc : key === "title" }));

  if (!rows.length) return <p className="muted">{emptyText}</p>;
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {COLS.map(([k, label]) => (
              <th key={k} onClick={() => clickHeader(k)}
                className={k === sort.key ? `sorted${sort.asc ? " asc" : ""}` : ""}>{label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((r) => (
            <tr key={r.paperId} onClick={() => onOpen(r.paperId)}>
              <td>{r.title}<div className="pm">{(r.authors ?? []).slice(0, 3).join(", ")}</div></td>
              <td className="num">{r.year ?? ""}</td>
              <td><Badge intent={r.intent} /></td>
              <td><Meter value={r.influence} /></td>
              <td className="num">{r.n_contexts}</td>
              <td className="num">{num(r.citationCount)}</td>
              <td>{r.s2_influential ? "✓" : ""}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
