import { useMemo } from "react";
import Cytoscape, { baseStyle } from "./Cytoscape.jsx";
import { Swatch } from "./bits.jsx";
import { CLUSTER_COLORS, css, short } from "../lib/format.js";

const LANE = 120;

export default function Evolution({ evolution, graph, onOpen }) {
  const E = evolution;

  const elements = useMemo(() => {
    if (!E.nodes?.length) return [];
    const years = E.nodes.map((n) => n.year);
    const y0 = Math.min(...years);
    const y1 = Math.max(...years);
    const XS = Math.max(40, Math.min(110, 1100 / Math.max(1, y1 - y0)));
    const stack = new Map();
    const nodes = E.nodes.map((n) => {
      const k = `${n.cluster}:${n.year}`;
      const j = stack.get(k) ?? 0;
      stack.set(k, j + 1);
      return {
        data: { id: n.paperId, title: n.title, cluster: n.cluster,
          type: n.role === "focus" ? "focus" : n.role === "descendant" ? "citation" : "reference",
          size: n.role === "focus" ? 34 : 7 + 3.2 * Math.log1p(n.citationCount),
          short: n.role === "focus" ? short(n.title, 48) : "" },
        position: { x: (n.year - y0) * XS, y: n.cluster * LANE + ((j % 7) - 3) * 12 },
      };
    });
    const fixed = { selectable: false, grabbable: false };
    const lanes = E.clusters.map((c) => ({ data: { id: `lane${c.id}`, short: short(c.label, 30), lane: 1 },
      position: { x: -150, y: c.id * LANE }, ...fixed }));
    const ticks = [];
    for (let y = y0; y <= y1; y += Math.max(1, Math.round((y1 - y0) / 10))) {
      ticks.push({ data: { id: `yr${y}`, short: String(y), lane: 2 }, position: { x: (y - y0) * XS, y: -70 }, ...fixed });
    }
    const ids = new Set(E.nodes.map((n) => n.paperId));
    const edges = graph.edges
      .filter((e) => ids.has(e.source) && ids.has(e.target) && (e.influence || 0) >= 0.5)
      .map((e, i) => ({ data: { id: `v${i}`, source: e.source, target: e.target, influence: e.influence } }));
    return [...nodes, ...lanes, ...ticks, ...edges];
  }, [E, graph]);

  const style = useMemo(() => baseStyle().concat([
    { selector: "node[cluster >= 0]", style: { "background-color": (n) => CLUSTER_COLORS[n.data("cluster") % 8] } },
    { selector: "node[type='focus']", style: { "background-color": css("--ink"), "z-index": 10 } },
    { selector: "node[lane]", style: { width: 1, height: 1, "background-opacity": 0, "text-valign": "center",
      "text-halign": "left", "font-size": 11, "font-weight": 600, color: css("--ink"), events: "no" } },
    { selector: "node[lane = 2]", style: { "text-halign": "center", color: css("--muted"), "font-weight": 400 } },
    { selector: "edge", style: { opacity: 0.25, "line-color": css("--muted"), "target-arrow-color": css("--muted") } },
  ]), []);

  if (!E.nodes?.length) return <p className="hint">Not enough connected papers to build an evolution map.</p>;

  const onReady = (cy) => {
    cy.on("mouseover", "node[!lane]", (e) => e.target.style("label", short(e.target.data("title"), 60)));
    cy.on("mouseout", "node[!lane]", (e) => { if (e.target.data("type") !== "focus") e.target.style("label", ""); });
  };

  return (
    <>
      <div className="evo-legend">
        {E.clusters.map((c) => (
          <div key={c.id}>
            <Swatch color={CLUSTER_COLORS[c.id % 8]} /><b>{c.label}</b>
            <div className="pm">
              {c.year_min}–{c.year_max} · {c.size} papers ({c.n_ancestors} earlier, {c.n_descendants} later)
              {c.has_focus && " · contains this paper"}
            </div>
          </div>
        ))}
      </div>
      <div className="canvas-wrap">
        <Cytoscape className="cy tall" elements={elements} style={style} layout={{ name: "preset", padding: 30 }}
          onTapNode={onOpen} tapSelector="node[!lane]" onReady={onReady} />
      </div>
      <p className="hint">
        x = year, rows = idea clusters ({E.source}, k-means, labelled with class-based TF-IDF). ● earlier work it cites
        ◆ later work citing it. Size = citations. Hover a dot for its title.
      </p>
    </>
  );
}
