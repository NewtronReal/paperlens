import { useMemo } from "react";
import Cytoscape, { baseStyle } from "./Cytoscape.jsx";
import { css, fmt, intentColor, short } from "../lib/format.js";

const COL = 360;
const ROW = 74;

/** Tidy tree: leaves stacked vertically, parents centred on their children, oldest generation on the left. */
function treePositions(lineage, rootId) {
  const kids = new Map(lineage.nodes.map((n) => [n.paperId, []]));
  lineage.edges.forEach((e) => kids.get(e.target)?.push(e.source));
  const pos = new Map();
  let leaf = 0;
  const place = (id, level) => {
    const ch = kids.get(id) ?? [];
    ch.forEach((c) => place(c, level + 1));
    const y = ch.length ? (pos.get(ch[0]).y + pos.get(ch[ch.length - 1]).y) / 2 : leaf++ * ROW;
    pos.set(id, { x: (lineage.depth - level) * COL, y });
  };
  place(rootId, 0);
  return pos;
}

export default function Lineage({ lineage, rootId, onOpen }) {
  const elements = useMemo(() => {
    const pos = treePositions(lineage, rootId);
    return [
      ...lineage.nodes.filter((n) => pos.has(n.paperId)).map((n) => ({
        data: { id: n.paperId, intent: n.intent, influence: n.influence, type: n.level === 0 ? "focus" : "reference",
          size: n.level === 0 ? 34 : 12 + 16 * (n.influence || 0),
          short: `${short(n.title, 70)} (${n.year || "?"})${n.level ? ` · ${fmt(n.influence)}` : ""}` },
        position: pos.get(n.paperId),
      })),
      ...lineage.edges.map((e, i) => ({ data: { id: `l${i}`, ...e } })),
    ];
  }, [lineage, rootId]);

  const style = useMemo(() => baseStyle().concat([
    { selector: "node", style: { "font-size": 11, color: css("--ink"), "text-max-width": 290, "text-wrap": "wrap" } },
    { selector: "edge", style: { "curve-style": "taxi", "taxi-direction": "horizontal",
      "line-color": (e) => intentColor(e.data("intent")), "target-arrow-color": (e) => intentColor(e.data("intent")) } },
  ]), []);

  return (
    <>
      <p className="hint">
        Ancestry tree, read left to right in time: at each level the {lineage.branch} references with the highest
        predicted influence (Model 2), {lineage.depth} levels deep. Edge width = influence, colour = why it was cited.
      </p>
      <div className="canvas-wrap">
        <Cytoscape elements={elements} style={style} layout={{ name: "preset", padding: 50 }} onTapNode={onOpen} />
      </div>
    </>
  );
}
