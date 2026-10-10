import { useEffect, useMemo, useRef, useState } from "react";
import Cytoscape from "./Cytoscape.jsx";
import { IntentLegend } from "./bits.jsx";
import { short } from "../lib/format.js";

export default function CitationGraph({ graph, onOpen }) {
  const [showRefs, setShowRefs] = useState(true);
  const [showCits, setShowCits] = useState(true);
  const [minInfl, setMinInfl] = useState(0);
  const cyRef = useRef(null);

  const elements = useMemo(() => [
    ...graph.nodes.map((n) => ({
      data: { ...n, short: n.type === "focus" ? short(n.label, 60) : n.influence > 0.6 ? short(n.label) : "",
        size: n.type === "focus" ? 46 : 10 + 30 * (n.influence || 0) },
    })),
    ...graph.edges.map((e, i) => ({ data: { id: `e${i}`, ...e } })),
  ], [graph]);

  const layout = useMemo(() => ({
    name: "cose", animate: false, randomize: true, nodeRepulsion: () => 9000,
    idealEdgeLength: (e) => 60 + 140 * (1 - (e.data("influence") || 0)),
  }), []);

  // filters toggle visibility on the live instance instead of re-running the layout
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || cy.destroyed()) return;
    cy.nodes().forEach((n) => {
      const t = n.data("type");
      const vis = t === "focus" ||
        ((t === "reference" ? showRefs : showCits) && (n.data("influence") || 0) >= minInfl);
      n.style("display", vis ? "element" : "none");
    });
  }, [showRefs, showCits, minInfl, elements]);

  return (
    <>
      <div className="toolbar">
        <label><input type="checkbox" checked={showRefs} onChange={(e) => setShowRefs(e.target.checked)} /> references</label>
        <label><input type="checkbox" checked={showCits} onChange={(e) => setShowCits(e.target.checked)} /> citing papers</label>
        <label>
          min influence{" "}
          <input type="range" min="0" max="0.95" step="0.05" value={minInfl}
            onChange={(e) => setMinInfl(parseFloat(e.target.value))} />{" "}
          <span>{minInfl.toFixed(2)}</span>
        </label>
        <IntentLegend extra={<span>◆ citing paper</span>} />
      </div>
      <div className="canvas-wrap">
        <Cytoscape elements={elements} layout={layout} onTapNode={onOpen} onReady={(cy) => { cyRef.current = cy; }} />
      </div>
      <p className="hint">Node size = predicted influence. Colour = citation intent. Click a node for the citation sentences.</p>
    </>
  );
}
