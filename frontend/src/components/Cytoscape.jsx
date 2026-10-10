import { useEffect, useRef } from "react";
import cytoscape from "cytoscape";
import { css, intentColor } from "../lib/format.js";

/** Style shared by all three graphs. Theme colours are read from CSS variables at build time. */
export function baseStyle() {
  return [
    {
      selector: "node",
      style: {
        "background-color": (n) => (n.data("type") === "focus" ? css("--ink") : intentColor(n.data("intent"))),
        width: "data(size)", height: "data(size)", label: "data(short)", "font-size": 9, color: css("--muted"),
        "text-valign": "bottom", "text-margin-y": 3, "text-wrap": "wrap", "text-max-width": 120,
        "border-width": (n) => (n.data("type") === "citation" ? 2 : 0), "border-color": css("--panel"),
        shape: (n) => (n.data("type") === "citation" ? "diamond" : "ellipse"),
      },
    },
    {
      selector: "node[type='focus']",
      style: { "font-size": 12, "font-weight": 700, color: css("--ink"), "border-width": 4,
        "border-color": css("--accent"), "text-max-width": 220 },
    },
    {
      selector: "edge",
      style: { width: (e) => 0.6 + 4 * (e.data("influence") || 0), "line-color": css("--line"), "curve-style": "bezier",
        "target-arrow-shape": "triangle", "target-arrow-color": css("--line"), "arrow-scale": 0.7, opacity: 0.85 },
    },
    { selector: "node:selected", style: { "border-width": 3, "border-color": css("--accent") } },
  ];
}

/**
 * Mounts a Cytoscape instance. Rebuilds when `elements` changes (pass a memoised array).
 * `onReady(cy)` gives the parent access for filters; `onTapNode(id)` handles clicks.
 */
export default function Cytoscape({ elements, style, layout, onTapNode, onReady, tapSelector = "node", className = "cy" }) {
  const box = useRef(null);
  const tapRef = useRef(onTapNode);
  tapRef.current = onTapNode;

  useEffect(() => {
    const cy = cytoscape({
      container: box.current, elements, style: style ?? baseStyle(), layout,
      maxZoom: 1.6, minZoom: 0.15,
    });
    cy.fit(undefined, 30);
    if (cy.zoom() > 1.1) { cy.zoom(1.1); cy.center(); }
    cy.on("tap", tapSelector, (e) => tapRef.current?.(e.target.id()));
    onReady?.(cy);
    return () => cy.destroy();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [elements]);

  return <div ref={box} className={className} />;
}
