export const INTENTS = ["background", "method", "result", "unknown"];
export const INTENT_TEXT = {
  background: "background",
  method: "method / tool",
  result: "result comparison",
  unknown: "no context",
};
export const CLUSTER_COLORS = ["#2c6bd6", "#d48a12", "#2e9e6b", "#b4462f", "#7a55c7", "#1f9bb0", "#c2477f", "#6b7d2a"];
export const STAGES = [
  "Resolving paper",
  "Fetching references",
  "Fetching citing papers",
  "Classifying citation intent (Model 1)",
  "Scoring citation influence (Model 2)",
  "Tracing lineage",
  "Mapping idea evolution",
  "Backtracking code repositories",
  "Writing the one-glance summary",
];

export const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
export const intentColor = (i) => css(`--${i === "background" ? "bgk" : i || "unknown"}`) || "#999";
export const fmt = (x, d = 2) => (x === null || x === undefined ? "–" : Number(x).toFixed(d));
export const num = (n) => (n === null || n === undefined ? "–" : Number(n).toLocaleString());
export const short = (t, n = 38) => ((t || "").length > n ? t.slice(0, n - 1) + "…" : t || "");
