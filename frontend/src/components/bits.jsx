import { INTENT_TEXT, fmt, intentColor } from "../lib/format.js";

export function Badge({ intent }) {
  const i = intent || "unknown";
  return <span className={`badge b-${i}`}>{INTENT_TEXT[i] || i}</span>;
}

export function Meter({ value }) {
  if (value === null || value === undefined) return null;
  return (
    <div className="meter">
      {fmt(value)}
      <div className="track"><div className="fill" style={{ width: `${Math.round(value * 100)}%` }} /></div>
    </div>
  );
}

export function Swatch({ color }) {
  return <span className="sw" style={{ background: color }} />;
}

export function IntentLegend({ extra }) {
  return (
    <span className="legend">
      {["background", "method", "result", "unknown"].map((i) => (
        <span key={i}><Swatch color={intentColor(i)} />{INTENT_TEXT[i]}</span>
      ))}
      {extra}
    </span>
  );
}

export function ExtLink({ href, children }) {
  return <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>;
}
