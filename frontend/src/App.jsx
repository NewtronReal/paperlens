import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { analyze, getExamples, getHealth } from "./api.js";
import TopBar from "./components/TopBar.jsx";
import { Empty, ErrorBox, Loading } from "./components/Status.jsx";
import Hero from "./components/Hero.jsx";
import Overview from "./components/Overview.jsx";
import CitationGraph from "./components/CitationGraph.jsx";
import Lineage from "./components/Lineage.jsx";
import Evolution from "./components/Evolution.jsx";
import PaperTable from "./components/PaperTable.jsx";
import Drawer from "./components/Drawer.jsx";

const TABS = [
  ["overview", "Overview"], ["graph", "Citation graph"], ["lineage", "Lineage"],
  ["evolution", "Idea evolution"], ["refs", "References"], ["cits", "Citing papers"],
];

export default function App() {
  const [health, setHealth] = useState(null);
  const [examples, setExamples] = useState([]);
  const [query, setQuery] = useState("");
  const [file, setFile] = useState(null);
  const [phase, setPhase] = useState("empty"); // empty | loading | error | result
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [tab, setTab] = useState("overview");
  const [openId, setOpenId] = useState(null);
  const abortRef = useRef(null);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => {});
    getExamples().then(setExamples).catch(() => {});
  }, []);

  const run = useCallback(async (q = query, f = file) => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setPhase("loading");
    setJob(null);
    setOpenId(null);
    try {
      const res = await analyze({ query: q, file: f }, setJob, ctrl.signal);
      setResult(res);
      setTab("overview");
      setPhase("result");
      window.scrollTo({ top: 0 });
    } catch (e) {
      if (e.name === "AbortError") return;
      setError(e.message);
      setPhase("error");
    }
  }, [query, file]);

  const analyzeOther = useCallback((q) => {
    setQuery(q);
    setFile(null);
    run(q, null);
  }, [run]);

  const byId = useMemo(
    () => new Map(result ? [...result.citations, ...result.references].map((p) => [p.paperId, p]) : []),
    [result],
  );
  const closeDrawer = useCallback(() => setOpenId(null), []);

  return (
    <>
      <TopBar query={query} setQuery={setQuery} file={file} setFile={setFile} busy={phase === "loading"}
        onSubmit={() => run()} health={health} />
      <main>
        {phase === "empty" && <Empty examples={examples} onPick={(q) => { setQuery(q); run(q, file); }} />}
        {phase === "loading" && <Loading job={job} />}
        {phase === "error" && <ErrorBox message={error} />}
        {phase === "result" && result && (
          <section>
            <Hero result={result} />
            <nav className="tabs">
              {TABS.map(([k, label]) => (
                <button key={k} className={tab === k ? "active" : ""} onClick={() => setTab(k)}>{label}</button>
              ))}
            </nav>
            {/* only the active tab is mounted, so each graph lays out at its real size */}
            {tab === "overview" && <Overview result={result} onOpen={setOpenId} />}
            {tab === "graph" && <CitationGraph graph={result.graph} onOpen={setOpenId} />}
            {tab === "lineage" && <Lineage lineage={result.lineage} rootId={result.paper.paperId} onOpen={setOpenId} />}
            {tab === "evolution" && <Evolution evolution={result.evolution} graph={result.graph} onOpen={setOpenId} />}
            {tab === "refs" && <PaperTable rows={result.references} emptyText="No references available." onOpen={setOpenId} />}
            {tab === "cits" && <PaperTable rows={result.citations} emptyText="No citing papers available." onOpen={setOpenId} />}
          </section>
        )}
      </main>
      {result && (
        <Drawer paperId={openId} result={result} byId={byId} onClose={closeDrawer} onAnalyze={analyzeOther} />
      )}
    </>
  );
}
