// Thin client for the FastAPI backend.

async function json(res) {
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch { /* not JSON */ }
    throw new Error(detail);
  }
  return res.json();
}

export const getHealth = () => fetch("/api/health").then(json);
export const getExamples = () => fetch("/api/examples").then(json);

/** Starts an analysis job and polls it. onProgress({stage, progress}) is called on every poll. */
export async function analyze({ query, file }, onProgress, signal) {
  const fd = new FormData();
  fd.append("query", query ?? "");
  if (file) fd.append("file", file);
  const { job_id } = await fetch("/api/analyze", { method: "POST", body: fd, signal }).then(json);
  for (;;) {
    await new Promise((r) => setTimeout(r, 700));
    if (signal?.aborted) throw new DOMException("aborted", "AbortError");
    const job = await fetch(`/api/jobs/${job_id}`, { signal }).then(json);
    onProgress?.(job);
    if (job.status === "done") return job.result;
    if (job.status === "error") throw new Error(job.error);
  }
}
