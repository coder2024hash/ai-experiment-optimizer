import React, { useState } from "react";
import "./App.css";

const API_BASE = "http://127.0.0.1:8001";

const ENDPOINTS = [
  { method: "GET", path: "/", description: "Health check" },
  { method: "POST", path: "/upload", description: "Upload CSV file" },
  { method: "POST", path: "/inspect", description: "Inspect dataset & suggest targets" },
  { method: "POST", path: "/train", description: "Run single experiment" },
  { method: "POST", path: "/run_experiment_batch", description: "Run batch of experiments" },
  { method: "GET", path: "/experiments", description: "List all experiments" },
  { method: "GET", path: "/best_experiments", description: "Top experiments by score" },
  { method: "POST", path: "/dataset_meta", description: "Get dataset meta-features" },
  { method: "POST", path: "/compute_experiment_labels", description: "Label experiments good/medium/bad" },
  { method: "POST", path: "/suggest_config", description: "Suggest next config (future brain)" },
];

function App() {
  const [csvFile, setCsvFile] = useState(null);
  const [uploadInfo, setUploadInfo] = useState(null);
  const [targetColumn, setTargetColumn] = useState("");
  const [training, setTraining] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [experiments, setExperiments] = useState([]);
  const [bestExperiments, setBestExperiments] = useState([]);

  // hyperparameters
  const [nEstimators, setNEstimators] = useState(200);
  const [maxDepth, setMaxDepth] = useState(""); // "" means None

  // 1. File selection
  const handleFileChange = (event) => {
    setCsvFile(event.target.files[0]);
    setUploadInfo(null);
    setTargetColumn("");
    setResult(null);
    setError(null);
  };

  // 2. Upload CSV to backend (/upload)
  const handleUpload = async () => {
    if (!csvFile) return;
    setError(null);

    const formData = new FormData();
    formData.append("file", csvFile);

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error("Upload failed");
      const data = await res.json();
      setUploadInfo(data);
      if (data.suggested_targets && data.suggested_targets.length > 0) {
        setTargetColumn(data.suggested_targets[0]);
      }
    } catch (e) {
      setError(e.message);
    }
  };

  // 3. Train single experiment (/train)
  const handleTrainModel = async () => {
    if (!uploadInfo || !targetColumn) return;
    setTraining(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("filename", uploadInfo.filename);
    formData.append("target_column", targetColumn);
    formData.append("problem_type", "auto");
    formData.append("test_size", "0.2");
    formData.append("random_state", "42");

    // hyperparameters from state
    formData.append("n_estimators", String(nEstimators));
    formData.append("max_depth", maxDepth); // backend: "" = None

    try {
      const res = await fetch(`${API_BASE}/train`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error("Training API error");
      const data = await res.json();
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setTraining(false);
    }
  };

  // 4. Load experiments
  const handleLoadExperiments = async () => {
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/experiments`);
      const data = await res.json();
      setExperiments(data);
    } catch (e) {
      setError(e.message);
    }
  };

  // 5. Load best experiments
  const handleLoadBestExperiments = async () => {
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/best_experiments?top_k=5`);
      const data = await res.json();
      setBestExperiments(data);
    } catch (e) {
      setError(e.message);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      {/* Top bar */}
      <header className="border-b border-slate-800 bg-slate-900/70 backdrop-blur">
        <div className="max-w-6xl mx-auto flex items-center justify-between py-3 px-4">
          <div className="flex items-center gap-2">
            <div className="h-8 w-8 rounded-lg bg-emerald-500 flex items-center justify-center text-slate-950 font-bold">
              AI
            </div>
            <div>
              <h1 className="text-lg font-semibold tracking-tight">
                AI Experiment Optimizer
              </h1>
              <p className="text-xs text-slate-400">
                Turn any CSV into ranked ML experiments
              </p>
            </div>
          </div>
          <span className="text-xs px-2 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
            v1 • Random Forest
          </span>
        </div>
      </header>

      {/* Main grid */}
      <main className="max-w-6xl mx-auto p-4 grid gap-4 md:grid-cols-3">
        {/* Left column: upload + target + hyperparams */}
        <div className="md:col-span-1 space-y-4">
          {/* Upload card */}
          <section className="bg-slate-900/70 border border-slate-800 rounded-2xl p-4 shadow-lg shadow-emerald-500/5">
            <h2 className="text-sm font-semibold text-slate-100 mb-2">
              1. Upload dataset
            </h2>
            <p className="text-xs text-slate-400 mb-3">
              Upload a CSV file to inspect columns and suggested targets.
            </p>
            <div className="space-y-2">
              <input
                type="file"
                accept=".csv"
                onChange={handleFileChange}
                className="block w-full text-xs text-slate-300
                           file:mr-3 file:py-1.5 file:px-3
                           file:rounded-full file:border-0
                           file:text-xs file:font-medium
                           file:bg-emerald-500 file:text-slate-950
                           hover:file:bg-emerald-400"
              />
              <button
                onClick={handleUpload}
                disabled={!csvFile}
                className="w-full inline-flex justify-center items-center py-1.5 px-3 rounded-full
                           text-xs font-medium bg-emerald-500 text-slate-950
                           hover:bg-emerald-400 disabled:opacity-40 disabled:hover:bg-emerald-500
                           transition"
              >
                Upload & Inspect
              </button>
              {uploadInfo && (
                <p className="text-[11px] text-slate-400 mt-2">
                  Loaded{" "}
                  <span className="font-semibold text-emerald-400">
                    {uploadInfo.filename}
                  </span>{" "}
                  • {uploadInfo.rows} rows • {uploadInfo.columns.length} columns
                </p>
              )}
            </div>
          </section>

          {/* Target + Hyperparams card */}
          <section className="bg-slate-900/70 border border-slate-800 rounded-2xl p-4 shadow-lg shadow-indigo-500/5">
            <h2 className="text-sm font-semibold text-slate-100 mb-2">
              2. Configure experiment
            </h2>

            {uploadInfo ? (
              <div className="space-y-3">
                <div>
                  <label className="text-xs text-slate-400 block mb-1">
                    Target column
                  </label>
                  <select
                    value={targetColumn}
                    onChange={(e) => setTargetColumn(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-700 rounded-lg text-xs px-2 py-1.5
                               focus:outline-none focus:ring-1 focus:ring-emerald-500 focus:border-emerald-500"
                  >
                    {uploadInfo.suggested_targets.map((col) => (
                      <option key={col} value={col}>
                        {col}
                      </option>
                    ))}
                  </select>
                  <p className="text-[11px] text-slate-500 mt-1">
                    Suggested: {uploadInfo.suggested_targets.join(", ")}
                  </p>
                </div>

                {/* Hyperparameters */}
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div>
                    <label className="block text-slate-400 mb-1">
                      n_estimators
                    </label>
                    <input
                      type="number"
                      min={10}
                      step={10}
                      value={nEstimators}
                      onChange={(e) => setNEstimators(Number(e.target.value))}
                      className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2 py-1.5
                                 focus:outline-none focus:ring-1 focus:ring-emerald-500 focus:border-emerald-500"
                    />
                  </div>
                  <div>
                    <label className="block text-slate-400 mb-1">
                      max_depth (blank = None)
                    </label>
                    <input
                      type="number"
                      value={maxDepth}
                      onChange={(e) => setMaxDepth(e.target.value)}
                      className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2 py-1.5
                                 focus:outline-none focus:ring-1 focus:ring-emerald-500 focus:border-emerald-500"
                    />
                  </div>
                </div>

                <button
                  onClick={handleTrainModel}
                  disabled={!uploadInfo || training}
                  className="w-full inline-flex justify-center items-center py-1.5 px-3 rounded-full
                             text-xs font-medium bg-indigo-500 text-slate-50
                             hover:bg-indigo-400 disabled:opacity-40 disabled:hover:bg-indigo-500
                             transition mt-1"
                >
                  {training ? "Training…" : "Run Experiment"}
                </button>
              </div>
            ) : (
              <p className="text-xs text-slate-500">
                Upload a dataset first to configure the experiment.
              </p>
            )}
          </section>

          {/* Errors */}
          {error && (
            <div className="bg-rose-500/10 border border-rose-500/40 text-rose-200 text-xs rounded-xl p-3">
              Error: {error}
            </div>
          )}
        </div>

        {/* Right 2 columns: results + experiments */}
        <div className="md:col-span-2 space-y-4">
          {/* Training result card */}
          <section className="bg-slate-900/70 border border-slate-800 rounded-2xl p-4 shadow-xl shadow-slate-900/40">
            <div className="flex items-center justify-between mb-3">
              <div>
                <h2 className="text-sm font-semibold text-slate-100">
                  3. Experiment result
                </h2>
                <p className="text-[11px] text-s
                   slate-400">
                  Performance metrics and top features for the latest run.
                </p>
              </div>
            </div>

            {result ? (
              <div className="space-y-4">
                {/* Metrics row */}
                {result.metrics && (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                    {result.metrics.r2_score !== undefined && (
                      <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-3">
                        <p className="text-slate-400 text-[11px]">R² score</p>
                        <p className="text-emerald-400 text-xl font-semibold">
                          {result.metrics.r2_score.toFixed(3)}
                        </p>
                      </div>
                    )}
                    {result.metrics.rmse !== undefined && (
                      <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-3">
                        <p className="text-slate-400 text-[11px]">RMSE</p>
                        <p className="text-indigo-400 text-xl font-semibold">
                          {result.metrics.rmse.toFixed(3)}
                        </p>
                      </div>
                    )}
                    <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-3">
                      <p className="text-slate-400 text-[11px]">Samples</p>
                      <p className="text-slate-100 text-lg font-semibold">
                        {result.n_samples}
                      </p>
                    </div>
                    <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-3">
                      <p className="text-slate-400 text-[11px]">Features</p>
                      <p className="text-slate-100 text-lg font-semibold">
                        {result.n_features}
                      </p>
                    </div>
                  </div>
                )}

                {/* Feature importance (top 5) */}
                {result.feature_importance && (
                  <div>
                    <h3 className="text-xs font-semibold text-slate-200 mb-2">
                      Top features
                    </h3>
                    <ul className="space-y-1 text-[11px]">
                      {Object.entries(result.feature_importance)
                        .filter(([name]) => name !== "target")
                        .sort(([, a], [, b]) => b - a)
                        .slice(0, 5)
                        .map(([name, val]) => (
                          <li
                            key={name}
                            className="flex items-center justify-between bg-slate-950/40 border border-slate-800 rounded-lg px-2 py-1"
                          >
                            <span className="text-slate-200">{name}</span>
                            <span className="text-emerald-400 font-mono">
                              {val.toFixed(3)}
                            </span>
                          </li>
                        ))}
                    </ul>
                  </div>
                )}

                {/* Sample predictions */}
                {result.results_table && result.results_table.length > 0 && (
                  <div>
                    <h3 className="text-xs font-semibold text-slate-200 mb-1">
                      Sample predictions
                    </h3>
                    <p className="text-[11px] text-slate-500 mb-2">
                      Actual vs predicted values for a few test samples.
                    </p>
                    <div className="overflow-auto border border-slate-800 rounded-xl text-[11px]">
                      <table className="min-w-full border-collapse">
                        <thead className="bg-slate-900/80">
                          <tr>
                            <th className="px-2 py-1 text-left border-b border-slate-800">
                              Index
                            </th>
                            <th className="px-2 py-1 text-left border-b border-slate-800">
                              Actual
                            </th>
                            <th className="px-2 py-1 text-left border-b border-slate-800">
                              Predicted
                            </th>
                          </tr>
                        </thead>
                        <tbody>
                          {result.results_table.slice(0, 5).map((row) => (
                            <tr key={row.index} className="odd:bg-slate-950/40">
                              <td className="px-2 py-1 border-b border-slate-900">
                                {row.index}
                              </td>
                              <td className="px-2 py-1 border-b border-slate-900">
                                {row.actual.toFixed(3)}
                              </td>
                              <td className="px-2 py-1 border-b border-slate-900">
                                {row.predicted.toFixed(3)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* Raw JSON toggle */}
                <details className="mt-2">
                  <summary className="text-[11px] text-slate-500 cursor-pointer">
                    Raw JSON
                  </summary>
                  <pre className="mt-1 text-[10px] bg-slate-950/80 border border-slate-800 rounded-xl p-2 overflow-auto">
                    {JSON.stringify(result, null, 2)}
                  </pre>
                </details>
              </div>
            ) : (
              <p className="text-xs text-slate-500">
                Run an experiment to see metrics and feature importance.
              </p>
            )}
          </section>

          {/* Experiments & Best Experiments */}
          <section className="bg-slate-900/70 border border-slate-800 rounded-2xl p-4">
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-sm font-semibold text-slate-100">
                4. Experiment history
              </h2>
              <div className="flex gap-2">
                <button
                  onClick={handleLoadExperiments}
                  className="text-[11px] px-2 py-1 rounded-full border border-slate-700 hover:border-emerald-500 hover:text-emerald-400"
                >
                  Load all
                </button>
                <button
                  onClick={handleLoadBestExperiments}
                  className="text-[11px] px-2 py-1 rounded-full border border-slate-700 hover:border-indigo-500 hover:text-indigo-400"
                >
                  Load top 5
                </button>
              </div>
            </div>
            <div className="grid md:grid-cols-2 gap-3 text-[11px]">
              <div className="bg-slate-950/50 border border-slate-800 rounded-xl p-2">
                <p className="text-slate-400 mb-1">All experiments</p>
                <pre className="max-h-48 overflow-auto text-[10px]">
                  {JSON.stringify(experiments, null, 2)}
                </pre>
              </div>
              <div className="bg-slate-950/50 border border-slate-800 rounded-xl p-2">
                <p className="text-slate-400 mb-1">Best experiments</p>
                <pre className="max-h-48 overflow-auto text-[10px]">
                  {JSON.stringify(bestExperiments, null, 2)}
                </pre>
              </div>
            </div>
          </section>

          {/* Optional: backend endpoints debug panel */}
          <section className="bg-slate-900/70 border border-slate-800 rounded-2xl p-4">
            <h2 className="text-sm font-semibold text-slate-100 mb-2">
              Backend API Endpoints
            </h2>
            <div className="overflow-auto text-[11px]">
              <table className="min-w-full border-collapse">
                <thead>
                  <tr className="bg-slate-900/80">
                    <th className="px-2 py-1 text-left border-b border-slate-800">
                      Method
                    </th>
                    <th className="px-2 py-1 text-left border-b border-slate-800">
                      Path
                    </th>
                    <th className="px-2 py-1 text-left border-b border-slate-800">
                      Description
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {ENDPOINTS.map((ep) => (
                    <tr key={ep.method + ep.path} className="odd:bg-slate-950/40">
                      <td className="px-2 py-1 border-b border-slate-900">
                        {ep.method}
                      </td>
                      <td className="px-2 py-1 border-b border-slate-900">
                        {ep.path}
                      </td>
                      <td className="px-2 py-1 border-b border-slate-900">
                        {ep.description}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}

export default App;
