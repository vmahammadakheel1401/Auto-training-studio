import React, { useMemo, useState } from 'react';
import {
  AlertCircle,
  Check,
  Code2,
  Copy,
  Download,
  FileJson,
  Loader2,
  Package,
  RotateCcw,
  ShoppingBasket,
  Sliders,
} from 'lucide-react';
import { predictSingleRow } from '../ml/algorithms';
import {
  generateModelStateJson,
  generatePickleSnippet,
  generatePythonScript,
} from '../ml/codeGenerator';
import { transformSingleRow } from '../ml/preprocessor';
import { generatePickleViaPyodide } from '../ml/pyodideRunner';
import { AnyRunResult, FeatureEncoderSpec, ParsedDataset } from '../ml/types';
import {
  predictUnsupervisedSingleRow,
  recommendBasketItems,
  transformUnsupervisedRow,
} from '../ml/unsupervised';

interface StepPredictExportProps {
  run: AnyRunResult;
  dataset: ParsedDataset;
}

function buildBaselineInputs(specs: FeatureEncoderSpec[]): Record<string, string | number> {
  const init: Record<string, string | number> = {};
  for (const spec of specs) {
    if (spec.type === 'numeric') {
      init[spec.rawName] = spec.median ?? 0;
    } else {
      init[spec.rawName] = spec.mode ?? spec.categories?.[0] ?? '';
    }
  }
  return init;
}

function downloadTextFile(content: string, filename: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function downloadBinaryFile(bytes: Uint8Array, filename: string) {
  const blob = new Blob([bytes.slice().buffer], {
    type: 'application/octet-stream',
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export const StepPredictExport: React.FC<StepPredictExportProps> = ({
  run,
  dataset,
}) => {
  const specs = run.preprocessor.specs;
  const [inputs, setInputs] = useState<Record<string, string | number>>(() =>
    buildBaselineInputs(specs)
  );
  const [selectedBasketItems, setSelectedBasketItems] = useState<string[]>(() => {
    if (run.learningMode === 'unsupervised' && run.model.key === 'apriori') {
      const firstAntecedent = run.model.associationRules?.[0]?.antecedents;
      if (firstAntecedent && firstAntecedent.length > 0) {
        return [...firstAntecedent];
      }
      return run.preprocessor.transactionItemsCatalog.slice(0, 1);
    }
    return [];
  });
  const [activeTab, setActiveTab] = useState<'python' | 'json'>('python');
  const [copied, setCopied] = useState(false);
  const [copiedPickle, setCopiedPickle] = useState(false);
  const [isBuildingPkl, setIsBuildingPkl] = useState(false);
  const [pklStatus, setPklStatus] = useState('');
  const [pklError, setPklError] = useState<string | null>(null);

  const [lastRunId, setLastRunId] = useState(run.runId);
  if (run.runId !== lastRunId) {
    setLastRunId(run.runId);
    setInputs(buildBaselineInputs(run.preprocessor.specs));
    setPklError(null);
    setPklStatus('');
    if (run.learningMode === 'unsupervised' && run.model.key === 'apriori') {
      const firstAntecedent = run.model.associationRules?.[0]?.antecedents;
      setSelectedBasketItems(
        firstAntecedent && firstAntecedent.length > 0
          ? [...firstAntecedent]
          : run.preprocessor.transactionItemsCatalog.slice(0, 1)
      );
    }
  }

  const supervisedPrediction = useMemo(() => {
    if (run.learningMode !== 'supervised') return null;
    const encoded = transformSingleRow(inputs, run.preprocessor);
    return predictSingleRow(run.winner.weights, encoded, run.preprocessor);
  }, [inputs, run]);

  const unsupervisedPrediction = useMemo(() => {
    if (run.learningMode !== 'unsupervised' || run.model.key === 'apriori') return null;
    const encoded = transformUnsupervisedRow(inputs, run.preprocessor);
    return predictUnsupervisedSingleRow(run.model, encoded);
  }, [inputs, run]);

  const basketRecommendations = useMemo(() => {
    if (run.learningMode !== 'unsupervised' || run.model.key !== 'apriori') return [];
    return recommendBasketItems(run.model.associationRules ?? [], selectedBasketItems);
  }, [run, selectedBasketItems]);

  const pythonScript = useMemo(() => generatePythonScript(run), [run]);
  const pickleSnippet = useMemo(() => generatePickleSnippet(run), [run]);
  const modelJson = useMemo(() => generateModelStateJson(run), [run]);
  const activeCode = activeTab === 'python' ? pythonScript : modelJson;

  const baseName = run.fileName.replace(/\.csv$/i, '').replace(/[^a-z0-9_-]+/gi, '_');

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(activeCode);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      // Fallback ignored if clipboard blocked
    }
  };

  const handleCopyPickleSnippet = async () => {
    try {
      await navigator.clipboard.writeText(pickleSnippet);
      setCopiedPickle(true);
      setTimeout(() => setCopiedPickle(false), 1800);
    } catch {
      // Fallback ignored if clipboard blocked
    }
  };

  const regressionGaugePct = useMemo(() => {
    if (!supervisedPrediction || supervisedPrediction.taskType !== 'regression') return 0;
    if (run.learningMode !== 'supervised') return 0;
    const min = run.preprocessor.targetMin ?? 0;
    const max = run.preprocessor.targetMax ?? 1;
    const val = Number(supervisedPrediction.rawPrediction);
    if (max <= min) return 50;
    return Math.min(100, Math.max(0, ((val - min) / (max - min)) * 100));
  }, [supervisedPrediction, run]);

  const isApriori =
    run.learningMode === 'unsupervised' && run.model.key === 'apriori';

  const activeModelName =
    run.learningMode === 'supervised' ? run.winner.name : run.model.name;

  const handleDownloadPickle = async () => {
    if (isBuildingPkl) return;
    setIsBuildingPkl(true);
    setPklError(null);
    setPklStatus('Initializing Python runtime & building .pkl...');
    try {
      const pklBytes = await generatePickleViaPyodide({
        dataset,
        fileName: run.fileName,
        pythonScript,
        needsMlxtend: isApriori,
        onStatus: setPklStatus,
      });
      downloadBinaryFile(pklBytes, 'trained_model.pkl');
    } catch (err) {
      setPklError(
        err instanceof Error
          ? err.message
          : 'Failed to generate trained_model.pkl in the browser Python runtime.'
      );
    } finally {
      setIsBuildingPkl(false);
      setPklStatus('');
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
      {/* Left Column: Live What-If Predictor / Cluster Projector / Basket Recommender */}
      <div className="lg:col-span-6 bg-white border border-slate-200 rounded-xl p-6 flex flex-col space-y-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              {isApriori ? (
                <ShoppingBasket className="w-4 h-4 text-indigo-600" />
              ) : (
                <Sliders className="w-4 h-4 text-indigo-600" />
              )}
              <h2 className="text-base font-semibold text-slate-900">
                {isApriori
                  ? 'Live Basket Recommender'
                  : run.learningMode === 'unsupervised'
                  ? run.model.key === 'pca'
                    ? 'Live PCA Component Projector'
                    : 'Live Cluster Segment Assigner'
                  : 'Live What-If Predictor'}
              </h2>
            </div>
            <p className="text-xs text-slate-500 mt-1">
              Powered in real time by {activeModelName}
            </p>
          </div>

          <button
            type="button"
            onClick={() => {
              if (isApriori) {
                setSelectedBasketItems([]);
              } else {
                setInputs(buildBaselineInputs(specs));
              }
            }}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-200 hover:bg-slate-50 text-xs font-medium text-slate-700 transition-colors cursor-pointer"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            {isApriori ? 'Clear Basket' : 'Reset Baseline'}
          </button>
        </div>

        {/* Supervised Prediction Card */}
        {run.learningMode === 'supervised' && supervisedPrediction && (
          <div className="rounded-xl border border-emerald-200 bg-emerald-50/70 p-5">
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs font-semibold text-emerald-800">
                Predicted {run.targetColumn}
              </span>
              <span className="text-[11px] font-mono text-emerald-800">
                {run.winner.name}
              </span>
            </div>

            <div className="mt-2 flex items-baseline justify-between gap-4">
              <div className="text-3xl font-bold font-mono tabular-nums text-slate-900">
                {supervisedPrediction.formattedPrediction}
              </div>
              {supervisedPrediction.taskType === 'classification' &&
                supervisedPrediction.confidence !== undefined && (
                  <div className="text-sm font-semibold font-mono tabular-nums text-emerald-800">
                    {(supervisedPrediction.confidence * 100).toFixed(1)}% confidence
                  </div>
                )}
            </div>

            {supervisedPrediction.taskType === 'regression' && (
              <div className="mt-4">
                <div className="h-2 w-full rounded-full bg-emerald-200/70 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-emerald-600 transition-all duration-150"
                    style={{ width: `${regressionGaugePct}%` }}
                  />
                </div>
                <div className="flex items-center justify-between text-[11px] font-mono tabular-nums text-emerald-900/75 mt-1.5">
                  <span>
                    Dataset Min:{' '}
                    {(run.preprocessor.targetMin ?? 0).toLocaleString(undefined, {
                      maximumFractionDigits: 1,
                    })}
                  </span>
                  <span>
                    Dataset Max:{' '}
                    {(run.preprocessor.targetMax ?? 0).toLocaleString(undefined, {
                      maximumFractionDigits: 1,
                    })}
                  </span>
                </div>
              </div>
            )}

            {supervisedPrediction.taskType === 'classification' &&
              supervisedPrediction.classProbabilities && (
                <div className="mt-4 space-y-2">
                  {supervisedPrediction.classProbabilities.map((cp) => (
                    <div key={cp.label} className="space-y-1">
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-medium text-slate-800">{cp.label}</span>
                        <span className="font-mono tabular-nums text-slate-700">
                          {(cp.probability * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="h-1.5 w-full rounded-full bg-emerald-200/70 overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-150 ${
                            cp.label === supervisedPrediction.formattedPrediction
                              ? 'bg-emerald-600'
                              : 'bg-slate-400'
                          }`}
                          style={{ width: `${Math.max(2, cp.probability * 100)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              )}
          </div>
        )}

        {/* Unsupervised PCA / Clustering Output Card */}
        {run.learningMode === 'unsupervised' &&
          !isApriori &&
          unsupervisedPrediction && (
            <div className="rounded-xl border border-emerald-200 bg-emerald-50/70 p-5">
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs font-semibold text-emerald-800">
                  {unsupervisedPrediction.modelKey === 'pca'
                    ? 'Projected Principal Coordinates'
                    : 'Assigned Segment'}
                </span>
                <span className="text-[11px] font-mono text-emerald-800">
                  {run.model.name}
                </span>
              </div>

              {unsupervisedPrediction.modelKey === 'pca' &&
                unsupervisedPrediction.projectedCoords && (
                  <div className="mt-3 grid grid-cols-3 gap-3">
                    {unsupervisedPrediction.projectedCoords.map((coord) => (
                      <div
                        key={coord.component}
                        className="rounded-lg border border-emerald-200 bg-white/80 px-3 py-2"
                      >
                        <div className="text-[11px] font-mono text-emerald-800">
                          {coord.component}
                        </div>
                        <div className="text-lg font-bold font-mono tabular-nums text-slate-900">
                          {coord.value >= 0 ? `+${coord.value.toFixed(2)}` : coord.value.toFixed(2)}
                        </div>
                      </div>
                    ))}
                  </div>
                )}

              {unsupervisedPrediction.assignedClusterLabel && (
                <>
                  <div className="mt-2 text-3xl font-bold font-mono text-slate-900">
                    {unsupervisedPrediction.assignedClusterLabel}
                  </div>
                  {unsupervisedPrediction.clusterProbabilities && (
                    <div className="mt-3 space-y-2">
                      {unsupervisedPrediction.clusterProbabilities.map((cp) => (
                        <div key={cp.clusterId} className="space-y-1">
                          <div className="flex items-center justify-between text-xs">
                            <span className="font-medium text-slate-800">{cp.label}</span>
                            <span className="font-mono tabular-nums text-slate-700">
                              {(cp.probability * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="h-1.5 w-full rounded-full bg-emerald-200/70 overflow-hidden">
                            <div
                              className="h-full rounded-full bg-emerald-600"
                              style={{ width: `${Math.max(2, cp.probability * 100)}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                  {unsupervisedPrediction.centroidDistances && (
                    <div className="mt-3 flex flex-wrap gap-2">
                      {unsupervisedPrediction.centroidDistances.map((cd) => (
                        <span
                          key={cd.clusterId}
                          className="text-[11px] font-mono px-2 py-1 rounded border border-emerald-200 bg-white/75 text-slate-700"
                        >
                          {cd.label}: d={cd.distance.toFixed(2)}
                        </span>
                      ))}
                    </div>
                  )}
                </>
              )}
            </div>
          )}

        {/* Unsupervised Apriori Basket Recommender */}
        {isApriori && run.learningMode === 'unsupervised' ? (
          <div className="space-y-4">
            <div className="rounded-xl border border-emerald-200 bg-emerald-50/70 p-5">
              <div className="text-xs font-semibold text-emerald-800 mb-2">
                Recommended Next Items ({basketRecommendations.length})
              </div>
              {basketRecommendations.length === 0 ? (
                <p className="text-xs text-slate-600">
                  Select one or more items below to generate live association-rule recommendations.
                </p>
              ) : (
                <div className="space-y-2">
                  {basketRecommendations.slice(0, 5).map((rec) => (
                    <div
                      key={rec.item}
                      className="flex items-center justify-between gap-3 rounded-lg border border-emerald-200 bg-white/85 px-3 py-2 text-xs"
                    >
                      <div>
                        <span className="font-mono font-bold text-slate-900">
                          + {rec.item}
                        </span>
                        <span className="text-[11px] text-slate-500 ml-2">
                          (from {rec.matchedAntecedent.join(' + ')})
                        </span>
                      </div>
                      <span className="font-mono tabular-nums font-semibold text-emerald-800">
                        {(rec.confidence * 100).toFixed(0)}% conf · {rec.lift.toFixed(2)}× lift
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div>
              <div className="text-xs font-semibold text-slate-700 mb-2">
                Toggle Items in Current Basket ({selectedBasketItems.length} selected)
              </div>
              <div className="flex flex-wrap gap-2 max-h-[340px] overflow-y-auto pr-1">
                {run.preprocessor.transactionItemsCatalog.map((item) => {
                  const active = selectedBasketItems.includes(item);
                  return (
                    <button
                      key={item}
                      type="button"
                      onClick={() =>
                        setSelectedBasketItems((prev) =>
                          prev.includes(item)
                            ? prev.filter((x) => x !== item)
                            : [...prev, item]
                        )
                      }
                      className={`px-3 py-1.5 rounded-lg border text-xs font-mono transition-colors cursor-pointer ${
                        active
                          ? 'bg-indigo-600 text-white border-indigo-600 font-semibold'
                          : 'bg-slate-50 text-slate-700 border-slate-200 hover:bg-slate-100'
                      }`}
                    >
                      {item}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        ) : (
          /* Feature Sliders & Dropdowns */
          <div className="space-y-4 max-h-[520px] overflow-y-auto pr-1">
            {specs.map((spec) => {
              if (spec.type === 'numeric') {
                const min = spec.min ?? 0;
                const max = spec.max ?? 100;
                const step = spec.step ?? 1;
                const current = Number(inputs[spec.rawName] ?? spec.median ?? min);

                return (
                  <div
                    key={spec.rawName}
                    className="rounded-lg border border-slate-200 bg-slate-50/50 px-4 py-3"
                  >
                    <div className="flex items-center justify-between gap-3 mb-2">
                      <label
                        htmlFor={`feat-${spec.rawName}`}
                        className="text-xs font-semibold text-slate-800 truncate"
                      >
                        {spec.rawName}
                      </label>
                      <input
                        type="number"
                        value={current}
                        step={step}
                        onChange={(e) => {
                          const val = Number(e.target.value);
                          if (Number.isFinite(val)) {
                            setInputs((prev) => ({
                              ...prev,
                              [spec.rawName]: val,
                            }));
                          }
                        }}
                        className="w-24 rounded-md border border-slate-300 bg-white px-2 py-1 text-right text-xs font-mono tabular-nums text-slate-900 focus:border-indigo-500 focus:outline-none"
                      />
                    </div>
                    <input
                      id={`feat-${spec.rawName}`}
                      type="range"
                      min={min}
                      max={max === min ? min + 1 : max}
                      step={step}
                      value={Math.min(max, Math.max(min, current))}
                      onChange={(e) =>
                        setInputs((prev) => ({
                          ...prev,
                          [spec.rawName]: Number(e.target.value),
                        }))
                      }
                      className="w-full accent-indigo-600 cursor-pointer"
                    />
                    <div className="flex items-center justify-between text-[11px] font-mono tabular-nums text-slate-400 mt-1">
                      <span>{min.toLocaleString()}</span>
                      <span>{max.toLocaleString()}</span>
                    </div>
                  </div>
                );
              }

              const categories = spec.categories ?? [];
              const currentCat = String(inputs[spec.rawName] ?? spec.mode ?? '');

              return (
                <div
                  key={spec.rawName}
                  className="rounded-lg border border-slate-200 bg-slate-50/50 px-4 py-3"
                >
                  <label
                    htmlFor={`feat-${spec.rawName}`}
                    className="block text-xs font-semibold text-slate-800 mb-2"
                  >
                    {spec.rawName}
                  </label>
                  <select
                    id={`feat-${spec.rawName}`}
                    value={currentCat}
                    onChange={(e) =>
                      setInputs((prev) => ({
                        ...prev,
                        [spec.rawName]: e.target.value,
                      }))
                    }
                    className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-medium text-slate-900 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/20"
                  >
                    {categories.map((cat) => (
                      <option key={cat} value={cat}>
                        {cat}
                      </option>
                    ))}
                  </select>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Right Column: Dual Code & Model Artifact Export Viewer */}
      <div className="lg:col-span-6 bg-white border border-slate-200 rounded-xl p-6 flex flex-col">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-base font-semibold text-slate-900">
              Export Model & Python Pickle (.pkl) Workflow
            </h2>
            <p className="text-xs text-slate-500 mt-1">
              Download the trained browser state (.json) or a Python script (.py) that saves{' '}
              <code className="font-mono text-slate-700">trained_model.pkl</code> for{' '}
              {activeModelName}.
            </p>
          </div>
        </div>

        {/* Pickle Deployment Note */}
        <div className="mt-3.5 rounded-lg border border-indigo-200 bg-indigo-50/70 px-3.5 py-2.5 flex items-start gap-2.5 text-xs text-indigo-900">
          <Package className="w-4 h-4 text-indigo-600 shrink-0 mt-0.5" />
          <span>
            Click <strong>Download .pkl</strong> to compile{' '}
            <code className="font-mono font-semibold">trained_model.pkl</code> directly in your browser (or run the exported{' '}
            <code className="font-mono font-semibold">.py</code> script locally)—ready to load and serve in{' '}
            <strong>FastAPI</strong>, <strong>Flask</strong>, or <strong>Streamlit</strong>.
          </span>
        </div>

        <div className="mt-4 flex flex-wrap items-center justify-between gap-2 pb-4 border-b border-slate-200">
          <div className="inline-flex items-center p-1 rounded-lg bg-slate-100">
            <button
              type="button"
              onClick={() => setActiveTab('python')}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors whitespace-nowrap cursor-pointer ${
                activeTab === 'python'
                  ? 'bg-white text-slate-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Code2 className="w-3.5 h-3.5" />
              Python Script (.py)
            </button>
            <button
              type="button"
              onClick={() => setActiveTab('json')}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors whitespace-nowrap cursor-pointer ${
                activeTab === 'json'
                  ? 'bg-white text-slate-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <FileJson className="w-3.5 h-3.5" />
              Model State (.json)
            </button>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={handleCopyPickleSnippet}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-indigo-200 bg-indigo-50/60 hover:bg-indigo-100/70 text-xs font-medium text-indigo-700 transition-colors whitespace-nowrap cursor-pointer"
            >
              {copiedPickle ? (
                <>
                  <Check className="w-3.5 h-3.5 text-emerald-600" />
                  Copied .pkl Snippet
                </>
              ) : (
                <>
                  <Package className="w-3.5 h-3.5" />
                  Copy Pickle Snippet
                </>
              )}
            </button>

            <button
              type="button"
              onClick={handleCopy}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-200 hover:bg-slate-50 text-xs font-medium text-slate-700 transition-colors whitespace-nowrap cursor-pointer"
            >
              {copied ? (
                <>
                  <Check className="w-3.5 h-3.5 text-emerald-600" />
                  Copied
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  Copy
                </>
              )}
            </button>
          </div>
        </div>

        <div className="mt-4 flex-1 min-h-[360px] max-h-[500px] rounded-lg border border-slate-200 bg-slate-950 text-slate-100 overflow-auto">
          <pre className="p-4 text-xs leading-relaxed font-mono">
            <code>{activeCode}</code>
          </pre>
        </div>

        {pklError && (
          <div className="mt-3 rounded-lg border border-red-200 bg-red-50 px-3.5 py-2.5 flex items-start gap-2 text-xs text-red-700">
            <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
            <span>{pklError}</span>
          </div>
        )}

        <div className="mt-4 pt-4 border-t border-slate-200 flex flex-wrap items-center justify-between gap-3">
          <span className="text-xs text-slate-500">
            Tailored to <span className="font-mono text-slate-700">{run.fileName}</span>
          </span>
          <div className="flex flex-wrap items-center gap-2.5">
            <button
              type="button"
              onClick={() =>
                downloadTextFile(
                  modelJson,
                  `${baseName}_model_state.json`,
                  'application/json'
                )
              }
              className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-lg border border-slate-300 bg-white hover:bg-slate-50 text-xs font-semibold text-slate-800 transition-colors whitespace-nowrap cursor-pointer"
            >
              <Download className="w-3.5 h-3.5" />
              Download .json
            </button>

            <button
              type="button"
              onClick={() =>
                downloadTextFile(
                  pythonScript,
                  `${baseName}_train_pipeline.py`,
                  'text/x-python'
                )
              }
              className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-lg border border-slate-300 bg-white hover:bg-slate-50 text-xs font-semibold text-slate-800 transition-colors whitespace-nowrap cursor-pointer"
            >
              <Download className="w-3.5 h-3.5" />
              Download .py
            </button>

            <button
              type="button"
              disabled={isBuildingPkl}
              onClick={handleDownloadPickle}
              className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-500 disabled:cursor-wait text-white text-xs font-semibold transition-colors whitespace-nowrap cursor-pointer"
            >
              {isBuildingPkl ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin shrink-0" />
                  <span>{pklStatus || 'Initializing Python runtime & building .pkl...'}</span>
                </>
              ) : (
                <>
                  <Package className="w-3.5 h-3.5 shrink-0" />
                  <span>Download .pkl</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
