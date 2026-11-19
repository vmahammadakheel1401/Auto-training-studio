import React from 'react';
import {
  AlertTriangle,
  ArrowRight,
  Award,
  CheckCircle2,
  Clock,
  GitBranch,
  Layers,
  Loader2,
  RotateCcw,
  ShieldAlert,
  Sparkles,
  Target,
} from 'lucide-react';
import {
  SUPERVISED_CLASSIFICATION_CATALOG,
  SUPERVISED_REGRESSION_CATALOG,
  UNSUPERVISED_CATALOG,
} from '../ml/modelSelector';
import {
  AnyRunResult,
  LearningMode,
  ModelAlgorithmKey,
  ParsedDataset,
} from '../ml/types';
import {
  ActualVsPredictedChart,
  AprioriRulesChart,
  ClusterDiagnosticsChart,
  ConfusionMatrixHeatmap,
  FeatureImportanceChart,
  PCAVarianceChart,
  ProjectionScatterChart,
} from './Charts';

interface StepModelOutputProps {
  dataset: ParsedDataset;
  learningMode: LearningMode;
  targetColumn: string;
  droppedIdColumns: string[];
  manualOverrideModel: ModelAlgorithmKey | null;
  isTraining: boolean;
  trainError: string | null;
  run: AnyRunResult | null;
  onChangeLearningMode: (mode: LearningMode) => void;
  onChangeTarget: (target: string) => void;
  onToggleDropColumn: (colName: string) => void;
  onSelectModelOverride: (key: ModelAlgorithmKey | null) => void;
  onProceedToPredict: () => void;
}

export const StepModelOutput: React.FC<StepModelOutputProps> = ({
  dataset,
  learningMode,
  targetColumn,
  droppedIdColumns,
  manualOverrideModel,
  isTraining,
  trainError,
  run,
  onChangeLearningMode,
  onChangeTarget,
  onToggleDropColumn,
  onSelectModelOverride,
  onProceedToPredict,
}) => {
  const autoDetectedIds = dataset.columnStats.filter(
    (c) => c.isLikelyId && (learningMode === 'unsupervised' || c.name !== targetColumn)
  );

  const availableModels =
    learningMode === 'unsupervised'
      ? UNSUPERVISED_CATALOG
      : run?.learningMode === 'supervised' && run.taskType === 'classification'
      ? SUPERVISED_CLASSIFICATION_CATALOG
      : run?.learningMode === 'supervised' && run.taskType === 'regression'
      ? SUPERVISED_REGRESSION_CATALOG
      : SUPERVISED_REGRESSION_CATALOG;

  const selectedModelKey =
    run?.selection.selectedModelKey ?? availableModels[0].key;
  const autoRecommendedKey =
    run?.selection.autoRecommendedModelKey ?? selectedModelKey;

  return (
    <div className="space-y-6">
      {/* Learning Mode, Target & Single-Model Control Bar */}
      <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          {/* Learning Mode Segmented Toggle */}
          <div className="space-y-1">
            <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wide">
              Learning Paradigm
            </div>
            <div className="inline-flex items-center p-1 rounded-lg bg-slate-100 border border-slate-200/70">
              <button
                type="button"
                disabled={isTraining}
                onClick={() => onChangeLearningMode('supervised')}
                className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition-colors cursor-pointer disabled:opacity-50 ${
                  learningMode === 'supervised'
                    ? 'bg-white text-indigo-700 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                <Target className="w-3.5 h-3.5" />
                Supervised (With Target)
              </button>
              <button
                type="button"
                disabled={isTraining}
                onClick={() => onChangeLearningMode('unsupervised')}
                className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition-colors cursor-pointer disabled:opacity-50 ${
                  learningMode === 'unsupervised'
                    ? 'bg-white text-emerald-700 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                <GitBranch className="w-3.5 h-3.5" />
                No Target / Unsupervised
              </button>
            </div>
          </div>

          {/* Target Column (if Supervised) + Single Model Switcher */}
          <div className="flex flex-wrap items-end gap-3 flex-1 justify-end">
            {learningMode === 'supervised' && (
              <div className="min-w-[210px]">
                <label
                  htmlFor="results-target-select"
                  className="block text-[11px] font-semibold text-slate-500 uppercase tracking-wide mb-1"
                >
                  Target Column
                </label>
                <select
                  id="results-target-select"
                  value={targetColumn}
                  disabled={isTraining}
                  onChange={(e) => onChangeTarget(e.target.value)}
                  className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-medium text-slate-900 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/20"
                >
                  {dataset.columnStats.map((col) => (
                    <option key={col.name} value={col.name}>
                      {col.name} ({col.inferredType} · {col.uniqueCount} uniq)
                    </option>
                  ))}
                </select>
              </div>
            )}

            <div className="min-w-[260px]">
              <div className="flex items-center justify-between mb-1">
                <label
                  htmlFor="single-model-select"
                  className="text-[11px] font-semibold text-slate-500 uppercase tracking-wide"
                >
                  Selected Model (Single-Run)
                </label>
                {manualOverrideModel && (
                  <button
                    type="button"
                    disabled={isTraining}
                    onClick={() => onSelectModelOverride(null)}
                    className="inline-flex items-center gap-1 text-[11px] font-semibold text-indigo-600 hover:text-indigo-700 cursor-pointer"
                  >
                    <RotateCcw className="w-3 h-3" />
                    Reset to Auto
                  </button>
                )}
              </div>
              <select
                id="single-model-select"
                value={selectedModelKey}
                disabled={isTraining}
                onChange={(e) => {
                  const chosen = e.target.value as ModelAlgorithmKey;
                  onSelectModelOverride(chosen === autoRecommendedKey ? null : chosen);
                }}
                className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-medium text-slate-900 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/20"
              >
                {availableModels.map((m) => (
                  <option key={m.key} value={m.key}>
                    {m.label}
                    {m.key === autoRecommendedKey ? ' (Auto-Best)' : ''}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {/* Auto-Detected ID Column Exclusion Pills */}
        {autoDetectedIds.length > 0 && (
          <div className="pt-3 border-t border-slate-200 flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-600 mr-1">
              <ShieldAlert className="w-3.5 h-3.5 text-amber-600" />
              ID Columns:
            </span>
            {autoDetectedIds.map((col) => {
              const isDropped = droppedIdColumns.includes(col.name);
              return (
                <button
                  key={col.name}
                  type="button"
                  disabled={isTraining}
                  onClick={() => onToggleDropColumn(col.name)}
                  className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-mono border transition-colors cursor-pointer ${
                    isDropped
                      ? 'bg-slate-100 text-slate-600 border-slate-200 line-through'
                      : 'bg-indigo-50 text-indigo-700 border-indigo-200'
                  }`}
                  title={col.idReason ?? 'Click to toggle inclusion'}
                >
                  <span>{col.name}</span>
                  <span className="text-[10px] font-sans no-underline font-semibold">
                    {isDropped ? 'Excluded' : 'Included'}
                  </span>
                </button>
              );
            })}
          </div>
        )}
      </div>

      {trainError && (
        <div className="flex items-start gap-2.5 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-xs text-red-700">
          <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
          <span>{trainError}</span>
        </div>
      )}

      {isTraining && (
        <div className="bg-white border border-slate-200 rounded-xl p-10 flex flex-col items-center justify-center text-center gap-3">
          <Loader2 className="w-6 h-6 text-indigo-600 animate-spin" />
          <div className="text-sm font-semibold text-slate-900">
            Inspecting metadata & training selected model...
          </div>
          <p className="text-xs text-slate-500">
            Single-model execution active (no exhaustive leaderboard training).
          </p>
        </div>
      )}

      {!isTraining && run && run.learningMode === 'supervised' && (
        <SupervisedOutputView run={run} onProceedToPredict={onProceedToPredict} />
      )}

      {!isTraining && run && run.learningMode === 'unsupervised' && (
        <UnsupervisedOutputView run={run} onProceedToPredict={onProceedToPredict} />
      )}
    </div>
  );
};

// ============================================================================
// Supervised Tailored Output
// ============================================================================

const SupervisedOutputView: React.FC<{
  run: Extract<AnyRunResult, { learningMode: 'supervised' }>;
  onProceedToPredict: () => void;
}> = ({ run, onProceedToPredict }) => {
  const { winner, selection, preprocessor } = run;
  const isRegression = run.taskType === 'regression';
  const primaryScoreLabel = isRegression
    ? `R² = ${winner.primaryScore.toFixed(4)}`
    : `Accuracy = ${(winner.primaryScore * 100).toFixed(1)}%`;

  return (
    <div className="space-y-6">
      {/* Auto-Selection Banner */}
      <div className="rounded-xl border border-emerald-200 bg-emerald-50/70 px-6 py-5 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-start gap-4 max-w-3xl">
          <div className="w-11 h-11 rounded-lg bg-emerald-600 text-white flex items-center justify-center shrink-0">
            <Award className="w-5 h-5" />
          </div>
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-emerald-600 text-white text-[11px] font-semibold">
                <Sparkles className="w-3 h-3" />
                {selection.isManualOverride ? 'Selected Model' : 'Auto-Selected Best Model'}
              </span>
              <span className="text-xs text-emerald-900/80 font-mono">
                {winner.sklearnName} · {winner.family}
              </span>
            </div>
            <h2 className="text-xl font-bold text-slate-900 mt-1">{winner.name}</h2>
            <p className="text-xs text-slate-700 mt-1 leading-relaxed">
              {selection.reason}
            </p>
            <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px] font-mono text-emerald-900/80">
              {selection.signals.map((sig, i) => (
                <span
                  key={i}
                  className="px-2 py-0.5 rounded border border-emerald-200 bg-white/70"
                >
                  {sig}
                </span>
              ))}
              <span className="px-2 py-0.5 rounded border border-emerald-200 bg-white/70">
                Split: {preprocessor.trainCount} train / {preprocessor.testCount} test
              </span>
            </div>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-5">
          <div>
            <div className="text-[11px] font-medium text-emerald-800">Holdout Score</div>
            <div className="text-xl font-bold font-mono tabular-nums text-emerald-950">
              {primaryScoreLabel}
            </div>
          </div>

          <div className="h-9 w-px bg-emerald-200 hidden sm:block" />

          <div>
            <div className="text-[11px] font-medium text-emerald-800">Train Time</div>
            <div className="text-lg font-semibold font-mono tabular-nums text-emerald-950 flex items-center gap-1">
              <Clock className="w-4 h-4 text-emerald-700" />
              {winner.trainingTimeMs} ms
            </div>
          </div>

          <button
            type="button"
            onClick={onProceedToPredict}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            Predict & Export
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Tailored Supervised Metrics Grid */}
      {isRegression ? (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="bg-white border border-slate-200 rounded-xl p-4">
            <div className="text-xs font-medium text-slate-500">R² Score (Test Split)</div>
            <div className="text-2xl font-bold font-mono tabular-nums text-slate-900 mt-1">
              {(winner.metrics.r2 ?? 0).toFixed(4)}
            </div>
            <div className="text-xs text-slate-500 mt-1">
              Proportion of target variance explained
            </div>
          </div>
          <div className="bg-white border border-slate-200 rounded-xl p-4">
            <div className="text-xs font-medium text-slate-500">RMSE</div>
            <div className="text-2xl font-bold font-mono tabular-nums text-slate-900 mt-1">
              {(winner.metrics.rmse ?? 0).toLocaleString(undefined, {
                maximumFractionDigits: 2,
              })}
            </div>
            <div className="text-xs text-slate-500 mt-1">
              Root mean squared error in {run.targetColumn} units
            </div>
          </div>
          <div className="bg-white border border-slate-200 rounded-xl p-4">
            <div className="text-xs font-medium text-slate-500">MAE</div>
            <div className="text-2xl font-bold font-mono tabular-nums text-slate-900 mt-1">
              {(winner.metrics.mae ?? 0).toLocaleString(undefined, {
                maximumFractionDigits: 2,
              })}
            </div>
            <div className="text-xs text-slate-500 mt-1">
              Mean absolute error across {preprocessor.testCount} holdout rows
            </div>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-white border border-slate-200 rounded-xl p-4">
            <div className="text-xs font-medium text-slate-500">Accuracy</div>
            <div className="text-2xl font-bold font-mono tabular-nums text-slate-900 mt-1">
              {((winner.metrics.accuracy ?? 0) * 100).toFixed(1)}%
            </div>
            <div className="text-xs text-slate-500 mt-1">Holdout classification accuracy</div>
          </div>
          <div className="bg-white border border-slate-200 rounded-xl p-4">
            <div className="text-xs font-medium text-slate-500">Macro Precision</div>
            <div className="text-2xl font-bold font-mono tabular-nums text-slate-900 mt-1">
              {((winner.metrics.precision ?? 0) * 100).toFixed(1)}%
            </div>
            <div className="text-xs text-slate-500 mt-1">Positive predictive precision</div>
          </div>
          <div className="bg-white border border-slate-200 rounded-xl p-4">
            <div className="text-xs font-medium text-slate-500">Macro Recall</div>
            <div className="text-2xl font-bold font-mono tabular-nums text-slate-900 mt-1">
              {((winner.metrics.recall ?? 0) * 100).toFixed(1)}%
            </div>
            <div className="text-xs text-slate-500 mt-1">Class sensitivity across labels</div>
          </div>
          <div className="bg-white border border-slate-200 rounded-xl p-4">
            <div className="text-xs font-medium text-slate-500">Macro F1 Score</div>
            <div className="text-2xl font-bold font-mono tabular-nums text-slate-900 mt-1">
              {((winner.metrics.f1 ?? 0) * 100).toFixed(1)}%
            </div>
            <div className="text-xs text-slate-500 mt-1">Harmonic precision-recall balance</div>
          </div>
        </div>
      )}

      {/* Side-by-Side Diagnostics */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        <div className="lg:col-span-7 bg-white border border-slate-200 rounded-xl p-6">
          {isRegression ? (
            <ActualVsPredictedChart
              points={winner.regressionPoints ?? []}
              targetColumn={run.targetColumn}
              modelName={winner.name}
            />
          ) : (
            <ConfusionMatrixHeatmap
              data={
                winner.confusionMatrix ?? {
                  labels: preprocessor.classLabels ?? [],
                  matrix: [],
                }
              }
              modelName={winner.name}
            />
          )}
        </div>

        <div className="lg:col-span-5 bg-white border border-slate-200 rounded-xl p-6">
          <FeatureImportanceChart
            importances={winner.featureImportances}
            modelName={winner.name}
          />
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// Unsupervised Tailored Output (PCA, Clustering, Apriori)
// ============================================================================

const UnsupervisedOutputView: React.FC<{
  run: Extract<AnyRunResult, { learningMode: 'unsupervised' }>;
  onProceedToPredict: () => void;
}> = ({ run, onProceedToPredict }) => {
  const { model, selection, preprocessor } = run;

  const headerHighlight = (() => {
    if (model.key === 'pca') {
      return `2D Var = ${((model.totalVarianceExplained2D ?? 0) * 100).toFixed(1)}%`;
    }
    if (model.key === 'apriori') {
      return `${model.associationRules?.length ?? 0} Rules Mined`;
    }
    return `Silhouette = ${(model.silhouetteScore ?? 0).toFixed(3)}`;
  })();

  return (
    <div className="space-y-6">
      {/* Unsupervised Auto-Selection Banner */}
      <div className="rounded-xl border border-emerald-200 bg-emerald-50/70 px-6 py-5 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-start gap-4 max-w-3xl">
          <div className="w-11 h-11 rounded-lg bg-emerald-600 text-white flex items-center justify-center shrink-0">
            <CheckCircle2 className="w-5 h-5" />
          </div>
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-emerald-600 text-white text-[11px] font-semibold">
                <Sparkles className="w-3 h-3" />
                {selection.isManualOverride
                  ? 'Selected Unsupervised Model'
                  : 'Auto-Selected Unsupervised Model'}
              </span>
              <span className="text-xs text-emerald-900/80 font-mono">
                {model.sklearnName} · {model.family}
              </span>
            </div>
            <h2 className="text-xl font-bold text-slate-900 mt-1">{model.name}</h2>
            <p className="text-xs text-slate-700 mt-1 leading-relaxed">
              {selection.reason}
            </p>
            <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px] font-mono text-emerald-900/80">
              {selection.signals.map((sig, i) => (
                <span
                  key={i}
                  className="px-2 py-0.5 rounded border border-emerald-200 bg-white/70"
                >
                  {sig}
                </span>
              ))}
              <span className="px-2 py-0.5 rounded border border-emerald-200 bg-white/70">
                {preprocessor.encodedFeatureCount} encoded dims
              </span>
            </div>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-5">
          <div>
            <div className="text-[11px] font-medium text-emerald-800">Summary</div>
            <div className="text-xl font-bold font-mono tabular-nums text-emerald-950">
              {headerHighlight}
            </div>
          </div>

          <div className="h-9 w-px bg-emerald-200 hidden sm:block" />

          <div>
            <div className="text-[11px] font-medium text-emerald-800">Run Time</div>
            <div className="text-lg font-semibold font-mono tabular-nums text-emerald-950 flex items-center gap-1">
              <Clock className="w-4 h-4 text-emerald-700" />
              {model.trainingTimeMs} ms
            </div>
          </div>

          <button
            type="button"
            onClick={onProceedToPredict}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            {model.key === 'apriori' ? 'Basket Recommender & Export' : 'Project / Assign & Export'}
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* 1) PCA Output */}
      {model.key === 'pca' && (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="bg-white border border-slate-200 rounded-xl p-4">
              <div className="text-xs font-medium text-slate-500">PC1 Variance</div>
              <div className="text-2xl font-bold font-mono tabular-nums text-slate-900 mt-1">
                {((model.pcaComponents?.[0]?.explainedVarianceRatio ?? 0) * 100).toFixed(1)}%
              </div>
              <div className="text-xs text-slate-500 mt-1">Primary axis variance share</div>
            </div>
            <div className="bg-white border border-slate-200 rounded-xl p-4">
              <div className="text-xs font-medium text-slate-500">2D Variance (PC1+PC2)</div>
              <div className="text-2xl font-bold font-mono tabular-nums text-slate-900 mt-1">
                {((model.totalVarianceExplained2D ?? 0) * 100).toFixed(1)}%
              </div>
              <div className="text-xs text-slate-500 mt-1">Total signal preserved in 2D</div>
            </div>
            <div className="bg-white border border-slate-200 rounded-xl p-4">
              <div className="text-xs font-medium text-slate-500">Components Extracted</div>
              <div className="text-2xl font-bold font-mono tabular-nums text-slate-900 mt-1">
                {model.pcaComponents?.length ?? 0}
              </div>
              <div className="text-xs text-slate-500 mt-1">
                Compressed from {preprocessor.encodedFeatureCount} encoded features
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
            <div className="lg:col-span-7 bg-white border border-slate-200 rounded-xl p-6">
              <ProjectionScatterChart
                points={model.projectionPoints ?? []}
                axisLabels={model.projectionAxisLabels ?? ['PC1', 'PC2']}
                title="2D Principal Component Projection"
                subtitle={`${preprocessor.rowCount} unlabeled samples projected onto orthogonal PC1 & PC2`}
              />
            </div>
            <div className="lg:col-span-5 bg-white border border-slate-200 rounded-xl p-6">
              <PCAVarianceChart components={model.pcaComponents ?? []} />
            </div>
          </div>

          <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
            <div className="px-6 py-4 border-b border-slate-200 flex items-center gap-2">
              <Layers className="w-4 h-4 text-indigo-600" />
              <h3 className="text-sm font-semibold text-slate-900">
                Principal Component Loadings (Top Feature Drivers)
              </h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-slate-200 bg-slate-50/70 text-xs font-semibold text-slate-600">
                    <th className="py-3 px-5">Component</th>
                    <th className="py-3 px-5 text-right">Eigenvalue</th>
                    <th className="py-3 px-5 text-right">Explained Var</th>
                    <th className="py-3 px-5 text-right">Cumulative</th>
                    <th className="py-3 px-5">Top Contributing Features (Loadings)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 text-xs">
                  {(model.pcaComponents ?? []).map((comp) => (
                    <tr key={comp.component} className="hover:bg-slate-50/70">
                      <td className="py-3 px-5 font-mono font-bold text-slate-900">
                        {comp.component}
                      </td>
                      <td className="py-3 px-5 text-right font-mono tabular-nums text-slate-700">
                        {comp.eigenvalue.toFixed(3)}
                      </td>
                      <td className="py-3 px-5 text-right font-mono tabular-nums font-semibold text-indigo-600">
                        {(comp.explainedVarianceRatio * 100).toFixed(1)}%
                      </td>
                      <td className="py-3 px-5 text-right font-mono tabular-nums text-slate-700">
                        {(comp.cumulativeVarianceRatio * 100).toFixed(1)}%
                      </td>
                      <td className="py-3 px-5 font-mono text-slate-700">
                        {comp.topLoadings
                          .slice(0, 4)
                          .map((l) => `${l.feature} (${l.loading >= 0 ? '+' : ''}${l.loading.toFixed(2)})`)
                          .join(', ')}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* 2) Clustering Output (K-Means, Hierarchical, GMM) */}
      {(model.key === 'kmeans' || model.key === 'hierarchical' || model.key === 'gmm') && (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="bg-white border border-slate-200 rounded-xl p-4">
              <div className="text-xs font-medium text-slate-500">Clusters Discovered</div>
              <div className="text-2xl font-bold font-mono tabular-nums text-slate-900 mt-1">
                {model.numClusters ?? 0}
              </div>
              <div className="text-xs text-slate-500 mt-1">
                Automatically selected via cluster quality
              </div>
            </div>
            <div className="bg-white border border-slate-200 rounded-xl p-4">
              <div className="text-xs font-medium text-slate-500">Silhouette Score</div>
              <div className="text-2xl font-bold font-mono tabular-nums text-slate-900 mt-1">
                {(model.silhouetteScore ?? 0).toFixed(4)}
              </div>
              <div className="text-xs text-slate-500 mt-1">
                Cohesion vs separation (-1 to +1)
              </div>
            </div>
            <div className="bg-white border border-slate-200 rounded-xl p-4">
              <div className="text-xs font-medium text-slate-500">
                {model.key === 'gmm'
                  ? 'BIC Score'
                  : model.key === 'kmeans'
                  ? 'Inertia (WCSS)'
                  : 'Linkage Criterion'}
              </div>
              <div className="text-2xl font-bold font-mono tabular-nums text-slate-900 mt-1">
                {model.key === 'gmm'
                  ? (model.bic ?? 0).toLocaleString(undefined, { maximumFractionDigits: 1 })
                  : model.key === 'kmeans'
                  ? (model.inertia ?? 0).toLocaleString(undefined, { maximumFractionDigits: 1 })
                  : 'Ward'}
              </div>
              <div className="text-xs text-slate-500 mt-1">
                {model.key === 'gmm'
                  ? `Log-Likelihood: ${(model.logLikelihood ?? 0).toFixed(1)}`
                  : model.key === 'kmeans'
                  ? 'Within-cluster sum of squares'
                  : 'Minimum variance agglomerative linkage'}
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
            <div className="lg:col-span-7 bg-white border border-slate-200 rounded-xl p-6">
              <ProjectionScatterChart
                points={model.projectionPoints ?? []}
                axisLabels={model.projectionAxisLabels ?? ['PC1', 'PC2']}
                title="Cluster Assignments in 2D PCA Space"
                subtitle={`${preprocessor.rowCount} samples colored by ${model.name}`}
              />
            </div>
            <div className="lg:col-span-5 bg-white border border-slate-200 rounded-xl p-6">
              <ClusterDiagnosticsChart
                kDiagnostics={model.kDiagnostics}
                mergeHistory={model.mergeHistory}
                selectedK={model.numClusters}
                modelName={model.name}
              />
            </div>
          </div>

          {/* Cluster Profiles Table */}
          <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
            <div className="px-6 py-4 border-b border-slate-200 flex items-center gap-2">
              <Layers className="w-4 h-4 text-indigo-600" />
              <h3 className="text-sm font-semibold text-slate-900">
                Cluster Segment Profiles
              </h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-slate-200 bg-slate-50/70 text-xs font-semibold text-slate-600">
                    <th className="py-3 px-5">Segment</th>
                    <th className="py-3 px-5 text-right">Size</th>
                    <th className="py-3 px-5 text-right">Share</th>
                    <th className="py-3 px-5">Distinctive Centroid Averages / Modes</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 text-xs">
                  {(model.clusterProfiles ?? []).map((prof) => (
                    <tr key={prof.clusterId} className="hover:bg-slate-50/70">
                      <td className="py-3.5 px-5 font-semibold text-slate-900">
                        {prof.label}
                      </td>
                      <td className="py-3.5 px-5 text-right font-mono tabular-nums text-slate-700">
                        {prof.count} rows
                      </td>
                      <td className="py-3.5 px-5 text-right font-mono tabular-nums font-semibold text-indigo-600">
                        {prof.percentage}%
                      </td>
                      <td className="py-3.5 px-5 text-slate-700 font-mono">
                        {prof.topFeatures
                          .map((f) => `${f.feature}: ${f.meanLabel}`)
                          .join(' · ')}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* 3) Apriori Association Rules Output */}
      {model.key === 'apriori' && (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-white border border-slate-200 rounded-xl p-4">
              <div className="text-xs font-medium text-slate-500">Transactions</div>
              <div className="text-2xl font-bold font-mono tabular-nums text-slate-900 mt-1">
                {model.totalTransactions ?? 0}
              </div>
              <div className="text-xs text-slate-500 mt-1">
                Avg {model.avgBasketSize ?? 0} items / basket
              </div>
            </div>
            <div className="bg-white border border-slate-200 rounded-xl p-4">
              <div className="text-xs font-medium text-slate-500">Frequent Itemsets</div>
              <div className="text-2xl font-bold font-mono tabular-nums text-slate-900 mt-1">
                {model.frequentItemsets?.length ?? 0}
              </div>
              <div className="text-xs text-slate-500 mt-1">
                Min support: {((model.minSupportUsed ?? 0.1) * 100).toFixed(0)}%
              </div>
            </div>
            <div className="bg-white border border-slate-200 rounded-xl p-4">
              <div className="text-xs font-medium text-slate-500">Rules Discovered</div>
              <div className="text-2xl font-bold font-mono tabular-nums text-slate-900 mt-1">
                {model.associationRules?.length ?? 0}
              </div>
              <div className="text-xs text-slate-500 mt-1">
                Across {model.uniqueItemsCount ?? 0} unique items
              </div>
            </div>
            <div className="bg-white border border-slate-200 rounded-xl p-4">
              <div className="text-xs font-medium text-slate-500">Highest Lift</div>
              <div className="text-2xl font-bold font-mono tabular-nums text-slate-900 mt-1">
                {(model.associationRules?.[0]?.lift ?? 1).toFixed(2)}×
              </div>
              <div className="text-xs text-slate-500 mt-1">
                Top confidence:{' '}
                {((model.associationRules?.[0]?.confidence ?? 0) * 100).toFixed(0)}%
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
            <div className="lg:col-span-6 bg-white border border-slate-200 rounded-xl p-6">
              <AprioriRulesChart rules={model.associationRules ?? []} />
            </div>

            <div className="lg:col-span-6 bg-white border border-slate-200 rounded-xl p-6">
              <div className="mb-3">
                <h3 className="text-sm font-semibold text-slate-900">
                  Top Frequent Itemsets
                </h3>
                <p className="text-xs text-slate-500">
                  Most common co-occurring item bundles across transactions
                </p>
              </div>
              <div className="space-y-2.5 max-h-[290px] overflow-y-auto pr-1">
                {(model.frequentItemsets ?? []).slice(0, 12).map((fs, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between gap-3 rounded-lg border border-slate-200 bg-slate-50/50 px-3 py-2 text-xs"
                  >
                    <span className="font-mono font-medium text-slate-800 truncate">
                      {fs.items.join(' + ')}
                    </span>
                    <span className="font-mono tabular-nums text-slate-600 shrink-0">
                      {(fs.support * 100).toFixed(1)}% ({fs.count})
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Association Rules Table */}
          <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
            <div className="px-6 py-4 border-b border-slate-200 flex items-center gap-2">
              <Layers className="w-4 h-4 text-indigo-600" />
              <h3 className="text-sm font-semibold text-slate-900">
                Discovered Association Rules
              </h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-slate-200 bg-slate-50/70 text-xs font-semibold text-slate-600">
                    <th className="py-3 px-5">Antecedent (If Basket Has)</th>
                    <th className="py-3 px-5">Consequent (Likely Added)</th>
                    <th className="py-3 px-5 text-right">Support</th>
                    <th className="py-3 px-5 text-right">Confidence</th>
                    <th className="py-3 px-5 text-right">Lift</th>
                    <th className="py-3 px-5 text-right">Conviction</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 text-xs">
                  {(model.associationRules ?? []).slice(0, 20).map((rule, idx) => (
                    <tr key={idx} className="hover:bg-slate-50/70">
                      <td className="py-3 px-5 font-mono font-medium text-slate-900">
                        {rule.antecedents.join(' + ')}
                      </td>
                      <td className="py-3 px-5 font-mono font-semibold text-indigo-600">
                        → {rule.consequents.join(' + ')}
                      </td>
                      <td className="py-3 px-5 text-right font-mono tabular-nums text-slate-700">
                        {(rule.support * 100).toFixed(1)}%
                      </td>
                      <td className="py-3 px-5 text-right font-mono tabular-nums font-semibold text-slate-900">
                        {(rule.confidence * 100).toFixed(1)}%
                      </td>
                      <td className="py-3 px-5 text-right font-mono tabular-nums font-semibold text-emerald-700">
                        {rule.lift.toFixed(2)}×
                      </td>
                      <td className="py-3 px-5 text-right font-mono tabular-nums text-slate-600">
                        {rule.conviction.toFixed(2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
};
