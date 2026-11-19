import React, { useState } from 'react';
import {
  AssociationRule,
  ConfusionMatrixData,
  FeatureImportanceItem,
  HierarchicalMergeStep,
  KDiagnosticPoint,
  PCAComponentInfo,
  ProjectionPoint,
  RegressionPoint,
} from '../ml/types';

interface ScatterChartProps {
  points: RegressionPoint[];
  targetColumn: string;
  modelName: string;
}

function formatCompactNumber(n: number): string {
  const abs = Math.abs(n);
  if (abs >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (abs >= 10_000) return `${Math.round(n / 1000)}k`;
  if (abs >= 1000) return `${(n / 1000).toFixed(1)}k`;
  if (Number.isInteger(n)) return String(n);
  return n.toFixed(1);
}

export const ActualVsPredictedChart: React.FC<ScatterChartProps> = ({
  points,
  targetColumn,
  modelName,
}) => {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  if (points.length === 0) {
    return (
      <div className="h-72 flex items-center justify-center text-sm text-slate-500">
        No holdout test predictions available.
      </div>
    );
  }

  const allVals = points.flatMap((p) => [p.actual, p.predicted]);
  const rawMin = Math.min(...allVals);
  const rawMax = Math.max(...allVals);
  const pad = Math.max(1e-6, (rawMax - rawMin) * 0.08);
  const minVal = rawMin - pad;
  const maxVal = rawMax + pad;
  const span = Math.max(1e-6, maxVal - minVal);

  const width = 520;
  const height = 310;
  const margin = { top: 20, right: 24, bottom: 44, left: 60 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;

  const toX = (v: number) => margin.left + ((v - minVal) / span) * plotW;
  const toY = (v: number) => margin.top + plotH - ((v - minVal) / span) * plotH;

  const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => minVal + t * span);
  const hovered = hoveredIdx !== null ? points[hoveredIdx] : null;

  return (
    <div className="relative">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="text-sm font-semibold text-slate-900">
            Actual vs. Predicted ({targetColumn})
          </h3>
          <p className="text-xs text-slate-500">
            Holdout 20% test split for {modelName} · Diagonal dashed line marks ideal y = x
          </p>
        </div>
      </div>

      <div className="relative w-full">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-auto overflow-visible select-none"
          role="img"
          aria-label="Actual versus predicted scatter plot"
        >
          {/* Grid lines */}
          {ticks.map((tick, idx) => {
            const x = toX(tick);
            const y = toY(tick);
            return (
              <g key={idx}>
                <line
                  x1={margin.left}
                  y1={y}
                  x2={width - margin.right}
                  y2={y}
                  stroke="#E2E8F0"
                  strokeWidth={1}
                />
                <line
                  x1={x}
                  y1={margin.top}
                  x2={x}
                  y2={height - margin.bottom}
                  stroke="#F1F5F9"
                  strokeWidth={1}
                />
                <text
                  x={margin.left - 8}
                  y={y + 4}
                  textAnchor="end"
                  className="fill-slate-500 text-[10px] font-mono tabular-nums"
                >
                  {formatCompactNumber(tick)}
                </text>
                <text
                  x={x}
                  y={height - margin.bottom + 16}
                  textAnchor="middle"
                  className="fill-slate-500 text-[10px] font-mono tabular-nums"
                >
                  {formatCompactNumber(tick)}
                </text>
              </g>
            );
          })}

          {/* Axes */}
          <line
            x1={margin.left}
            y1={margin.top}
            x2={margin.left}
            y2={height - margin.bottom}
            stroke="#CBD5E1"
            strokeWidth={1.25}
          />
          <line
            x1={margin.left}
            y1={height - margin.bottom}
            x2={width - margin.right}
            y2={height - margin.bottom}
            stroke="#CBD5E1"
            strokeWidth={1.25}
          />

          {/* Ideal y = x diagonal */}
          <line
            x1={toX(minVal)}
            y1={toY(minVal)}
            x2={toX(maxVal)}
            y2={toY(maxVal)}
            stroke="#10B981"
            strokeWidth={1.75}
            strokeDasharray="5 4"
          />

          {/* Data Points */}
          {points.map((pt, idx) => {
            const cx = toX(pt.actual);
            const cy = toY(pt.predicted);
            const isHovered = hoveredIdx === idx;
            return (
              <circle
                key={idx}
                cx={cx}
                cy={cy}
                r={isHovered ? 6 : 4.2}
                fill={isHovered ? '#4F46E5' : '#6366F1'}
                fillOpacity={isHovered ? 1 : 0.72}
                stroke="#FFFFFF"
                strokeWidth={1.25}
                className="cursor-pointer transition-transform duration-150"
                onMouseEnter={() => setHoveredIdx(idx)}
                onMouseLeave={() => setHoveredIdx(null)}
              />
            );
          })}

          {/* Axis Labels */}
          <text
            x={margin.left + plotW / 2}
            y={height - 6}
            textAnchor="middle"
            className="fill-slate-600 text-[11px] font-medium"
          >
            Actual {targetColumn}
          </text>
          <text
            x={14}
            y={margin.top + plotH / 2}
            textAnchor="middle"
            transform={`rotate(-90 14 ${margin.top + plotH / 2})`}
            className="fill-slate-600 text-[11px] font-medium"
          >
            Predicted {targetColumn}
          </text>
        </svg>

        {/* Interactive Tooltip */}
        {hovered && (
          <div className="mt-2 flex flex-wrap items-center justify-between gap-3 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-700">
            <span className="font-medium text-slate-900">
              Test Sample #{hovered.rowIndex}
            </span>
            <div className="flex items-center gap-4 font-mono tabular-nums">
              <span>
                Actual:{' '}
                <strong className="text-slate-900">
                  {hovered.actual.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                </strong>
              </span>
              <span>
                Predicted:{' '}
                <strong className="text-indigo-600">
                  {hovered.predicted.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                </strong>
              </span>
              <span>
                Error:{' '}
                <strong className="text-slate-700">
                  {(hovered.predicted - hovered.actual >= 0 ? '+' : '') +
                    (hovered.predicted - hovered.actual).toLocaleString(undefined, {
                      maximumFractionDigits: 2,
                    })}
                </strong>
              </span>
            </div>
          </div>
        )}
        {!hovered && (
          <div className="mt-2 rounded-lg border border-transparent px-3 py-2 text-xs text-slate-400">
            Hover any point to inspect actual vs. predicted values and residual error.
          </div>
        )}
      </div>
    </div>
  );
};

interface ConfusionMatrixProps {
  data: ConfusionMatrixData;
  modelName: string;
}

export const ConfusionMatrixHeatmap: React.FC<ConfusionMatrixProps> = ({
  data,
  modelName,
}) => {
  const [hoveredCell, setHoveredCell] = useState<{
    actual: string;
    predicted: string;
    count: number;
    pct: string;
    isDiagonal: boolean;
  } | null>(null);

  const { labels, matrix } = data;
  const maxCell = Math.max(1, ...matrix.flat());
  const totalSamples = Math.max(
    1,
    matrix.flat().reduce((a, b) => a + b, 0)
  );

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="text-sm font-semibold text-slate-900">
            Confusion Matrix Heatmap
          </h3>
          <p className="text-xs text-slate-500">
            Holdout 20% test split ({totalSamples} samples) for {modelName}
          </p>
        </div>
      </div>

      <div className="overflow-x-auto">
        <div className="inline-block min-w-full">
          <div className="flex items-center justify-center mb-2 text-xs font-medium text-slate-600">
            Predicted Class
          </div>

          <div className="flex items-stretch gap-3">
            <div className="flex items-center justify-center">
              <span className="text-xs font-medium text-slate-600 [writing-mode:vertical-rl] rotate-180">
                Actual Class
              </span>
            </div>

            <div className="flex-1">
              <div
                className="grid gap-2"
                style={{
                  gridTemplateColumns: `minmax(76px, auto) repeat(${labels.length}, minmax(72px, 1fr))`,
                }}
              >
                <div />
                {labels.map((predLabel) => (
                  <div
                    key={predLabel}
                    className="text-center text-xs font-semibold text-slate-700 truncate px-1 py-1"
                    title={predLabel}
                  >
                    {predLabel}
                  </div>
                ))}

                {labels.map((actualLabel, rIdx) => {
                  const rowSum = Math.max(
                    1,
                    matrix[rIdx].reduce((a, b) => a + b, 0)
                  );
                  return (
                    <React.Fragment key={actualLabel}>
                      <div
                        className="flex items-center justify-end pr-2 text-xs font-semibold text-slate-700 truncate"
                        title={actualLabel}
                      >
                        {actualLabel}
                      </div>

                      {labels.map((predLabel, cIdx) => {
                        const count = matrix[rIdx][cIdx];
                        const intensity = count / maxCell;
                        const rowPct = ((count / rowSum) * 100).toFixed(1);
                        const isDiag = rIdx === cIdx;

                        const bgStyle = isDiag
                          ? `rgba(16, 185, 129, ${0.12 + intensity * 0.72})`
                          : count > 0
                          ? `rgba(79, 70, 229, ${0.08 + intensity * 0.45})`
                          : '#F8FAFC';

                        const darkText = isDiag
                          ? intensity > 0.62
                            ? 'text-white'
                            : 'text-emerald-950'
                          : intensity > 0.65 && count > 0
                          ? 'text-white'
                          : 'text-slate-800';

                        return (
                          <button
                            key={`${rIdx}-${cIdx}`}
                            type="button"
                            onMouseEnter={() =>
                              setHoveredCell({
                                actual: actualLabel,
                                predicted: predLabel,
                                count,
                                pct: rowPct,
                                isDiagonal: isDiag,
                              })
                            }
                            onMouseLeave={() => setHoveredCell(null)}
                            style={{ backgroundColor: bgStyle }}
                            className={`rounded-lg border px-3 py-4 text-center transition-all focus-visible:outline-2 focus-visible:outline-indigo-600 ${
                              isDiag
                                ? 'border-emerald-300/80'
                                : 'border-slate-200'
                            } ${darkText}`}
                          >
                            <div className="text-lg font-bold font-mono tabular-nums leading-tight">
                              {count}
                            </div>
                            <div className="text-[11px] font-mono tabular-nums opacity-80 mt-0.5">
                              {rowPct}%
                            </div>
                          </button>
                        );
                      })}
                    </React.Fragment>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-3 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600">
        {hoveredCell ? (
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span>
              Actual <strong className="text-slate-900">{hoveredCell.actual}</strong>{' '}
              → Predicted{' '}
              <strong className="text-slate-900">{hoveredCell.predicted}</strong>
            </span>
            <span className="font-mono tabular-nums">
              <strong className="text-slate-900">{hoveredCell.count}</strong> samples (
              {hoveredCell.pct}% of actual class ·{' '}
              {hoveredCell.isDiagonal ? 'Correct classification' : 'Misclassified'})
            </span>
          </div>
        ) : (
          <span>
            Diagonal emerald cells indicate correct predictions; off-diagonal cells show misclassifications.
          </span>
        )}
      </div>
    </div>
  );
};

interface FeatureImportanceChartProps {
  importances: FeatureImportanceItem[];
  modelName: string;
}

export const FeatureImportanceChart: React.FC<FeatureImportanceChartProps> = ({
  importances,
  modelName,
}) => {
  const [hoveredFeat, setHoveredFeat] = useState<string | null>(null);
  const topItems = importances.slice(0, 8);
  const maxVal = Math.max(1, ...topItems.map((i) => i.importance));

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="text-sm font-semibold text-slate-900">
            Feature Importance
          </h3>
          <p className="text-xs text-slate-500">
            Relative contribution (%) computed from {modelName}
          </p>
        </div>
      </div>

      {topItems.length === 0 ? (
        <div className="h-60 flex items-center justify-center text-sm text-slate-500">
          No feature importances available.
        </div>
      ) : (
        <div className="space-y-3 mt-4">
          {topItems.map((item, idx) => {
            const barWidth = Math.max(2, (item.importance / maxVal) * 100);
            const isHovered = hoveredFeat === item.feature;
            const isTop = idx === 0;
            return (
              <div
                key={item.feature}
                onMouseEnter={() => setHoveredFeat(item.feature)}
                onMouseLeave={() => setHoveredFeat(null)}
                className={`rounded-lg px-2.5 py-1.5 transition-colors ${
                  isHovered ? 'bg-slate-50' : ''
                }`}
              >
                <div className="flex items-center justify-between text-xs mb-1.5">
                  <span
                    className="font-medium text-slate-800 truncate pr-3"
                    title={item.feature}
                  >
                    {item.feature}
                  </span>
                  <span className="font-mono tabular-nums font-semibold text-slate-900 shrink-0">
                    {item.importance.toFixed(1)}%
                  </span>
                </div>
                <div className="h-2.5 w-full rounded-full bg-slate-100 overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-300 ${
                      isTop ? 'bg-indigo-600' : 'bg-indigo-400'
                    }`}
                    style={{ width: `${barWidth}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

// ============================================================================
// Unsupervised Visualizations
// ============================================================================

const CLUSTER_COLORS = [
  '#4F46E5', // Indigo
  '#10B981', // Emerald
  '#F59E0B', // Amber
  '#EC4899', // Pink
  '#06B6D4', // Cyan
  '#8B5CF6', // Violet
];

interface ProjectionScatterProps {
  points: ProjectionPoint[];
  axisLabels: [string, string];
  title: string;
  subtitle: string;
}

export const ProjectionScatterChart: React.FC<ProjectionScatterProps> = ({
  points,
  axisLabels,
  title,
  subtitle,
}) => {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  if (points.length === 0) {
    return (
      <div className="h-72 flex items-center justify-center text-sm text-slate-500">
        No 2D projection points available.
      </div>
    );
  }

  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const padX = Math.max(1e-4, (maxX - minX) * 0.1);
  const padY = Math.max(1e-4, (maxY - minY) * 0.1);
  const x0 = minX - padX;
  const x1 = maxX + padX;
  const y0 = minY - padY;
  const y1 = maxY + padY;

  const width = 520;
  const height = 310;
  const margin = { top: 20, right: 24, bottom: 44, left: 52 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;

  const toX = (v: number) => margin.left + ((v - x0) / Math.max(1e-6, x1 - x0)) * plotW;
  const toY = (v: number) => margin.top + plotH - ((v - y0) / Math.max(1e-6, y1 - y0)) * plotH;

  const xTicks = [0, 0.25, 0.5, 0.75, 1].map((t) => x0 + t * (x1 - x0));
  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((t) => y0 + t * (y1 - y0));

  const clusters = Array.from(
    new Set(points.map((p) => p.cluster).filter((c): c is number => c !== undefined))
  ).sort((a, b) => a - b);

  const hovered = hoveredIdx !== null ? points[hoveredIdx] : null;

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
        <div>
          <h3 className="text-sm font-semibold text-slate-900">{title}</h3>
          <p className="text-xs text-slate-500">{subtitle}</p>
        </div>
        {clusters.length > 0 && (
          <div className="flex flex-wrap items-center gap-2.5">
            {clusters.map((c) => (
              <span
                key={c}
                className="inline-flex items-center gap-1.5 text-[11px] font-medium text-slate-700"
              >
                <span
                  className="w-2.5 h-2.5 rounded-full"
                  style={{ backgroundColor: CLUSTER_COLORS[c % CLUSTER_COLORS.length] }}
                />
                C{c + 1}
              </span>
            ))}
          </div>
        )}
      </div>

      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full h-auto overflow-visible select-none"
        role="img"
        aria-label={title}
      >
        {yTicks.map((tick, idx) => (
          <g key={`y-${idx}`}>
            <line
              x1={margin.left}
              y1={toY(tick)}
              x2={width - margin.right}
              y2={toY(tick)}
              stroke="#E2E8F0"
              strokeWidth={1}
            />
            <text
              x={margin.left - 8}
              y={toY(tick) + 4}
              textAnchor="end"
              className="fill-slate-500 text-[10px] font-mono tabular-nums"
            >
              {tick.toFixed(1)}
            </text>
          </g>
        ))}

        {xTicks.map((tick, idx) => (
          <g key={`x-${idx}`}>
            <line
              x1={toX(tick)}
              y1={margin.top}
              x2={toX(tick)}
              y2={height - margin.bottom}
              stroke="#F1F5F9"
              strokeWidth={1}
            />
            <text
              x={toX(tick)}
              y={height - margin.bottom + 16}
              textAnchor="middle"
              className="fill-slate-500 text-[10px] font-mono tabular-nums"
            >
              {tick.toFixed(1)}
            </text>
          </g>
        ))}

        <line
          x1={margin.left}
          y1={margin.top}
          x2={margin.left}
          y2={height - margin.bottom}
          stroke="#CBD5E1"
          strokeWidth={1.25}
        />
        <line
          x1={margin.left}
          y1={height - margin.bottom}
          x2={width - margin.right}
          y2={height - margin.bottom}
          stroke="#CBD5E1"
          strokeWidth={1.25}
        />

        {points.map((pt, idx) => {
          const cx = toX(pt.x);
          const cy = toY(pt.y);
          const color =
            pt.cluster !== undefined
              ? CLUSTER_COLORS[pt.cluster % CLUSTER_COLORS.length]
              : '#4F46E5';
          const isHovered = hoveredIdx === idx;
          return (
            <circle
              key={idx}
              cx={cx}
              cy={cy}
              r={isHovered ? 6.2 : 4}
              fill={color}
              fillOpacity={isHovered ? 1 : 0.76}
              stroke="#FFFFFF"
              strokeWidth={1.1}
              className="cursor-pointer"
              onMouseEnter={() => setHoveredIdx(idx)}
              onMouseLeave={() => setHoveredIdx(null)}
            />
          );
        })}

        <text
          x={margin.left + plotW / 2}
          y={height - 6}
          textAnchor="middle"
          className="fill-slate-600 text-[11px] font-medium"
        >
          {axisLabels[0]}
        </text>
        <text
          x={14}
          y={margin.top + plotH / 2}
          textAnchor="middle"
          transform={`rotate(-90 14 ${margin.top + plotH / 2})`}
          className="fill-slate-600 text-[11px] font-medium"
        >
          {axisLabels[1]}
        </text>
      </svg>

      <div className="mt-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600">
        {hovered ? (
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="font-semibold text-slate-900">
              Row #{hovered.rowIndex}
              {hovered.label ? ` · ${hovered.label}` : ''}
              {hovered.probability !== undefined
                ? ` (${(hovered.probability * 100).toFixed(1)}% prob)`
                : ''}
            </span>
            <span className="font-mono tabular-nums">
              PC1: <strong>{hovered.x.toFixed(2)}</strong> · PC2:{' '}
              <strong>{hovered.y.toFixed(2)}</strong>
            </span>
          </div>
        ) : (
          <span>Hover any projected sample to inspect its coordinates and cluster assignment.</span>
        )}
      </div>
    </div>
  );
};

interface PCAVarianceProps {
  components: PCAComponentInfo[];
}

export const PCAVarianceChart: React.FC<PCAVarianceProps> = ({ components }) => {
  if (components.length === 0) {
    return (
      <div className="h-60 flex items-center justify-center text-sm text-slate-500">
        No principal components available.
      </div>
    );
  }

  return (
    <div>
      <div className="mb-3">
        <h3 className="text-sm font-semibold text-slate-900">
          Explained Variance by Component
        </h3>
        <p className="text-xs text-slate-500">
          Individual & cumulative variance retained across top principal components
        </p>
      </div>

      <div className="space-y-3 mt-4">
        {components.map((comp, idx) => {
          const pct = comp.explainedVarianceRatio * 100;
          const cumPct = comp.cumulativeVarianceRatio * 100;
          return (
            <div key={comp.component} className="space-y-1.5">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-slate-800 font-mono">
                  {comp.component}
                </span>
                <span className="font-mono tabular-nums text-slate-600">
                  <strong className="text-slate-900">{pct.toFixed(1)}%</strong> · Cum{' '}
                  {cumPct.toFixed(1)}%
                </span>
              </div>
              <div className="h-2.5 w-full rounded-full bg-slate-100 overflow-hidden">
                <div
                  className={`h-full rounded-full ${
                    idx === 0 ? 'bg-indigo-600' : 'bg-indigo-400'
                  }`}
                  style={{ width: `${Math.max(3, pct)}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

interface ClusterDiagnosticsProps {
  kDiagnostics?: KDiagnosticPoint[];
  mergeHistory?: HierarchicalMergeStep[];
  selectedK?: number;
  modelName: string;
}

export const ClusterDiagnosticsChart: React.FC<ClusterDiagnosticsProps> = ({
  kDiagnostics,
  mergeHistory,
  selectedK,
  modelName,
}) => {
  if (mergeHistory && mergeHistory.length > 0) {
    const maxDist = Math.max(1e-6, ...mergeHistory.map((m) => m.distance));
    return (
      <div>
        <div className="mb-3">
          <h3 className="text-sm font-semibold text-slate-900">
            Agglomerative Merge Distances (Ward Linkage)
          </h3>
          <p className="text-xs text-slate-500">
            Final {mergeHistory.length} hierarchical merges for {modelName}
          </p>
        </div>

        <div className="space-y-2.5 mt-4">
          {mergeHistory.slice(-8).map((step) => {
            const widthPct = Math.max(4, (step.distance / maxDist) * 100);
            return (
              <div key={step.step} className="space-y-1">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-mono text-slate-700">
                    Merge #{step.step} (size {step.size})
                  </span>
                  <span className="font-mono tabular-nums font-semibold text-slate-900">
                    dist = {step.distance.toFixed(2)}
                  </span>
                </div>
                <div className="h-2 w-full rounded-full bg-slate-100 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-indigo-500"
                    style={{ width: `${widthPct}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  }

  const diags = kDiagnostics ?? [];
  if (diags.length === 0) {
    return (
      <div className="h-60 flex items-center justify-center text-sm text-slate-500">
        No cluster diagnostics available.
      </div>
    );
  }

  return (
    <div>
      <div className="mb-3">
        <h3 className="text-sm font-semibold text-slate-900">
          Automatic K-Selection Diagnostics
        </h3>
        <p className="text-xs text-slate-500">
          Silhouette score across candidate cluster counts (higher = tighter separation)
        </p>
      </div>

      <div className="space-y-3 mt-4">
        {diags.map((d) => {
          const isChosen = d.k === selectedK;
          const pct = Math.max(5, Math.min(100, ((d.silhouette + 0.1) / 1.1) * 100));
          return (
            <div
              key={d.k}
              className={`rounded-lg border px-3 py-2.5 ${
                isChosen
                  ? 'border-emerald-300 bg-emerald-50/50'
                  : 'border-slate-200 bg-slate-50/40'
              }`}
            >
              <div className="flex items-center justify-between text-xs mb-1.5">
                <span className="font-semibold text-slate-900 font-mono">
                  K = {d.k} {isChosen ? '· Selected' : ''}
                </span>
                <span className="font-mono tabular-nums text-slate-700">
                  Silhouette: <strong>{d.silhouette.toFixed(3)}</strong>
                  {d.inertia !== undefined ? ` · Inertia: ${d.inertia.toFixed(0)}` : ''}
                  {d.bic !== undefined ? ` · BIC: ${d.bic.toFixed(0)}` : ''}
                </span>
              </div>
              <div className="h-2 w-full rounded-full bg-slate-200/70 overflow-hidden">
                <div
                  className={`h-full rounded-full ${
                    isChosen ? 'bg-emerald-600' : 'bg-indigo-400'
                  }`}
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

interface AprioriRulesChartProps {
  rules: AssociationRule[];
}

export const AprioriRulesChart: React.FC<AprioriRulesChartProps> = ({ rules }) => {
  const topRules = rules.slice(0, 7);
  const maxLift = Math.max(1.5, ...topRules.map((r) => r.lift));

  return (
    <div>
      <div className="mb-3">
        <h3 className="text-sm font-semibold text-slate-900">
          Top Association Rules by Lift
        </h3>
        <p className="text-xs text-slate-500">
          Lift &gt; 1.0 indicates positive co-occurrence above random chance
        </p>
      </div>

      {topRules.length === 0 ? (
        <div className="h-56 flex items-center justify-center text-sm text-slate-500">
          No association rules met minimum support/lift thresholds.
        </div>
      ) : (
        <div className="space-y-3 mt-4">
          {topRules.map((r, idx) => {
            const label = `${r.antecedents.join(' + ')} → ${r.consequents.join(' + ')}`;
            const widthPct = Math.max(6, (r.lift / maxLift) * 100);
            return (
              <div key={idx} className="space-y-1">
                <div className="flex items-center justify-between gap-2 text-xs">
                  <span className="font-medium text-slate-800 truncate" title={label}>
                    {label}
                  </span>
                  <span className="font-mono tabular-nums text-slate-700 shrink-0">
                    Lift <strong className="text-indigo-600">{r.lift.toFixed(2)}×</strong> ·{' '}
                    {(r.confidence * 100).toFixed(0)}% conf
                  </span>
                </div>
                <div className="h-2 w-full rounded-full bg-slate-100 overflow-hidden">
                  <div
                    className={`h-full rounded-full ${
                      idx === 0 ? 'bg-emerald-600' : 'bg-indigo-500'
                    }`}
                    style={{ width: `${widthPct}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

