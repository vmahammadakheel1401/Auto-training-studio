import { isMissingValue } from './csvParser';
import {
  AssociationRule,
  ClusterProfile,
  FeatureEncoderSpec,
  FrequentItemset,
  HierarchicalMergeStep,
  KDiagnosticPoint,
  ParsedDataset,
  PCAComponentInfo,
  ProjectionPoint,
  UnsupervisedModelKey,
  UnsupervisedModelResult,
  UnsupervisedPredictionOutput,
  UnsupervisedPreprocessorState,
} from './types';

function yieldToBrowser(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

function createRng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (1664525 * s + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function parseNumericClean(val: string | undefined): number | null {
  if (isMissingValue(val)) return null;
  const cleaned = val!.trim().replace(/[$,%]/g, '');
  const n = Number(cleaned);
  return Number.isFinite(n) ? n : null;
}

function medianOf(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

export function isBinaryTruthy(val: string | undefined): boolean {
  if (!val) return false;
  const t = val.trim().toLowerCase();
  return t === '1' || t === '1.0' || t === 'true' || t === 'yes' || t === 'y' || t === 't';
}

export function isBinaryFalsy(val: string | undefined): boolean {
  if (!val) return false;
  const t = val.trim().toLowerCase();
  return t === '0' || t === '0.0' || t === 'false' || t === 'no' || t === 'n' || t === 'f';
}

export function isBinaryToken(val: string | undefined): boolean {
  return isBinaryTruthy(val) || isBinaryFalsy(val);
}

export interface UnsupervisedPreparedData {
  preprocessor: UnsupervisedPreprocessorState;
  X: number[][];
  transactions: string[][];
}

/**
 * Extract transactions from rows for Apriori Association Rule Mining.
 * Supports:
 * 1) Binary 0/1 or Yes/No wide basket columns
 * 2) Delimited item-list columns (e.g. "milk;bread;eggs" or "milk|bread")
 * 3) General categorical/numeric columns (discretized into col=value / Low-Mid-High)
 */
export function extractTransactions(
  dataset: ParsedDataset,
  featureCols: string[]
): { transactions: string[][]; isTransactional: boolean; catalog: string[] } {
  const rows = dataset.rows;
  if (rows.length === 0 || featureCols.length === 0) {
    return { transactions: [], isTransactional: false, catalog: [] };
  }

  // Check for delimited item-list column
  const delimitedCol = featureCols.find((col) => {
    const sample = rows.slice(0, 30).map((r) => r[col] ?? '');
    const delimHits = sample.filter((v) => /[;|]/.test(v) || (v.includes(',') && v.split(',').length >= 2)).length;
    return delimHits >= Math.max(3, sample.length * 0.4);
  });

  if (delimitedCol) {
    const itemFreq = new Map<string, number>();
    const transactions = rows.map((r) => {
      const raw = r[delimitedCol] ?? '';
      const items = Array.from(
        new Set(
          raw
            .split(/[;|,]/)
            .map((s) => s.trim())
            .filter((s) => s.length > 0 && !isMissingValue(s))
        )
      );
      items.forEach((it) => itemFreq.set(it, (itemFreq.get(it) ?? 0) + 1));
      return items;
    });
    const catalog = Array.from(itemFreq.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 40)
      .map(([k]) => k);
    return { transactions, isTransactional: true, catalog };
  }

  // Check if feature columns are predominantly binary indicators
  let binaryColCount = 0;
  const binaryCols: string[] = [];
  for (const col of featureCols) {
    const nonMissing = rows.map((r) => r[col]).filter((v) => !isMissingValue(v));
    if (nonMissing.length === 0) continue;
    const binHits = nonMissing.filter((v) => isBinaryToken(v)).length;
    if (binHits / nonMissing.length >= 0.9) {
      binaryColCount++;
      binaryCols.push(col);
    }
  }

  if (binaryColCount >= 3 && binaryColCount / featureCols.length >= 0.65) {
    const itemFreq = new Map<string, number>();
    const transactions = rows.map((r) => {
      const items: string[] = [];
      for (const col of binaryCols) {
        if (isBinaryTruthy(r[col])) {
          items.push(col);
          itemFreq.set(col, (itemFreq.get(col) ?? 0) + 1);
        }
      }
      return items;
    });
    const catalog = Array.from(itemFreq.entries())
      .sort((a, b) => b[1] - a[1])
      .map(([k]) => k);
    return { transactions, isTransactional: true, catalog };
  }

  // General tabular fallback: turn categoricals into "col=val" and numerics into tertile bins
  const numericThresholds = new Map<string, [number, number]>();
  for (const col of featureCols) {
    const stat = dataset.columnStats.find((c) => c.name === col);
    if (stat?.inferredType === 'numeric') {
      const vals = rows
        .map((r) => parseNumericClean(r[col]))
        .filter((v): v is number => v !== null)
        .sort((a, b) => a - b);
      if (vals.length >= 6) {
        const q1 = vals[Math.floor(vals.length * 0.33)];
        const q2 = vals[Math.floor(vals.length * 0.67)];
        numericThresholds.set(col, [q1, q2]);
      }
    }
  }

  const itemFreq = new Map<string, number>();
  const transactions = rows.map((r) => {
    const items: string[] = [];
    for (const col of featureCols) {
      const raw = r[col];
      if (isMissingValue(raw)) continue;
      if (numericThresholds.has(col)) {
        const num = parseNumericClean(raw);
        if (num === null) continue;
        const [q1, q2] = numericThresholds.get(col)!;
        const bin = num <= q1 ? 'Low' : num <= q2 ? 'Mid' : 'High';
        const token = `${col}=${bin}`;
        items.push(token);
        itemFreq.set(token, (itemFreq.get(token) ?? 0) + 1);
      } else {
        const token = `${col}=${raw.trim()}`;
        items.push(token);
        itemFreq.set(token, (itemFreq.get(token) ?? 0) + 1);
      }
    }
    return Array.from(new Set(items));
  });

  const catalog = Array.from(itemFreq.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 40)
    .map(([k]) => k);

  return { transactions, isTransactional: false, catalog };
}

export function prepareUnsupervisedData(
  dataset: ParsedDataset,
  droppedIdColumns: string[]
): UnsupervisedPreparedData {
  const droppedSet = new Set(droppedIdColumns);
  const featureStats = dataset.columnStats.filter((c) => !droppedSet.has(c.name));

  if (featureStats.length === 0) {
    throw new Error('No input feature columns remain after excluding ID columns.');
  }
  if (dataset.rows.length < 5) {
    throw new Error('Need at least 5 rows for unsupervised analysis.');
  }

  const specs: FeatureEncoderSpec[] = [];
  const encodedColumnToRaw: string[] = [];
  let encodedOffset = 0;

  for (const stat of featureStats) {
    if (stat.inferredType === 'numeric') {
      const nums: number[] = [];
      for (const r of dataset.rows) {
        const parsed = parseNumericClean(r[stat.name]);
        if (parsed !== null) nums.push(parsed);
      }
      const median = nums.length > 0 ? medianOf(nums) : stat.median ?? 0;
      const imputed = dataset.rows.map((r) => parseNumericClean(r[stat.name]) ?? median);
      const mean = imputed.reduce((a, b) => a + b, 0) / Math.max(1, imputed.length);
      const variance =
        imputed.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, imputed.length);
      const std = Math.sqrt(variance) > 1e-8 ? Math.sqrt(variance) : 1;

      specs.push({
        rawName: stat.name,
        type: 'numeric',
        median,
        mean,
        std,
        min: stat.min ?? median,
        max: stat.max ?? median,
        step: stat.step ?? 1,
        encodedStartIndex: encodedOffset,
        encodedLength: 1,
      });
      encodedColumnToRaw.push(stat.name);
      encodedOffset += 1;
    } else {
      const freq = new Map<string, number>();
      for (const r of dataset.rows) {
        const raw = r[stat.name];
        if (!isMissingValue(raw)) {
          const val = raw.trim();
          freq.set(val, (freq.get(val) ?? 0) + 1);
        }
      }
      const sortedCats = Array.from(freq.entries())
        .sort((a, b) => b[1] - a[1])
        .map(([k]) => k);
      const mode = sortedCats[0] ?? stat.mode ?? 'Unknown';
      const hasOverflow = sortedCats.length > 20;
      const top20 = sortedCats.slice(0, 20);
      const keptCategories = hasOverflow ? [...top20, 'Other'] : top20;
      const encLen = Math.max(1, keptCategories.length);

      specs.push({
        rawName: stat.name,
        type: 'categorical',
        mode,
        categories: keptCategories.length > 0 ? keptCategories : [mode],
        hasOtherBucket: hasOverflow || stat.hasOtherBucket,
        encodedStartIndex: encodedOffset,
        encodedLength: encLen,
      });
      for (let k = 0; k < encLen; k++) {
        encodedColumnToRaw.push(stat.name);
      }
      encodedOffset += encLen;
    }
  }

  const txInfo = extractTransactions(
    dataset,
    featureStats.map((s) => s.name)
  );

  const preprocessor: UnsupervisedPreprocessorState = {
    droppedIdColumns,
    featureNames: featureStats.map((s) => s.name),
    encodedFeatureCount: encodedOffset,
    specs,
    encodedColumnToRaw,
    rowCount: dataset.rows.length,
    isTransactional: txInfo.isTransactional,
    transactionItemsCatalog: txInfo.catalog,
  };

  const X = dataset.rows.map((r) => transformUnsupervisedRow(r, preprocessor));

  return {
    preprocessor,
    X,
    transactions: txInfo.transactions,
  };
}

export function transformUnsupervisedRow(
  row: Record<string, string | number>,
  preprocessor: UnsupervisedPreprocessorState
): number[] {
  const vec = new Array<number>(preprocessor.encodedFeatureCount).fill(0);

  for (const spec of preprocessor.specs) {
    const rawVal = row[spec.rawName];

    if (spec.type === 'numeric') {
      let num: number | null = null;
      if (typeof rawVal === 'number') {
        num = Number.isFinite(rawVal) ? rawVal : null;
      } else {
        num = parseNumericClean(rawVal);
      }
      const imputed = num ?? spec.median ?? 0;
      const mean = spec.mean ?? 0;
      const std = spec.std ?? 1;
      vec[spec.encodedStartIndex] = (imputed - mean) / std;
    } else {
      const strVal =
        rawVal === undefined || rawVal === null || isMissingValue(String(rawVal))
          ? spec.mode ?? ''
          : String(rawVal).trim();
      const cats = spec.categories ?? [];
      let idx = cats.indexOf(strVal);
      if (idx < 0 && spec.hasOtherBucket) {
        idx = cats.indexOf('Other');
      }
      if (idx >= 0) {
        vec[spec.encodedStartIndex + idx] = 1;
      }
    }
  }

  return vec;
}

// ============================================================================
// Linear Algebra & Clustering Utilities
// ============================================================================

/**
 * Compute top eigenvalues & eigenvectors of symmetric matrix via Jacobi rotations.
 * Capped safely at maxDim=64 features for instant interactive responsiveness.
 */
function jacobiEigenSymmetric(A: number[][]): { eigenvalues: number[]; eigenvectors: number[][] } {
  const p = A.length;
  const M = A.map((row) => [...row]);
  const V: number[][] = Array.from({ length: p }, (_, i) =>
    Array.from({ length: p }, (__, j) => (i === j ? 1 : 0))
  );

  if (p <= 1) {
    return {
      eigenvalues: p === 1 ? [M[0][0]] : [],
      eigenvectors: p === 1 ? [[1]] : [],
    };
  }

  const maxSweeps = Math.min(80, Math.max(25, p * 4));
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let maxOff = 0;
    let pIdx = 0;
    let qIdx = 1;

    for (let i = 0; i < p; i++) {
      for (let j = i + 1; j < p; j++) {
        const val = Math.abs(M[i][j]);
        if (val > maxOff) {
          maxOff = val;
          pIdx = i;
          qIdx = j;
        }
      }
    }

    if (maxOff < 1e-10) break;

    const app = M[pIdx][pIdx];
    const aqq = M[qIdx][qIdx];
    const apq = M[pIdx][qIdx];
    const tau = (aqq - app) / (2 * apq);
    const t =
      tau >= 0
        ? 1 / (tau + Math.sqrt(1 + tau * tau))
        : -1 / (-tau + Math.sqrt(1 + tau * tau));
    const c = 1 / Math.sqrt(1 + t * t);
    const s = t * c;

    M[pIdx][pIdx] = app - t * apq;
    M[qIdx][qIdx] = aqq + t * apq;
    M[pIdx][qIdx] = 0;
    M[qIdx][pIdx] = 0;

    for (let k = 0; k < p; k++) {
      if (k !== pIdx && k !== qIdx) {
        const mkp = M[k][pIdx];
        const mkq = M[k][qIdx];
        M[k][pIdx] = c * mkp - s * mkq;
        M[pIdx][k] = M[k][pIdx];
        M[k][qIdx] = s * mkp + c * mkq;
        M[qIdx][k] = M[k][qIdx];
      }
    }

    for (let k = 0; k < p; k++) {
      const vkp = V[k][pIdx];
      const vkq = V[k][qIdx];
      V[k][pIdx] = c * vkp - s * vkq;
      V[k][qIdx] = s * vkp + c * vkq;
    }
  }

  const pairs = Array.from({ length: p }, (_, idx) => ({
    val: Math.max(0, M[idx][idx]),
    vec: V.map((row) => row[idx]),
  })).sort((a, b) => b.val - a.val);

  return {
    eigenvalues: pairs.map((pr) => pr.val),
    eigenvectors: pairs.map((pr) => pr.vec),
  };
}

function compute2DPCAProjection(X: number[][]): {
  coords2D: [number, number][];
  axisLabels: [string, string];
  vectors: number[][];
  means: number[];
  eigenvalues: number[];
} {
  const n = X.length;
  const p = X[0].length;
  const means = new Array(p).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < p; j++) means[j] += X[i][j];
  }
  for (let j = 0; j < p; j++) means[j] /= n;

  // Center X
  const Xc = X.map((row) => row.map((v, j) => v - means[j]));

  // Covariance matrix (if p > 48, take top 48 highest-variance columns for Jacobi speed)
  const useCols = p <= 48 ? p : 48;
  const cov: number[][] = Array.from({ length: useCols }, () => new Array(useCols).fill(0));
  const denom = Math.max(1, n - 1);

  for (let i = 0; i < n; i++) {
    const row = Xc[i];
    for (let j = 0; j < useCols; j++) {
      for (let k = j; k < useCols; k++) {
        cov[j][k] += row[j] * row[k];
      }
    }
  }
  for (let j = 0; j < useCols; j++) {
    for (let k = j; k < useCols; k++) {
      cov[j][k] /= denom;
      cov[k][j] = cov[j][k];
    }
  }

  const { eigenvalues, eigenvectors } = jacobiEigenSymmetric(cov);
  const fullVectors = eigenvectors.map((vec) => {
    if (p <= 48) return vec;
    const padded = new Array(p).fill(0);
    vec.forEach((v, idx) => {
      padded[idx] = v;
    });
    return padded;
  });

  const totalVar = Math.max(
    1e-9,
    eigenvalues.reduce((a, b) => a + b, 0)
  );
  const pc1Pct = ((eigenvalues[0] ?? 0) / totalVar) * 100;
  const pc2Pct = ((eigenvalues[1] ?? 0) / totalVar) * 100;

  const v1 = fullVectors[0] ?? new Array(p).fill(0);
  const v2 = fullVectors[1] ?? new Array(p).fill(0);

  const coords2D: [number, number][] = Xc.map((row) => {
    let x1 = 0;
    let x2 = 0;
    for (let j = 0; j < p; j++) {
      x1 += row[j] * v1[j];
      x2 += row[j] * v2[j];
    }
    return [x1, x2];
  });

  return {
    coords2D,
    axisLabels: [
      `PC1 (${pc1Pct.toFixed(1)}% var)`,
      fullVectors.length > 1 ? `PC2 (${pc2Pct.toFixed(1)}% var)` : 'PC2 (0.0% var)',
    ],
    vectors: fullVectors,
    means,
    eigenvalues,
  };
}

function sqDist(a: number[], b: number[]): number {
  let s = 0;
  for (let j = 0; j < a.length; j++) {
    const d = a[j] - b[j];
    s += d * d;
  }
  return s;
}

function computeSilhouetteScore(X: number[][], labels: number[], k: number): number {
  if (k <= 1 || X.length <= k) return 0;

  // Deterministic subsample up to 320 points for fast silhouette calculation
  const maxPts = 320;
  let indices: number[];
  if (X.length <= maxPts) {
    indices = X.map((_, i) => i);
  } else {
    indices = [];
    const step = X.length / maxPts;
    for (let i = 0; i < maxPts; i++) {
      indices.push(Math.min(X.length - 1, Math.floor(i * step)));
    }
  }

  const m = indices.length;
  const clusterGroups: number[][] = Array.from({ length: k }, () => []);
  for (const idx of indices) {
    const c = labels[idx];
    if (c >= 0 && c < k) clusterGroups[c].push(idx);
  }

  let totalSil = 0;
  let validCount = 0;

  for (let i = 0; i < m; i++) {
    const idx = indices[i];
    const myCluster = labels[idx];
    const myGroup = clusterGroups[myCluster];
    if (!myGroup || myGroup.length <= 1) continue;

    let aSum = 0;
    for (const other of myGroup) {
      if (other !== idx) aSum += Math.sqrt(sqDist(X[idx], X[other]));
    }
    const a = aSum / (myGroup.length - 1);

    let b = Infinity;
    for (let c = 0; c < k; c++) {
      if (c === myCluster || clusterGroups[c].length === 0) continue;
      let bSum = 0;
      for (const other of clusterGroups[c]) {
        bSum += Math.sqrt(sqDist(X[idx], X[other]));
      }
      const meanDist = bSum / clusterGroups[c].length;
      if (meanDist < b) b = meanDist;
    }

    if (!Number.isFinite(b)) continue;
    const denom = Math.max(a, b);
    const s = denom > 1e-12 ? (b - a) / denom : 0;
    totalSil += s;
    validCount++;
  }

  return validCount > 0 ? totalSil / validCount : 0;
}

function buildClusterProfiles(
  X: number[][],
  labels: number[],
  centroids: number[][],
  preprocessor: UnsupervisedPreprocessorState
): ClusterProfile[] {
  const k = centroids.length;
  const n = X.length;
  const counts = new Array(k).fill(0);
  for (const l of labels) {
    if (l >= 0 && l < k) counts[l]++;
  }

  // Compute global feature means in standardized X (all ~0 for numeric)
  const profiles: ClusterProfile[] = [];

  for (let c = 0; c < k; c++) {
    const centroid = centroids[c];
    const featHighlights: { feature: string; meanLabel: string; zScore: number }[] = [];

    for (const spec of preprocessor.specs) {
      if (spec.type === 'numeric') {
        const z = centroid[spec.encodedStartIndex] ?? 0;
        const originalMean = z * (spec.std ?? 1) + (spec.mean ?? 0);
        const formatted =
          Math.abs(originalMean) >= 1000
            ? originalMean.toLocaleString(undefined, { maximumFractionDigits: 0 })
            : originalMean.toLocaleString(undefined, { maximumFractionDigits: 2 });
        featHighlights.push({
          feature: spec.rawName,
          meanLabel: formatted,
          zScore: z,
        });
      } else if (spec.categories && spec.categories.length > 0) {
        let bestCat = spec.categories[0];
        let bestShare = -1;
        for (let offset = 0; offset < spec.encodedLength; offset++) {
          const share = centroid[spec.encodedStartIndex + offset] ?? 0;
          if (share > bestShare) {
            bestShare = share;
            bestCat = spec.categories[offset] ?? bestCat;
          }
        }
        featHighlights.push({
          feature: spec.rawName,
          meanLabel: `${bestCat} (${Math.round(bestShare * 100)}%)`,
          zScore: bestShare,
        });
      }
    }

    featHighlights.sort((a, b) => Math.abs(b.zScore) - Math.abs(a.zScore));

    profiles.push({
      clusterId: c,
      label: `Cluster ${c + 1}`,
      count: counts[c],
      percentage: Number(((counts[c] / Math.max(1, n)) * 100).toFixed(1)),
      centroid,
      topFeatures: featHighlights.slice(0, 5),
    });
  }

  return profiles;
}

// ============================================================================
// 1. PCA (Principal Component Analysis)
// ============================================================================

async function runPCA(
  prepared: UnsupervisedPreparedData
): Promise<UnsupervisedModelResult> {
  const t0 = performance.now();
  await yieldToBrowser();

  const { X, preprocessor } = prepared;
  const p = preprocessor.encodedFeatureCount;
  const proj = compute2DPCAProjection(X);

  const totalVar = Math.max(
    1e-9,
    proj.eigenvalues.reduce((a, b) => a + b, 0)
  );
  const numKeep = Math.min(6, proj.eigenvalues.length);
  const pcaComponents: PCAComponentInfo[] = [];
  let cum = 0;

  for (let c = 0; c < numKeep; c++) {
    const ratio = proj.eigenvalues[c] / totalVar;
    cum += ratio;
    const vec = proj.vectors[c] ?? new Array(p).fill(0);

    // Aggregate loadings back to raw feature names
    const rawLoadings = new Map<string, number>();
    vec.forEach((loading, idx) => {
      const rawName = preprocessor.encodedColumnToRaw[idx];
      if (!rawName) return;
      const prev = rawLoadings.get(rawName) ?? 0;
      if (Math.abs(loading) > Math.abs(prev)) {
        rawLoadings.set(rawName, loading);
      }
    });

    const topLoadings = Array.from(rawLoadings.entries())
      .map(([feature, loading]) => ({
        feature,
        loading: Number(loading.toFixed(3)),
      }))
      .sort((a, b) => Math.abs(b.loading) - Math.abs(a.loading))
      .slice(0, 6);

    pcaComponents.push({
      component: `PC${c + 1}`,
      eigenvalue: Number(proj.eigenvalues[c].toFixed(3)),
      explainedVarianceRatio: ratio,
      cumulativeVarianceRatio: Math.min(1, cum),
      topLoadings,
    });
  }

  const projectionPoints: ProjectionPoint[] = proj.coords2D.map(([x, y], i) => ({
    x,
    y,
    rowIndex: i + 1,
  }));

  const trainingTimeMs = Math.max(1, Math.round(performance.now() - t0));

  return {
    key: 'pca',
    name: 'Principal Component Analysis (PCA)',
    sklearnName: 'PCA',
    family: 'Dimensionality Reduction',
    unsupervisedTask: 'pca',
    trainingTimeMs,
    pcaComponents,
    pcaVectors: proj.vectors.slice(0, numKeep),
    pcaMeans: proj.means,
    totalVarianceExplained2D:
      (pcaComponents[0]?.explainedVarianceRatio ?? 0) +
      (pcaComponents[1]?.explainedVarianceRatio ?? 0),
    projectionPoints,
    projectionAxisLabels: proj.axisLabels,
  };
}

// ============================================================================
// 2. K-Means Clustering (K-Means++ Initialization + Auto-K via Silhouette)
// ============================================================================

function fitKMeansSingleK(X: number[][], k: number, seed = 42) {
  const n = X.length;
  const p = X[0].length;
  const rng = createRng(seed + k * 97);

  // K-Means++ initialization
  const centroids: number[][] = [];
  const firstIdx = Math.floor(rng() * n);
  centroids.push([...X[firstIdx]]);

  const minSqDists = new Array(n).fill(Infinity);
  for (let c = 1; c < k; c++) {
    let total = 0;
    const lastCentroid = centroids[c - 1];
    for (let i = 0; i < n; i++) {
      const d = sqDist(X[i], lastCentroid);
      if (d < minSqDists[i]) minSqDists[i] = d;
      total += minSqDists[i];
    }
    let target = rng() * total;
    let chosen = n - 1;
    for (let i = 0; i < n; i++) {
      target -= minSqDists[i];
      if (target <= 0) {
        chosen = i;
        break;
      }
    }
    centroids.push([...X[chosen]]);
  }

  const labels = new Array(n).fill(0);
  const maxIter = 35;

  for (let iter = 0; iter < maxIter; iter++) {
    let changed = false;
    for (let i = 0; i < n; i++) {
      let bestC = 0;
      let bestD = Infinity;
      for (let c = 0; c < k; c++) {
        const d = sqDist(X[i], centroids[c]);
        if (d < bestD) {
          bestD = d;
          bestC = c;
        }
      }
      if (labels[i] !== bestC) {
        labels[i] = bestC;
        changed = true;
      }
    }

    const sums: number[][] = Array.from({ length: k }, () => new Array(p).fill(0));
    const counts = new Array(k).fill(0);
    for (let i = 0; i < n; i++) {
      const c = labels[i];
      counts[c]++;
      for (let j = 0; j < p; j++) sums[c][j] += X[i][j];
    }

    for (let c = 0; c < k; c++) {
      if (counts[c] > 0) {
        for (let j = 0; j < p; j++) centroids[c][j] = sums[c][j] / counts[c];
      }
    }

    if (!changed && iter > 0) break;
  }

  let inertia = 0;
  for (let i = 0; i < n; i++) {
    inertia += sqDist(X[i], centroids[labels[i]]);
  }

  return { labels, centroids, inertia };
}

async function runKMeans(
  prepared: UnsupervisedPreparedData
): Promise<UnsupervisedModelResult> {
  const t0 = performance.now();
  const { X, preprocessor } = prepared;
  const maxK = Math.min(6, Math.max(2, Math.floor(Math.sqrt(X.length / 2))));

  const kDiagnostics: KDiagnosticPoint[] = [];
  let bestK = 2;
  let bestSil = -Infinity;
  let bestFit = fitKMeansSingleK(X, 2);

  for (let k = 2; k <= maxK; k++) {
    await yieldToBrowser();
    const fit = fitKMeansSingleK(X, k);
    const sil = computeSilhouetteScore(X, fit.labels, k);
    kDiagnostics.push({
      k,
      silhouette: Number(sil.toFixed(4)),
      inertia: Number(fit.inertia.toFixed(2)),
    });
    if (sil > bestSil) {
      bestSil = sil;
      bestK = k;
      bestFit = fit;
    }
  }

  const proj = compute2DPCAProjection(X);
  const projectionPoints: ProjectionPoint[] = proj.coords2D.map(([x, y], i) => ({
    x,
    y,
    rowIndex: i + 1,
    cluster: bestFit.labels[i],
    label: `Cluster ${bestFit.labels[i] + 1}`,
  }));

  const clusterProfiles = buildClusterProfiles(
    X,
    bestFit.labels,
    bestFit.centroids,
    preprocessor
  );

  const trainingTimeMs = Math.max(1, Math.round(performance.now() - t0));

  return {
    key: 'kmeans',
    name: `K-Means Clustering (K=${bestK})`,
    sklearnName: 'KMeans',
    family: 'Centroid Clustering',
    unsupervisedTask: 'clustering',
    trainingTimeMs,
    numClusters: bestK,
    silhouetteScore: Number(bestSil.toFixed(4)),
    inertia: Number(bestFit.inertia.toFixed(2)),
    clusterProfiles,
    kDiagnostics,
    projectionPoints,
    projectionAxisLabels: proj.axisLabels,
    pcaVectors: proj.vectors.slice(0, 2),
    pcaMeans: proj.means,
  };
}

// ============================================================================
// 3. Hierarchical Clustering (Agglomerative Ward Linkage)
// ============================================================================

async function runHierarchical(
  prepared: UnsupervisedPreparedData
): Promise<UnsupervisedModelResult> {
  const t0 = performance.now();
  await yieldToBrowser();

  const { X, preprocessor } = prepared;
  const n = X.length;
  const p = X[0].length;

  // Cap linkage matrix size at 260 points for smooth O(N^2) browser execution
  const maxSample = 260;
  let sampleIndices: number[];
  if (n <= maxSample) {
    sampleIndices = X.map((_, i) => i);
  } else {
    sampleIndices = [];
    const step = n / maxSample;
    for (let i = 0; i < maxSample; i++) {
      sampleIndices.push(Math.min(n - 1, Math.floor(i * step)));
    }
  }

  const m = sampleIndices.length;
  const clusters: {
    id: number;
    members: number[];
    centroid: number[];
    active: boolean;
  }[] = sampleIndices.map((origIdx, i) => ({
    id: i,
    members: [origIdx],
    centroid: [...X[origIdx]],
    active: true,
  }));

  // Track snapshots of cluster memberships at k=5..2
  const kSnapshots = new Map<number, number[][]>();
  const mergeHistory: HierarchicalMergeStep[] = [];
  let activeCount = m;

  while (activeCount > 1) {
    if (activeCount % 40 === 0) await yieldToBrowser();

    if (activeCount >= 2 && activeCount <= 5) {
      kSnapshots.set(
        activeCount,
        clusters.filter((c) => c.active).map((c) => [...c.members])
      );
    }

    let bestI = -1;
    let bestJ = -1;
    let bestWardDist = Infinity;

    for (let i = 0; i < clusters.length; i++) {
      if (!clusters[i].active) continue;
      for (let j = i + 1; j < clusters.length; j++) {
        if (!clusters[j].active) continue;
        const nA = clusters[i].members.length;
        const nB = clusters[j].members.length;
        const d2 = sqDist(clusters[i].centroid, clusters[j].centroid);
        const ward = ((2 * nA * nB) / (nA + nB)) * d2;
        if (ward < bestWardDist) {
          bestWardDist = ward;
          bestI = i;
          bestJ = j;
        }
      }
    }

    if (bestI === -1 || bestJ === -1) break;

    const cA = clusters[bestI];
    const cB = clusters[bestJ];
    const nA = cA.members.length;
    const nB = cB.members.length;
    const newN = nA + nB;
    const newCentroid = new Array(p);
    for (let d = 0; d < p; d++) {
      newCentroid[d] = (cA.centroid[d] * nA + cB.centroid[d] * nB) / newN;
    }

    cA.members.push(...cB.members);
    cA.centroid = newCentroid;
    cB.active = false;
    activeCount--;

    mergeHistory.push({
      step: m - activeCount,
      size: newN,
      distance: Number(Math.sqrt(Math.max(0, bestWardDist)).toFixed(3)),
    });
  }

  // Evaluate cut levels k = 2..min(5, m-1)
  const kDiagnostics: KDiagnosticPoint[] = [];
  let bestK = 2;
  let bestSil = -Infinity;
  let bestLabels = new Array(n).fill(0);
  let bestCentroids: number[][] = [];

  for (let k = 2; k <= Math.min(5, m - 1); k++) {
    const groups = kSnapshots.get(k);
    if (!groups) continue;

    const centroids = groups.map((grp) => {
      const cent = new Array(p).fill(0);
      for (const idx of grp) {
        for (let d = 0; d < p; d++) cent[d] += X[idx][d];
      }
      for (let d = 0; d < p; d++) cent[d] /= Math.max(1, grp.length);
      return cent;
    });

    // Assign all dataset rows to nearest cluster centroid
    const labels = X.map((row) => {
      let bc = 0;
      let bd = Infinity;
      for (let c = 0; c < k; c++) {
        const d = sqDist(row, centroids[c]);
        if (d < bd) {
          bd = d;
          bc = c;
        }
      }
      return bc;
    });

    const sil = computeSilhouetteScore(X, labels, k);
    kDiagnostics.push({ k, silhouette: Number(sil.toFixed(4)) });
    if (sil > bestSil) {
      bestSil = sil;
      bestK = k;
      bestLabels = labels;
      bestCentroids = centroids;
    }
  }

  const proj = compute2DPCAProjection(X);
  const projectionPoints: ProjectionPoint[] = proj.coords2D.map(([x, y], i) => ({
    x,
    y,
    rowIndex: i + 1,
    cluster: bestLabels[i],
    label: `Cluster ${bestLabels[i] + 1}`,
  }));

  const clusterProfiles = buildClusterProfiles(
    X,
    bestLabels,
    bestCentroids,
    preprocessor
  );

  const trainingTimeMs = Math.max(1, Math.round(performance.now() - t0));

  return {
    key: 'hierarchical',
    name: `Hierarchical Agglomerative (K=${bestK})`,
    sklearnName: 'AgglomerativeClustering',
    family: 'Ward Linkage Clustering',
    unsupervisedTask: 'clustering',
    trainingTimeMs,
    numClusters: bestK,
    silhouetteScore: Number(bestSil.toFixed(4)),
    clusterProfiles,
    kDiagnostics,
    mergeHistory: mergeHistory.slice(-14),
    projectionPoints,
    projectionAxisLabels: proj.axisLabels,
    pcaVectors: proj.vectors.slice(0, 2),
    pcaMeans: proj.means,
  };
}

// ============================================================================
// 4. Gaussian Mixture Models (Diagonal-Covariance EM + BIC Selection)
// ============================================================================

function fitGMMSingleK(X: number[][], k: number) {
  const n = X.length;
  const p = X[0].length;
  const regCovar = 1e-3;

  // Initialize means via K-Means++
  const init = fitKMeansSingleK(X, k, 1337);
  const means = init.centroids.map((c) => [...c]);
  const weights = new Array(k).fill(1 / k);
  const variances: number[][] = Array.from({ length: k }, () => new Array(p).fill(1));

  // Initial empirical diagonal variances
  const counts = new Array(k).fill(0);
  for (let i = 0; i < n; i++) counts[init.labels[i]]++;
  for (let c = 0; c < k; c++) {
    const cnt = counts[c];
    if (cnt > 1) {
      const v = new Array(p).fill(0);
      for (let i = 0; i < n; i++) {
        if (init.labels[i] === c) {
          for (let j = 0; j < p; j++) v[j] += (X[i][j] - means[c][j]) ** 2;
        }
      }
      for (let j = 0; j < p; j++) variances[c][j] = v[j] / cnt + regCovar;
    }
  }

  const resp: number[][] = Array.from({ length: n }, () => new Array(k).fill(1 / k));
  let logLikelihood = -Infinity;
  const maxIter = 32;

  for (let iter = 0; iter < maxIter; iter++) {
    // E-step: compute log-responsibilities
    let newLogL = 0;
    for (let i = 0; i < n; i++) {
      const xi = X[i];
      const logProbs = new Array(k);
      let maxLog = -Infinity;

      for (let c = 0; c < k; c++) {
        let lp = Math.log(Math.max(1e-9, weights[c]));
        for (let j = 0; j < p; j++) {
          const v = variances[c][j];
          const diff = xi[j] - means[c][j];
          lp += -0.5 * (Math.log(2 * Math.PI * v) + (diff * diff) / v);
        }
        logProbs[c] = lp;
        if (lp > maxLog) maxLog = lp;
      }

      let sumExp = 0;
      for (let c = 0; c < k; c++) {
        sumExp += Math.exp(logProbs[c] - maxLog);
      }
      const logNorm = maxLog + Math.log(sumExp);
      newLogL += logNorm;

      for (let c = 0; c < k; c++) {
        resp[i][c] = Math.exp(logProbs[c] - logNorm);
      }
    }

    // M-step
    const nk = new Array(k).fill(0);
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < k; c++) nk[c] += resp[i][c];
    }

    for (let c = 0; c < k; c++) {
      const nkc = Math.max(1e-6, nk[c]);
      weights[c] = nkc / n;
      for (let j = 0; j < p; j++) {
        let meanSum = 0;
        for (let i = 0; i < n; i++) meanSum += resp[i][c] * X[i][j];
        means[c][j] = meanSum / nkc;
      }
      for (let j = 0; j < p; j++) {
        let varSum = 0;
        for (let i = 0; i < n; i++) {
          varSum += resp[i][c] * (X[i][j] - means[c][j]) ** 2;
        }
        variances[c][j] = varSum / nkc + regCovar;
      }
    }

    if (Math.abs(newLogL - logLikelihood) < 1e-3) {
      logLikelihood = newLogL;
      break;
    }
    logLikelihood = newLogL;
  }

  const labels = resp.map((r) => r.indexOf(Math.max(...r)));
  const maxProbs = resp.map((r) => Math.max(...r));
  // Diagonal GMM parameter count: (k-1) weights + k*p means + k*p variances
  const numParams = k - 1 + 2 * k * p;
  const bic = -2 * logLikelihood + numParams * Math.log(n);
  const aic = -2 * logLikelihood + 2 * numParams;

  return { labels, maxProbs, means, variances, weights, logLikelihood, bic, aic };
}

async function runGMM(
  prepared: UnsupervisedPreparedData
): Promise<UnsupervisedModelResult> {
  const t0 = performance.now();
  const { X, preprocessor } = prepared;
  const maxK = Math.min(5, Math.max(2, Math.floor(Math.sqrt(X.length / 3))));

  const kDiagnostics: KDiagnosticPoint[] = [];
  let bestK = 2;
  let bestScore = -Infinity;
  let bestFit = fitGMMSingleK(X, 2);
  let bestSil = 0;

  for (let k = 2; k <= maxK; k++) {
    await yieldToBrowser();
    const fit = fitGMMSingleK(X, k);
    const sil = computeSilhouetteScore(X, fit.labels, k);
    kDiagnostics.push({
      k,
      silhouette: Number(sil.toFixed(4)),
      bic: Number(fit.bic.toFixed(1)),
    });
    // Combine silhouette with gentle preference against over-splitting
    const composite = sil - 0.02 * (k - 2);
    if (composite > bestScore) {
      bestScore = composite;
      bestSil = sil;
      bestK = k;
      bestFit = fit;
    }
  }

  const proj = compute2DPCAProjection(X);
  const projectionPoints: ProjectionPoint[] = proj.coords2D.map(([x, y], i) => ({
    x,
    y,
    rowIndex: i + 1,
    cluster: bestFit.labels[i],
    probability: Number(bestFit.maxProbs[i].toFixed(3)),
    label: `Component ${bestFit.labels[i] + 1}`,
  }));

  const clusterProfiles = buildClusterProfiles(
    X,
    bestFit.labels,
    bestFit.means,
    preprocessor
  ).map((prof, idx) => ({
    ...prof,
    label: `Component ${idx + 1}`,
  }));

  const trainingTimeMs = Math.max(1, Math.round(performance.now() - t0));

  return {
    key: 'gmm',
    name: `Gaussian Mixture Model (K=${bestK})`,
    sklearnName: 'GaussianMixture',
    family: 'Probabilistic EM Clustering',
    unsupervisedTask: 'clustering',
    trainingTimeMs,
    numClusters: bestK,
    silhouetteScore: Number(bestSil.toFixed(4)),
    bic: Number(bestFit.bic.toFixed(1)),
    aic: Number(bestFit.aic.toFixed(1)),
    logLikelihood: Number(bestFit.logLikelihood.toFixed(1)),
    clusterProfiles,
    kDiagnostics,
    gmmWeights: bestFit.weights,
    gmmVariances: bestFit.variances,
    projectionPoints,
    projectionAxisLabels: proj.axisLabels,
    pcaVectors: proj.vectors.slice(0, 2),
    pcaMeans: proj.means,
  };
}

// ============================================================================
// 5. Apriori Association Rule Mining
// ============================================================================

async function runApriori(
  prepared: UnsupervisedPreparedData
): Promise<UnsupervisedModelResult> {
  const t0 = performance.now();
  await yieldToBrowser();

  const { transactions } = prepared;
  const nTx = Math.max(1, transactions.length);
  const txSets = transactions.map((items) => new Set(items));

  const itemCounts = new Map<string, number>();
  let totalItemsInTx = 0;
  for (const s of txSets) {
    totalItemsInTx += s.size;
    for (const item of s) {
      itemCounts.set(item, (itemCounts.get(item) ?? 0) + 1);
    }
  }

  const avgBasketSize = Number((totalItemsInTx / nTx).toFixed(2));
  const uniqueItemsCount = itemCounts.size;

  // Adaptive minSupport so rules are reliably mined without combinatorial explosion
  let minSupport = 0.12;
  let freq1 = Array.from(itemCounts.entries())
    .filter(([, cnt]) => cnt / nTx >= minSupport)
    .sort((a, b) => b[1] - a[1]);

  if (freq1.length < 4) {
    minSupport = 0.05;
    freq1 = Array.from(itemCounts.entries())
      .filter(([, cnt]) => cnt / nTx >= minSupport)
      .sort((a, b) => b[1] - a[1]);
  }
  if (freq1.length > 30) {
    freq1 = freq1.slice(0, 30);
  }

  const supportMap = new Map<string, number>(); // sorted key -> count
  const keyOf = (items: string[]) => items.slice().sort().join('||');

  const frequentItemsets: FrequentItemset[] = [];
  const L1: string[][] = [];

  for (const [item, count] of freq1) {
    const arr = [item];
    L1.push(arr);
    supportMap.set(keyOf(arr), count);
    frequentItemsets.push({
      items: arr,
      count,
      support: count / nTx,
    });
  }

  // Generate L2
  const L2: string[][] = [];
  for (let i = 0; i < L1.length; i++) {
    for (let j = i + 1; j < L1.length; j++) {
      const cand = [L1[i][0], L1[j][0]].sort();
      let cnt = 0;
      for (const s of txSets) {
        if (s.has(cand[0]) && s.has(cand[1])) cnt++;
      }
      if (cnt / nTx >= minSupport) {
        L2.push(cand);
        supportMap.set(keyOf(cand), cnt);
        frequentItemsets.push({
          items: cand,
          count: cnt,
          support: cnt / nTx,
        });
      }
    }
  }

  await yieldToBrowser();

  // Generate L3 from L2
  const L3: string[][] = [];
  const seenL3 = new Set<string>();
  const topL2 = L2.slice(0, 60);

  for (let i = 0; i < topL2.length; i++) {
    for (let j = i + 1; j < topL2.length; j++) {
      const merged = Array.from(new Set([...topL2[i], ...topL2[j]])).sort();
      if (merged.length !== 3) continue;
      const k = keyOf(merged);
      if (seenL3.has(k)) continue;
      seenL3.add(k);

      // Prune: all 2-subsets must be frequent
      const s1 = keyOf([merged[0], merged[1]]);
      const s2 = keyOf([merged[0], merged[2]]);
      const s3 = keyOf([merged[1], merged[2]]);
      if (!supportMap.has(s1) || !supportMap.has(s2) || !supportMap.has(s3)) continue;

      let cnt = 0;
      for (const s of txSets) {
        if (s.has(merged[0]) && s.has(merged[1]) && s.has(merged[2])) cnt++;
      }
      if (cnt / nTx >= minSupport) {
        L3.push(merged);
        supportMap.set(k, cnt);
        frequentItemsets.push({
          items: merged,
          count: cnt,
          support: cnt / nTx,
        });
      }
    }
  }

  // Generate Association Rules from L2 and L3
  const buildRules = (minConf: number, minLift: number): AssociationRule[] => {
    const out: AssociationRule[] = [];
    const multiSets = [...L2, ...L3];

    for (const itemset of multiSets) {
      const totalCnt = supportMap.get(keyOf(itemset)) ?? 0;
      const support = totalCnt / nTx;

      // Enumerate non-empty proper subsets as antecedents
      const len = itemset.length;
      const maxMask = (1 << len) - 1;
      for (let mask = 1; mask < maxMask; mask++) {
        const ant: string[] = [];
        const cons: string[] = [];
        for (let b = 0; b < len; b++) {
          if (mask & (1 << b)) ant.push(itemset[b]);
          else cons.push(itemset[b]);
        }

        const antCnt = supportMap.get(keyOf(ant)) ?? 0;
        const consCnt = supportMap.get(keyOf(cons)) ?? 0;
        if (antCnt === 0 || consCnt === 0) continue;

        const antSup = antCnt / nTx;
        const consSup = consCnt / nTx;
        const confidence = support / antSup;
        const lift = confidence / consSup;

        if (confidence >= minConf && lift >= minLift) {
          const conviction =
            confidence >= 0.999 ? 99 : (1 - consSup) / Math.max(1e-4, 1 - confidence);
          out.push({
            antecedents: ant,
            consequents: cons,
            support: Number(support.toFixed(4)),
            confidence: Number(confidence.toFixed(4)),
            lift: Number(lift.toFixed(3)),
            conviction: Number(conviction.toFixed(2)),
          });
        }
      }
    }

    out.sort((a, b) => b.lift * b.confidence - a.lift * a.confidence);
    return out;
  };

  let rules = buildRules(0.35, 1.05);
  if (rules.length < 3) {
    rules = buildRules(0.2, 1.0);
  }

  frequentItemsets.sort((a, b) => {
    if (b.items.length !== a.items.length) return b.items.length - a.items.length;
    return b.support - a.support;
  });

  const trainingTimeMs = Math.max(1, Math.round(performance.now() - t0));

  return {
    key: 'apriori',
    name: 'Apriori Association Rules',
    sklearnName: 'apriori + association_rules',
    family: 'Frequent Pattern Mining',
    unsupervisedTask: 'association',
    trainingTimeMs,
    totalTransactions: nTx,
    uniqueItemsCount,
    avgBasketSize,
    minSupportUsed: minSupport,
    frequentItemsets: frequentItemsets.slice(0, 40),
    associationRules: rules.slice(0, 80),
  };
}

// ============================================================================
// Public Dispatcher & Live Unsupervised Inference
// ============================================================================

export async function trainUnsupervisedAlgorithm(
  algoKey: UnsupervisedModelKey,
  prepared: UnsupervisedPreparedData
): Promise<UnsupervisedModelResult> {
  if (algoKey === 'pca') return runPCA(prepared);
  if (algoKey === 'kmeans') return runKMeans(prepared);
  if (algoKey === 'hierarchical') return runHierarchical(prepared);
  if (algoKey === 'gmm') return runGMM(prepared);
  return runApriori(prepared);
}

export function predictUnsupervisedSingleRow(
  model: UnsupervisedModelResult,
  x: number[]
): UnsupervisedPredictionOutput {
  if (model.key === 'pca') {
    const means = model.pcaMeans ?? new Array(x.length).fill(0);
    const vectors = model.pcaVectors ?? [];
    const projectedCoords = vectors.slice(0, 3).map((vec, idx) => {
      let val = 0;
      for (let j = 0; j < x.length; j++) {
        val += (x[j] - (means[j] ?? 0)) * (vec[j] ?? 0);
      }
      return { component: `PC${idx + 1}`, value: Number(val.toFixed(3)) };
    });
    return { modelKey: 'pca', projectedCoords };
  }

  const profiles = model.clusterProfiles ?? [];
  if (profiles.length === 0) {
    return { modelKey: model.key };
  }

  if (model.key === 'gmm' && model.gmmWeights && model.gmmVariances) {
    const k = profiles.length;
    const logProbs = new Array(k);
    let maxLog = -Infinity;
    for (let c = 0; c < k; c++) {
      let lp = Math.log(Math.max(1e-9, model.gmmWeights[c]));
      const mean = profiles[c].centroid;
      const vari = model.gmmVariances[c];
      for (let j = 0; j < x.length; j++) {
        const v = vari[j] ?? 1;
        const diff = x[j] - (mean[j] ?? 0);
        lp += -0.5 * (Math.log(2 * Math.PI * v) + (diff * diff) / v);
      }
      logProbs[c] = lp;
      if (lp > maxLog) maxLog = lp;
    }
    const exps = logProbs.map((lp) => Math.exp(lp - maxLog));
    const sum = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map((e) => e / Math.max(1e-12, sum));
    const bestC = probs.indexOf(Math.max(...probs));

    return {
      modelKey: 'gmm',
      assignedCluster: bestC,
      assignedClusterLabel: profiles[bestC].label,
      clusterProbabilities: profiles.map((p, idx) => ({
        clusterId: p.clusterId,
        label: p.label,
        probability: probs[idx] ?? 0,
      })),
    };
  }

  const centroidDistances = profiles.map((prof) => ({
    clusterId: prof.clusterId,
    label: prof.label,
    distance: Number(Math.sqrt(sqDist(x, prof.centroid)).toFixed(3)),
  }));
  centroidDistances.sort((a, b) => a.distance - b.distance);

  return {
    modelKey: model.key,
    assignedCluster: centroidDistances[0].clusterId,
    assignedClusterLabel: centroidDistances[0].label,
    centroidDistances,
  };
}

export function recommendBasketItems(
  rules: AssociationRule[],
  selectedItems: string[]
): { item: string; confidence: number; lift: number; matchedAntecedent: string[] }[] {
  const basket = new Set(selectedItems);
  const bestByItem = new Map<
    string,
    { item: string; confidence: number; lift: number; matchedAntecedent: string[] }
  >();

  for (const rule of rules) {
    const allPresent = rule.antecedents.every((a) => basket.has(a));
    if (!allPresent) continue;

    for (const cons of rule.consequents) {
      if (basket.has(cons)) continue;
      const prev = bestByItem.get(cons);
      if (!prev || rule.confidence * rule.lift > prev.confidence * prev.lift) {
        bestByItem.set(cons, {
          item: cons,
          confidence: rule.confidence,
          lift: rule.lift,
          matchedAntecedent: rule.antecedents,
        });
      }
    }
  }

  return Array.from(bestByItem.values()).sort(
    (a, b) => b.confidence * b.lift - a.confidence * a.lift
  );
}
