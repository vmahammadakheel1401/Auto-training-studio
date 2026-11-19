import {
  ExpandedSupervisedModelKey,
  SerializedModelWeights,
  SerializedTreeNode,
  SupervisedTaskType,
} from './types';

function yieldToBrowser(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

function sigmoid(z: number): number {
  if (z > 30) return 1 - 1e-13;
  if (z < -30) return 1e-13;
  return 1 / (1 + Math.exp(-z));
}

function softmax(logits: number[]): number[] {
  const maxL = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - maxL));
  const sum = exps.reduce((a, b) => a + b, 0);
  return sum > 1e-15 ? exps.map((e) => e / sum) : logits.map(() => 1 / logits.length);
}

function solveLinearSystem(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col++) {
    let pivotRow = col;
    let maxVal = Math.abs(M[col][col]);
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r][col]) > maxVal) {
        maxVal = Math.abs(M[r][col]);
        pivotRow = r;
      }
    }
    if (pivotRow !== col) {
      const tmp = M[col];
      M[col] = M[pivotRow];
      M[pivotRow] = tmp;
    }

    const pivot = Math.abs(M[col][col]) < 1e-10 ? 1e-10 : M[col][col];
    for (let j = col; j <= n; j++) {
      M[col][j] /= pivot;
    }

    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = M[r][col];
      if (factor === 0) continue;
      for (let j = col; j <= n; j++) {
        M[r][j] -= factor * M[col][j];
      }
    }
  }

  return M.map((row) => row[n]);
}

// ============================================================================
// 1. Linear Regression (OLS with tiny numerical jitter for rank-deficiency)
// ============================================================================

function fitLinearRegression(X: number[][], y: number[]) {
  const n = X.length;
  const p = X[0].length;
  const dim = p + 1;
  const XtX: number[][] = Array.from({ length: dim }, () => new Array(dim).fill(0));
  const Xty: number[] = new Array(dim).fill(0);

  for (let i = 0; i < n; i++) {
    const xi = X[i];
    const yi = y[i];
    Xty[0] += yi;
    XtX[0][0] += 1;
    for (let j = 0; j < p; j++) {
      const xij = xi[j];
      XtX[0][j + 1] += xij;
      XtX[j + 1][0] += xij;
      Xty[j + 1] += xij * yi;
      for (let k = j; k < p; k++) {
        XtX[j + 1][k + 1] += xij * xi[k];
      }
    }
  }

  for (let j = 0; j < p; j++) {
    for (let k = 0; k < j; k++) {
      XtX[j + 1][k + 1] = XtX[k + 1][j + 1];
    }
    XtX[j + 1][j + 1] += 1e-6; // numerical stability for collinear one-hot columns
  }

  const sol = solveLinearSystem(XtX, Xty);
  const bias = sol[0];
  const weights = sol.slice(1);
  return {
    weights: { kind: 'linear' as const, weights, bias },
    encodedImportances: weights.map((w) => Math.abs(w)),
  };
}

// ============================================================================
// 2. Lasso Regression (Coordinate Descent with L1 Soft-Thresholding)
// ============================================================================

async function fitLassoRegression(X: number[][], y: number[]) {
  const n = X.length;
  const p = X[0].length;

  const yMean = y.reduce((a, b) => a + b, 0) / n;
  const yVar = y.reduce((a, b) => a + (b - yMean) ** 2, 0) / n;
  const yStd = Math.sqrt(yVar) > 1e-8 ? Math.sqrt(yVar) : 1;
  const yScaled = y.map((v) => (v - yMean) / yStd);

  const weights = new Array(p).fill(0);
  let bias = 0;
  const alpha = 0.025; // gentle L1 penalty on standardized scale
  const maxIter = 180;

  // Precompute column squared norms
  const colSqSum = new Array(p).fill(0);
  for (let j = 0; j < p; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += X[i][j] * X[i][j];
    colSqSum[j] = s;
  }

  const residuals = [...yScaled];

  for (let iter = 0; iter < maxIter; iter++) {
    if (iter > 0 && iter % 40 === 0) {
      await yieldToBrowser();
    }

    // Update bias
    let resSum = 0;
    for (let i = 0; i < n; i++) resSum += residuals[i];
    const deltaB = resSum / n;
    if (Math.abs(deltaB) > 1e-12) {
      bias += deltaB;
      for (let i = 0; i < n; i++) residuals[i] -= deltaB;
    }

    // Coordinate update per feature
    for (let j = 0; j < p; j++) {
      if (colSqSum[j] < 1e-10) continue;
      const oldW = weights[j];
      let rho = 0;
      for (let i = 0; i < n; i++) {
        const partial = residuals[i] + oldW * X[i][j];
        rho += X[i][j] * partial;
      }

      const threshold = n * alpha;
      let newW = 0;
      if (rho > threshold) {
        newW = (rho - threshold) / colSqSum[j];
      } else if (rho < -threshold) {
        newW = (rho + threshold) / colSqSum[j];
      }

      const diff = newW - oldW;
      if (Math.abs(diff) > 1e-12) {
        weights[j] = newW;
        for (let i = 0; i < n; i++) {
          residuals[i] -= diff * X[i][j];
        }
      }
    }
  }

  // Rescale back to original target units
  const unscaledWeights = weights.map((w) => w * yStd);
  const unscaledBias = bias * yStd + yMean;

  return {
    weights: { kind: 'linear' as const, weights: unscaledWeights, bias: unscaledBias },
    encodedImportances: unscaledWeights.map((w) => Math.abs(w)),
  };
}

// ============================================================================
// 3. Support Vector Regression (Linear SVR with Epsilon-Insensitive Loss)
// ============================================================================

async function fitSVR(X: number[][], y: number[]) {
  const n = X.length;
  const p = X[0].length;

  const yMean = y.reduce((a, b) => a + b, 0) / n;
  const yVar = y.reduce((a, b) => a + (b - yMean) ** 2, 0) / n;
  const yStd = Math.sqrt(yVar) > 1e-8 ? Math.sqrt(yVar) : 1;
  const yScaled = y.map((v) => (v - yMean) / yStd);

  const weights = new Array(p).fill(0);
  let bias = 0;
  const epsilon = 0.08;
  const l2 = 0.02;
  const lr = 0.12;
  const epochs = 260;

  for (let ep = 0; ep < epochs; ep++) {
    if (ep > 0 && ep % 60 === 0) await yieldToBrowser();

    const gradW = new Array(p).fill(0);
    let gradB = 0;

    for (let i = 0; i < n; i++) {
      let pred = bias;
      const xi = X[i];
      for (let j = 0; j < p; j++) pred += weights[j] * xi[j];
      const err = pred - yScaled[i];
      if (Math.abs(err) > epsilon) {
        const sign = err > 0 ? 1 : -1;
        gradB += sign;
        for (let j = 0; j < p; j++) gradW[j] += sign * xi[j];
      }
    }

    const step = lr / (1 + ep * 0.015);
    bias -= step * (gradB / n);
    for (let j = 0; j < p; j++) {
      weights[j] -= step * (gradW[j] / n + l2 * weights[j]);
    }
  }

  const unscaledWeights = weights.map((w) => w * yStd);
  const unscaledBias = bias * yStd + yMean;

  return {
    weights: { kind: 'linear' as const, weights: unscaledWeights, bias: unscaledBias },
    encodedImportances: unscaledWeights.map((w) => Math.abs(w)),
  };
}

// ============================================================================
// 4. Support Vector Machine Classifier (Linear SVM with Hinge Loss + L2)
// ============================================================================

async function fitSVM(X: number[][], y: number[], numClasses: number) {
  const n = X.length;
  const p = X[0].length;
  const l2 = 0.015;
  const lr = 0.2;
  const epochs = 260;

  const trainBinaryHinge = async (targets: number[]) => {
    const w = new Array(p).fill(0);
    let b = 0;

    for (let ep = 0; ep < epochs; ep++) {
      if (ep > 0 && ep % 65 === 0) await yieldToBrowser();
      const gradW = new Array(p).fill(0);
      let gradB = 0;

      for (let i = 0; i < n; i++) {
        const xi = X[i];
        let margin = b;
        for (let j = 0; j < p; j++) margin += w[j] * xi[j];
        const yi = targets[i]; // +1 or -1
        if (yi * margin < 1) {
          gradB -= yi;
          for (let j = 0; j < p; j++) gradW[j] -= yi * xi[j];
        }
      }

      const step = lr / (1 + ep * 0.012);
      b -= step * (gradB / n);
      for (let j = 0; j < p; j++) {
        w[j] -= step * (gradW[j] / n + l2 * w[j]);
      }
    }
    return { w, b };
  };

  if (numClasses === 2) {
    const targets = y.map((yi) => (yi === 1 ? 1 : -1));
    const { w, b } = await trainBinaryHinge(targets);
    // Scale slightly by 1.5 for smoother sigmoid probability calibration while preserving sign
    return {
      weights: {
        kind: 'linear' as const,
        weights: w.map((v) => v * 1.5),
        bias: b * 1.5,
      },
      encodedImportances: w.map((v) => Math.abs(v)),
    };
  }

  // One-vs-Rest multiclass SVM
  const multiWeights: number[][] = [];
  const multiBias: number[] = [];
  const encodedImportances = new Array(p).fill(0);

  for (let c = 0; c < numClasses; c++) {
    const targets = y.map((yi) => (yi === c ? 1 : -1));
    const { w, b } = await trainBinaryHinge(targets);
    multiWeights.push(w);
    multiBias.push(b);
    w.forEach((v, idx) => {
      encodedImportances[idx] += Math.abs(v);
    });
  }

  return {
    weights: {
      kind: 'linear' as const,
      weights: [],
      bias: 0,
      multiWeights,
      multiBias,
    },
    encodedImportances,
  };
}

// ============================================================================
// 5. K-Nearest Neighbors (Classifier & Regressor)
// ============================================================================

function computeKnnFeatureImportance(
  X: number[][],
  y: number[],
  taskType: SupervisedTaskType,
  numClasses: number
): number[] {
  const n = X.length;
  const p = X[0].length;
  const scores = new Array(p).fill(0);

  if (taskType === 'regression') {
    const yMean = y.reduce((a, b) => a + b, 0) / n;
    for (let j = 0; j < p; j++) {
      let xMean = 0;
      for (let i = 0; i < n; i++) xMean += X[i][j];
      xMean /= n;
      let num = 0;
      let dx2 = 0;
      let dy2 = 0;
      for (let i = 0; i < n; i++) {
        const dx = X[i][j] - xMean;
        const dy = y[i] - yMean;
        num += dx * dy;
        dx2 += dx * dx;
        dy2 += dy * dy;
      }
      const denom = Math.sqrt(dx2 * dy2);
      scores[j] = denom > 1e-10 ? Math.abs(num / denom) : 0;
    }
    return scores;
  }

  // Classification: between-class variance / total variance ratio
  for (let j = 0; j < p; j++) {
    let globalMean = 0;
    for (let i = 0; i < n; i++) globalMean += X[i][j];
    globalMean /= n;

    const classSums = new Array(numClasses).fill(0);
    const classCounts = new Array(numClasses).fill(0);
    for (let i = 0; i < n; i++) {
      classSums[y[i]] += X[i][j];
      classCounts[y[i]]++;
    }

    let between = 0;
    for (let c = 0; c < numClasses; c++) {
      if (classCounts[c] > 0) {
        const cMean = classSums[c] / classCounts[c];
        between += classCounts[c] * (cMean - globalMean) ** 2;
      }
    }

    let totalVar = 0;
    for (let i = 0; i < n; i++) {
      totalVar += (X[i][j] - globalMean) ** 2;
    }
    scores[j] = totalVar > 1e-10 ? between / totalVar : 0;
  }
  return scores;
}

async function fitKNN(
  X: number[][],
  y: number[],
  taskType: SupervisedTaskType,
  numClasses: number
) {
  await yieldToBrowser();
  const maxRefPoints = 800;
  let refX = X;
  let refY = y;

  if (X.length > maxRefPoints) {
    const step = X.length / maxRefPoints;
    refX = [];
    refY = [];
    for (let i = 0; i < maxRefPoints; i++) {
      const idx = Math.min(X.length - 1, Math.floor(i * step));
      refX.push(X[idx]);
      refY.push(y[idx]);
    }
  }

  const k = Math.min(5, Math.max(1, refX.length));
  const encodedImportances = computeKnnFeatureImportance(X, y, taskType, numClasses);

  return {
    weights: {
      kind: 'knn' as const,
      k,
      XTrain: refX,
      yTrain: refY,
    },
    encodedImportances,
  };
}

// ============================================================================
// 6. Naive Bayes (Gaussian & Multinomial)
// ============================================================================

async function fitNaiveBayes(X: number[][], y: number[], numClasses: number) {
  await yieldToBrowser();
  const n = X.length;
  const p = X[0].length;

  // Detect if features are strictly non-negative binary/count indicators -> Multinomial NB
  let isNonNegative = true;
  for (let i = 0; i < Math.min(n, 200); i++) {
    for (let j = 0; j < p; j++) {
      if (X[i][j] < -1e-9) {
        isNonNegative = false;
        break;
      }
    }
    if (!isNonNegative) break;
  }

  const classCounts = new Array(numClasses).fill(0);
  for (let i = 0; i < n; i++) classCounts[y[i]]++;
  const priors = classCounts.map((c) => Math.max(1e-6, c / n));

  if (isNonNegative) {
    // Multinomial Naive Bayes with Laplace smoothing alpha = 1.0
    const alpha = 1.0;
    const logProbs: number[][] = Array.from({ length: numClasses }, () =>
      new Array(p).fill(0)
    );
    const featureClassRates: number[][] = Array.from({ length: numClasses }, () =>
      new Array(p).fill(0)
    );

    for (let c = 0; c < numClasses; c++) {
      const featSums = new Array(p).fill(0);
      let totalCount = 0;
      for (let i = 0; i < n; i++) {
        if (y[i] === c) {
          for (let j = 0; j < p; j++) {
            featSums[j] += Math.max(0, X[i][j]);
            totalCount += Math.max(0, X[i][j]);
          }
        }
      }
      const denom = totalCount + alpha * p;
      for (let j = 0; j < p; j++) {
        const prob = (featSums[j] + alpha) / Math.max(1e-9, denom);
        logProbs[c][j] = Math.log(prob);
        featureClassRates[c][j] = prob;
      }
    }

    const encodedImportances = new Array(p).fill(0);
    for (let j = 0; j < p; j++) {
      const vals = featureClassRates.map((r) => r[j]);
      encodedImportances[j] = Math.max(...vals) - Math.min(...vals);
    }

    return {
      variant: 'multinomial' as const,
      weights: {
        kind: 'naive_bayes' as const,
        variant: 'multinomial' as const,
        priors,
        logProbs,
      },
      encodedImportances,
    };
  }

  // Gaussian Naive Bayes with variance smoothing
  const means: number[][] = Array.from({ length: numClasses }, () => new Array(p).fill(0));
  const variances: number[][] = Array.from({ length: numClasses }, () => new Array(p).fill(1));

  // Compute global feature variance for smoothing floor
  let maxGlobalVar = 1e-4;
  for (let j = 0; j < p; j++) {
    let m = 0;
    for (let i = 0; i < n; i++) m += X[i][j];
    m /= n;
    let v = 0;
    for (let i = 0; i < n; i++) v += (X[i][j] - m) ** 2;
    v /= n;
    if (v > maxGlobalVar) maxGlobalVar = v;
  }
  const varFloor = 1e-4 * maxGlobalVar + 1e-6;

  for (let c = 0; c < numClasses; c++) {
    const count = Math.max(1, classCounts[c]);
    for (let i = 0; i < n; i++) {
      if (y[i] === c) {
        for (let j = 0; j < p; j++) means[c][j] += X[i][j];
      }
    }
    for (let j = 0; j < p; j++) means[c][j] /= count;

    for (let i = 0; i < n; i++) {
      if (y[i] === c) {
        for (let j = 0; j < p; j++) {
          variances[c][j] += (X[i][j] - means[c][j]) ** 2;
        }
      }
    }
    for (let j = 0; j < p; j++) {
      variances[c][j] = variances[c][j] / count + varFloor;
    }
  }

  // Feature importance: standardized mean separation across classes
  const encodedImportances = new Array(p).fill(0);
  for (let j = 0; j < p; j++) {
    let sep = 0;
    for (let c1 = 0; c1 < numClasses; c1++) {
      for (let c2 = c1 + 1; c2 < numClasses; c2++) {
        const pooledStd = Math.sqrt((variances[c1][j] + variances[c2][j]) / 2);
        sep += Math.abs(means[c1][j] - means[c2][j]) / Math.max(1e-6, pooledStd);
      }
    }
    encodedImportances[j] = sep;
  }

  return {
    variant: 'gaussian' as const,
    weights: {
      kind: 'naive_bayes' as const,
      variant: 'gaussian' as const,
      priors,
      means,
      variances,
    },
    encodedImportances,
  };
}

// ============================================================================
// 7. Linear Discriminant Analysis (LDA)
// ============================================================================

async function fitLDA(X: number[][], y: number[], numClasses: number) {
  await yieldToBrowser();
  const n = X.length;
  const p = X[0].length;

  const classCounts = new Array(numClasses).fill(0);
  const classMeans: number[][] = Array.from({ length: numClasses }, () =>
    new Array(p).fill(0)
  );

  for (let i = 0; i < n; i++) {
    const c = y[i];
    classCounts[c]++;
    for (let j = 0; j < p; j++) {
      classMeans[c][j] += X[i][j];
    }
  }

  for (let c = 0; c < numClasses; c++) {
    const cnt = Math.max(1, classCounts[c]);
    for (let j = 0; j < p; j++) classMeans[c][j] /= cnt;
  }

  // Pooled within-class covariance matrix
  const cov: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));
  for (let i = 0; i < n; i++) {
    const c = y[i];
    const mu = classMeans[c];
    const xi = X[i];
    for (let j = 0; j < p; j++) {
      const dj = xi[j] - mu[j];
      for (let k = j; k < p; k++) {
        cov[j][k] += dj * (xi[k] - mu[k]);
      }
    }
  }

  const denom = Math.max(1, n - numClasses);
  for (let j = 0; j < p; j++) {
    for (let k = j; k < p; k++) {
      cov[j][k] /= denom;
      cov[k][j] = cov[j][k];
    }
  }

  // Ledoit-Wolf style shrinkage towards diagonal for guaranteed positive-definiteness
  const shrinkage = 0.08;
  for (let j = 0; j < p; j++) {
    for (let k = 0; k < p; k++) {
      if (j !== k) {
        cov[j][k] *= 1 - shrinkage;
      } else {
        cov[j][j] = cov[j][j] + 1e-3;
      }
    }
  }

  const multiWeights: number[][] = [];
  const multiBias: number[] = [];
  const encodedImportances = new Array(p).fill(0);

  for (let c = 0; c < numClasses; c++) {
    const w = solveLinearSystem(cov, classMeans[c]);
    const prior = Math.max(1e-6, classCounts[c] / n);
    let quad = 0;
    for (let j = 0; j < p; j++) quad += classMeans[c][j] * w[j];
    const b = -0.5 * quad + Math.log(prior);
    multiWeights.push(w);
    multiBias.push(b);
  }

  // Importance = max difference in discriminant weights across classes
  for (let j = 0; j < p; j++) {
    const vals = multiWeights.map((wRow) => wRow[j]);
    encodedImportances[j] = Math.max(...vals) - Math.min(...vals);
  }

  return {
    weights: {
      kind: 'linear' as const,
      weights: [],
      bias: 0,
      multiWeights,
      multiBias,
    },
    encodedImportances,
  };
}

// ============================================================================
// 8. AdaBoost Classifier (SAMME with Weighted Decision Stumps)
// ============================================================================

function fitWeightedStump(
  X: number[][],
  y: number[],
  sampleWeights: number[],
  numClasses: number
): { stump: SerializedTreeNode; errRate: number; bestFeature: number; gain: number } {
  const n = X.length;
  const p = X[0].length;

  let bestFeature = 0;
  let bestThreshold = 0;
  let bestWeightedErr = Infinity;
  let bestLeftClass = 0;
  let bestRightClass = 0;

  for (let j = 0; j < p; j++) {
    const order = Array.from({ length: n }, (_, i) => i).sort(
      (a, b) => X[a][j] - X[b][j]
    );

    const totalClassW = new Array(numClasses).fill(0);
    for (let i = 0; i < n; i++) {
      totalClassW[y[i]] += sampleWeights[i];
    }

    const leftClassW = new Array(numClasses).fill(0);
    for (let pos = 0; pos < n - 1; pos++) {
      const idx = order[pos];
      leftClassW[y[idx]] += sampleWeights[idx];
      const nextIdx = order[pos + 1];
      if (Math.abs(X[idx][j] - X[nextIdx][j]) < 1e-9) continue;

      let leftBestClass = 0;
      let leftBestW = -1;
      let leftTotal = 0;
      let rightBestClass = 0;
      let rightBestW = -1;
      let rightTotal = 0;

      for (let c = 0; c < numClasses; c++) {
        const lw = leftClassW[c];
        const rw = totalClassW[c] - lw;
        leftTotal += lw;
        rightTotal += rw;
        if (lw > leftBestW) {
          leftBestW = lw;
          leftBestClass = c;
        }
        if (rw > rightBestW) {
          rightBestW = rw;
          rightBestClass = c;
        }
      }

      const weightedErr = (leftTotal - leftBestW) + (rightTotal - rightBestW);
      if (weightedErr < bestWeightedErr) {
        bestWeightedErr = weightedErr;
        bestFeature = j;
        bestThreshold = (X[idx][j] + X[nextIdx][j]) / 2;
        bestLeftClass = leftBestClass;
        bestRightClass = rightBestClass;
      }
    }
  }

  // Fallback if all feature values are identical
  if (!Number.isFinite(bestWeightedErr)) {
    const classW = new Array(numClasses).fill(0);
    for (let i = 0; i < n; i++) classW[y[i]] += sampleWeights[i];
    const maj = classW.indexOf(Math.max(...classW));
    const probs = classW.map((_, c) => (c === maj ? 1 : 0));
    return {
      stump: { isLeaf: true, probs },
      errRate: Math.max(0.01, 1 - classW[maj]),
      bestFeature: 0,
      gain: 0,
    };
  }

  const leftProbs = new Array(numClasses).fill(0);
  leftProbs[bestLeftClass] = 1;
  const rightProbs = new Array(numClasses).fill(0);
  rightProbs[bestRightClass] = 1;

  const stump: SerializedTreeNode = {
    isLeaf: false,
    featureIndex: bestFeature,
    threshold: bestThreshold,
    left: { isLeaf: true, probs: leftProbs },
    right: { isLeaf: true, probs: rightProbs },
  };

  return {
    stump,
    errRate: Math.min(0.99, Math.max(1e-6, bestWeightedErr)),
    bestFeature,
    gain: Math.max(0, 1 - 2 * bestWeightedErr),
  };
}

function predictStumpClass(stump: SerializedTreeNode, x: number[]): number {
  if (stump.isLeaf) {
    const probs = stump.probs ?? [1];
    return probs.indexOf(Math.max(...probs));
  }
  const branch = x[stump.featureIndex!] <= stump.threshold! ? stump.left! : stump.right!;
  const probs = branch.probs ?? [1];
  return probs.indexOf(Math.max(...probs));
}

async function fitAdaBoost(X: number[][], y: number[], numClasses: number) {
  const n = X.length;
  const p = X[0].length;
  const nEstimators = 30;
  const learningRate = 0.8;

  let sampleWeights = new Array(n).fill(1 / n);
  const stumps: SerializedTreeNode[] = [];
  const alphas: number[] = [];
  const encodedImportances = new Array(p).fill(0);

  for (let m = 0; m < nEstimators; m++) {
    if (m > 0 && m % 6 === 0) await yieldToBrowser();

    const { stump, errRate, bestFeature, gain } = fitWeightedStump(
      X,
      y,
      sampleWeights,
      numClasses
    );

    if (errRate >= 1 - 1 / numClasses) {
      if (stumps.length === 0) {
        stumps.push(stump);
        alphas.push(0.1);
      }
      break;
    }

    const alpha =
      learningRate * (Math.log((1 - errRate) / errRate) + Math.log(numClasses - 1));
    stumps.push(stump);
    alphas.push(alpha);
    encodedImportances[bestFeature] += Math.abs(alpha) * Math.max(0.05, gain);

    let weightSum = 0;
    const nextWeights = new Array(n);
    for (let i = 0; i < n; i++) {
      const pred = predictStumpClass(stump, X[i]);
      const mult = pred !== y[i] ? Math.exp(alpha) : 1;
      nextWeights[i] = sampleWeights[i] * mult;
      weightSum += nextWeights[i];
    }
    if (weightSum <= 1e-12) break;
    sampleWeights = nextWeights.map((w) => w / weightSum);
  }

  return {
    weights: {
      kind: 'adaboost' as const,
      numClasses,
      stumps,
      alphas,
    },
    encodedImportances,
  };
}

// ============================================================================
// Dispatcher & Inference for Expanded Supervised Algorithms
// ============================================================================

export async function trainExtraSupervisedAlgorithm(
  algoKey: ExpandedSupervisedModelKey,
  XTrain: number[][],
  yTrain: number[],
  taskType: SupervisedTaskType,
  numClasses: number
): Promise<{
  name: string;
  sklearnName: string;
  family: string;
  weights: SerializedModelWeights;
  encodedImportances: number[];
}> {
  if (algoKey === 'linear_regression') {
    const res = fitLinearRegression(XTrain, yTrain);
    return {
      name: 'Linear Regression',
      sklearnName: 'LinearRegression',
      family: 'Linear (OLS)',
      ...res,
    };
  }

  if (algoKey === 'lasso') {
    const res = await fitLassoRegression(XTrain, yTrain);
    return {
      name: 'Lasso Regression',
      sklearnName: 'Lasso',
      family: 'Linear (L1)',
      ...res,
    };
  }

  if (algoKey === 'svr') {
    const res = await fitSVR(XTrain, yTrain);
    return {
      name: 'Support Vector Regression (SVR)',
      sklearnName: 'SVR',
      family: 'Support Vector',
      ...res,
    };
  }

  if (algoKey === 'svm') {
    const res = await fitSVM(XTrain, yTrain, numClasses);
    return {
      name: 'Support Vector Machine (SVM)',
      sklearnName: 'SVC',
      family: 'Support Vector',
      ...res,
    };
  }

  if (algoKey === 'knn') {
    const res = await fitKNN(XTrain, yTrain, taskType, numClasses);
    return {
      name: taskType === 'regression' ? 'KNN Regressor' : 'KNN Classifier',
      sklearnName: taskType === 'regression' ? 'KNeighborsRegressor' : 'KNeighborsClassifier',
      family: 'Instance-Based (k=5)',
      ...res,
    };
  }

  if (algoKey === 'naive_bayes') {
    const res = await fitNaiveBayes(XTrain, yTrain, numClasses);
    return {
      name:
        res.variant === 'multinomial'
          ? 'Naive Bayes (Multinomial)'
          : 'Naive Bayes (Gaussian)',
      sklearnName: res.variant === 'multinomial' ? 'MultinomialNB' : 'GaussianNB',
      family: 'Probabilistic Bayes',
      weights: res.weights,
      encodedImportances: res.encodedImportances,
    };
  }

  if (algoKey === 'lda') {
    const res = await fitLDA(XTrain, yTrain, numClasses);
    return {
      name: 'Linear Discriminant Analysis (LDA)',
      sklearnName: 'LinearDiscriminantAnalysis',
      family: 'Discriminant Analysis',
      ...res,
    };
  }

  // adaboost
  const res = await fitAdaBoost(XTrain, yTrain, numClasses);
  return {
    name: 'AdaBoost Classifier',
    sklearnName: 'AdaBoostClassifier',
    family: 'Boosted Stumps (SAMME)',
    ...res,
  };
}

export function predictExtraVector(
  weights: SerializedModelWeights,
  x: number[],
  taskType: SupervisedTaskType,
  numClasses: number
): { value: number; probs?: number[] } {
  if (weights.kind === 'knn') {
    const { k, XTrain, yTrain } = weights;
    const dists: { dist: number; target: number }[] = [];
    for (let i = 0; i < XTrain.length; i++) {
      const xi = XTrain[i];
      let d2 = 0;
      for (let j = 0; j < x.length; j++) {
        const diff = x[j] - xi[j];
        d2 += diff * diff;
      }
      dists.push({ dist: Math.sqrt(d2), target: yTrain[i] });
    }
    dists.sort((a, b) => a.dist - b.dist);
    const neighbors = dists.slice(0, k);

    if (taskType === 'regression') {
      let wSum = 0;
      let valSum = 0;
      for (const nb of neighbors) {
        const w = 1 / (nb.dist + 1e-4);
        wSum += w;
        valSum += w * nb.target;
      }
      return { value: valSum / Math.max(1e-9, wSum) };
    }

    const classScores = new Array(numClasses).fill(0);
    let wTotal = 0;
    for (const nb of neighbors) {
      const w = 1 / (nb.dist + 1e-4);
      const cls = Math.round(nb.target);
      if (cls >= 0 && cls < numClasses) {
        classScores[cls] += w;
      }
      wTotal += w;
    }
    const probs = classScores.map((s) => s / Math.max(1e-9, wTotal));
    return { value: probs.indexOf(Math.max(...probs)), probs };
  }

  if (weights.kind === 'naive_bayes') {
    const logPost = new Array(numClasses).fill(0);
    if (weights.variant === 'multinomial' && weights.logProbs) {
      for (let c = 0; c < numClasses; c++) {
        let lp = Math.log(weights.priors[c]);
        for (let j = 0; j < x.length; j++) {
          lp += Math.max(0, x[j]) * weights.logProbs[c][j];
        }
        logPost[c] = lp;
      }
    } else if (weights.means && weights.variances) {
      for (let c = 0; c < numClasses; c++) {
        let lp = Math.log(weights.priors[c]);
        for (let j = 0; j < x.length; j++) {
          const v = weights.variances[c][j];
          const diff = x[j] - weights.means[c][j];
          lp += -0.5 * (Math.log(2 * Math.PI * v) + (diff * diff) / v);
        }
        logPost[c] = lp;
      }
    }
    const probs = softmax(logPost);
    return { value: probs.indexOf(Math.max(...probs)), probs };
  }

  if (weights.kind === 'adaboost') {
    const scores = new Array(numClasses).fill(0);
    for (let m = 0; m < weights.stumps.length; m++) {
      const cls = predictStumpClass(weights.stumps[m], x);
      scores[cls] += weights.alphas[m];
    }
    const probs = softmax(scores);
    return { value: probs.indexOf(Math.max(...probs)), probs };
  }

  // Fallback for linear
  if (weights.kind === 'linear') {
    let z = weights.bias;
    for (let j = 0; j < x.length; j++) z += weights.weights[j] * x[j];
    if (taskType === 'regression') return { value: z };
    const p1 = sigmoid(z);
    return { value: p1 >= 0.5 ? 1 : 0, probs: [1 - p1, p1] };
  }

  return { value: 0 };
}
