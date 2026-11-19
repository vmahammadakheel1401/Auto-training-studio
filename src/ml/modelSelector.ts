import { inferTaskType, isMissingValue } from './csvParser';
import { isBinaryToken } from './unsupervised';
import {
  LearningMode,
  ModelAlgorithmKey,
  ModelSelectionInfo,
  ParsedDataset,
  SupervisedModelKey,
  TaskType,
  UnsupervisedModelKey,
} from './types';

export interface ModelCatalogItem {
  key: ModelAlgorithmKey;
  label: string;
  family: string;
  mode: LearningMode;
  task: TaskType;
}

export const SUPERVISED_REGRESSION_CATALOG: ModelCatalogItem[] = [
  { key: 'linear_regression', label: 'Linear Regression (OLS)', family: 'Linear', mode: 'supervised', task: 'regression' },
  { key: 'ridge', label: 'Ridge Regression (L2)', family: 'Linear', mode: 'supervised', task: 'regression' },
  { key: 'lasso', label: 'Lasso Regression (L1)', family: 'Linear', mode: 'supervised', task: 'regression' },
  { key: 'svr', label: 'Support Vector Regression (SVR)', family: 'Support Vector', mode: 'supervised', task: 'regression' },
  { key: 'knn', label: 'KNN Regressor (k=5)', family: 'Instance-Based', mode: 'supervised', task: 'regression' },
  { key: 'decision_tree', label: 'Decision Tree Regressor', family: 'Tree', mode: 'supervised', task: 'regression' },
  { key: 'random_forest', label: 'Random Forest Regressor', family: 'Ensemble', mode: 'supervised', task: 'regression' },
  { key: 'gradient_boosting', label: 'Gradient Boosting Regressor', family: 'Ensemble', mode: 'supervised', task: 'regression' },
];

export const SUPERVISED_CLASSIFICATION_CATALOG: ModelCatalogItem[] = [
  { key: 'logistic', label: 'Logistic Regression', family: 'Linear', mode: 'supervised', task: 'classification' },
  { key: 'svm', label: 'Support Vector Machine (SVM)', family: 'Support Vector', mode: 'supervised', task: 'classification' },
  { key: 'knn', label: 'KNN Classifier (k=5)', family: 'Instance-Based', mode: 'supervised', task: 'classification' },
  { key: 'naive_bayes', label: 'Naive Bayes (Gaussian / Multinomial)', family: 'Probabilistic', mode: 'supervised', task: 'classification' },
  { key: 'lda', label: 'Linear Discriminant Analysis (LDA)', family: 'Discriminant', mode: 'supervised', task: 'classification' },
  { key: 'decision_tree', label: 'Decision Tree Classifier', family: 'Tree', mode: 'supervised', task: 'classification' },
  { key: 'random_forest', label: 'Random Forest Classifier', family: 'Ensemble', mode: 'supervised', task: 'classification' },
  { key: 'gradient_boosting', label: 'Gradient Boosting Classifier', family: 'Ensemble', mode: 'supervised', task: 'classification' },
  { key: 'adaboost', label: 'AdaBoost Classifier', family: 'Ensemble', mode: 'supervised', task: 'classification' },
];

export const UNSUPERVISED_CATALOG: ModelCatalogItem[] = [
  { key: 'pca', label: 'PCA (Principal Component Analysis)', family: 'Dimensionality Reduction', mode: 'unsupervised', task: 'pca' },
  { key: 'kmeans', label: 'K-Means Clustering', family: 'Centroid Clustering', mode: 'unsupervised', task: 'clustering' },
  { key: 'hierarchical', label: 'Hierarchical Clustering (Agglomerative)', family: 'Ward Linkage', mode: 'unsupervised', task: 'clustering' },
  { key: 'gmm', label: 'Gaussian Mixture Model (GMM)', family: 'Probabilistic EM', mode: 'unsupervised', task: 'clustering' },
  { key: 'apriori', label: 'Apriori (Association Rule Mining)', family: 'Frequent Pattern Mining', mode: 'unsupervised', task: 'association' },
];

const TARGET_COLUMN_HINTS = new Set([
  'target',
  'label',
  'class',
  'y',
  'outcome',
  'status',
  'churn',
  'price',
  'sale_price',
  'salary',
  'revenue',
  'sales',
  'diagnosis',
  'fraud',
  'default',
  'survived',
  'species',
  'quality',
  'medv',
  'rating',
  'score',
]);

function inspectBinaryAndTransactional(
  dataset: ParsedDataset,
  featureCols: string[]
): {
  binaryColCount: number;
  binaryColRatio: number;
  hasDelimitedItemsCol: boolean;
  isTransactional: boolean;
} {
  if (featureCols.length === 0 || dataset.rows.length === 0) {
    return {
      binaryColCount: 0,
      binaryColRatio: 0,
      hasDelimitedItemsCol: false,
      isTransactional: false,
    };
  }

  let binaryColCount = 0;
  for (const col of featureCols) {
    const nonMissing = dataset.rows
      .map((r) => r[col])
      .filter((v) => !isMissingValue(v));
    if (nonMissing.length === 0) continue;
    const hits = nonMissing.filter((v) => isBinaryToken(v)).length;
    if (hits / nonMissing.length >= 0.9) {
      binaryColCount++;
    }
  }

  const hasDelimitedItemsCol = featureCols.some((col) => {
    const sample = dataset.rows.slice(0, 25).map((r) => r[col] ?? '');
    const delimHits = sample.filter(
      (v) => /[;|]/.test(v) || (v.includes(',') && v.split(',').length >= 2)
    ).length;
    return delimHits >= Math.max(3, sample.length * 0.4);
  });

  const binaryColRatio = binaryColCount / featureCols.length;
  const isTransactional =
    hasDelimitedItemsCol || (binaryColCount >= 4 && binaryColRatio >= 0.65);

  return {
    binaryColCount,
    binaryColRatio,
    hasDelimitedItemsCol,
    isTransactional,
  };
}

/**
 * Infer whether an imported dataset should start in Supervised or Unsupervised mode.
 */
export function inferInitialLearningMode(
  dataset: ParsedDataset,
  explicitMode?: LearningMode
): LearningMode {
  if (explicitMode) return explicitMode;

  const lowerFile = dataset.fileName.toLowerCase();
  if (
    lowerFile.includes('segment') ||
    lowerFile.includes('cluster') ||
    lowerFile.includes('basket') ||
    lowerFile.includes('unlabeled') ||
    lowerFile.includes('apriori') ||
    lowerFile.includes('pca')
  ) {
    return 'unsupervised';
  }

  const nonIdCols = dataset.columnStats
    .filter((c) => !c.isLikelyId)
    .map((c) => c.name);
  const hasExplicitTargetName = nonIdCols.some((name) =>
    TARGET_COLUMN_HINTS.has(name.toLowerCase())
  );

  const txCheck = inspectBinaryAndTransactional(dataset, nonIdCols);
  if (txCheck.isTransactional && !hasExplicitTargetName) {
    return 'unsupervised';
  }

  return 'supervised';
}

export function getUnsupervisedTaskForModel(key: UnsupervisedModelKey): TaskType {
  if (key === 'pca') return 'pca';
  if (key === 'apriori') return 'association';
  return 'clustering';
}

/**
 * Inspect dataset metadata and automatically select the single best model.
 */
export function inspectAndSelectBestModel(params: {
  dataset: ParsedDataset;
  learningMode: LearningMode;
  targetColumn: string;
  droppedIdColumns: string[];
  manualOverrideModel?: ModelAlgorithmKey | null;
}): ModelSelectionInfo {
  const { dataset, learningMode, targetColumn, droppedIdColumns, manualOverrideModel } =
    params;

  const droppedSet = new Set(droppedIdColumns);
  const rowCount = dataset.totalRows;

  if (learningMode === 'unsupervised') {
    const activeCols = dataset.columnStats.filter((c) => !droppedSet.has(c.name));
    const numCols = activeCols.filter((c) => c.inferredType === 'numeric');
    const catCols = activeCols.filter((c) => c.inferredType === 'categorical');
    const featCount = activeCols.length;
    const estEncodedDims = activeCols.reduce(
      (sum, c) =>
        sum + (c.inferredType === 'numeric' ? 1 : Math.min(20, Math.max(1, c.uniqueCount))),
      0
    );

    const tx = inspectBinaryAndTransactional(
      dataset,
      activeCols.map((c) => c.name)
    );

    // Compute std disparity across numeric columns to help distinguish spherical vs elliptical clusters
    let maxStd = 0;
    let minStd = Infinity;
    for (const col of numCols) {
      const vals = dataset.rows
        .map((r) => Number((r[col.name] ?? '').replace(/[$,%]/g, '')))
        .filter((n) => Number.isFinite(n));
      if (vals.length > 2) {
        const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
        const std = Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length);
        if (std > maxStd) maxStd = std;
        if (std < minStd && std > 1e-6) minStd = std;
      }
    }
    const scaleDisparity = minStd < Infinity && minStd > 0 ? maxStd / minStd : 1;

    let autoKey: UnsupervisedModelKey = 'kmeans';
    let reason = '';

    if (tx.isTransactional) {
      autoKey = 'apriori';
      reason = tx.hasDelimitedItemsCol
        ? `Detected delimited transaction basket items across ${rowCount} unlabeled records — selected Apriori to mine frequent itemsets and association rules.`
        : `Detected ${tx.binaryColCount}/${featCount} binary item-presence indicators (${Math.round(tx.binaryColRatio * 100)}%) with no target column — selected Apriori Association Rule Mining.`;
    } else if (estEncodedDims >= 9 || numCols.length >= 8 || rowCount < featCount * 5) {
      autoKey = 'pca';
      reason = `Detected high-dimensional unlabeled feature space (${featCount} active features, ~${estEncodedDims} encoded dims across ${rowCount} rows) — selected PCA for variance compression and component projection.`;
    } else if (rowCount <= 140 && numCols.length >= 2 && catCols.length === 0) {
      autoKey = 'hierarchical';
      reason = `Compact continuous unlabeled dataset (${rowCount} rows, ${numCols.length} numeric features) — selected Hierarchical Agglomerative Clustering (Ward linkage) to reveal nested cluster structure.`;
    } else if (numCols.length >= 3 && catCols.length === 0 && scaleDisparity >= 4.5 && rowCount >= 120) {
      autoKey = 'gmm';
      reason = `Continuous multi-scale unlabeled features (${numCols.length} numeric features, scale disparity ${scaleDisparity.toFixed(1)}×) — selected Gaussian Mixture Model (EM) for soft probabilistic cluster boundaries.`;
    } else {
      autoKey = 'kmeans';
      reason = `Unlabeled tabular dataset (${rowCount} rows, ${numCols.length} numeric & ${catCols.length} categorical features) — selected K-Means Clustering with automatic Silhouette-guided K selection.`;
    }

    const chosenKey = (
      manualOverrideModel &&
      UNSUPERVISED_CATALOG.some((m) => m.key === manualOverrideModel)
        ? manualOverrideModel
        : autoKey
    ) as UnsupervisedModelKey;

    const isManualOverride = chosenKey !== autoKey;
    const taskType = getUnsupervisedTaskForModel(chosenKey);

    return {
      learningMode: 'unsupervised',
      taskType,
      selectedModelKey: chosenKey,
      autoRecommendedModelKey: autoKey,
      isManualOverride,
      reason: isManualOverride
        ? `Manual override active. Auto-selector originally recommended ${
            UNSUPERVISED_CATALOG.find((m) => m.key === autoKey)?.label ?? autoKey
          }: ${reason}`
        : reason,
      signals: [
        'Unlabeled / No Target',
        `${rowCount} rows`,
        `${featCount} active features`,
        tx.isTransactional
          ? 'Transactional Basket'
          : `${numCols.length} num · ${catCols.length} cat`,
      ],
    };
  }

  // Supervised Mode
  const supervisedTask = inferTaskType(dataset, targetColumn);
  const featureStats = dataset.columnStats.filter(
    (c) => c.name !== targetColumn && !droppedSet.has(c.name)
  );
  const numCount = featureStats.filter((c) => c.inferredType === 'numeric').length;
  const catCount = featureStats.filter((c) => c.inferredType === 'categorical').length;
  const featCount = featureStats.length;
  const estEncodedDims = featureStats.reduce(
    (sum, c) =>
      sum + (c.inferredType === 'numeric' ? 1 : Math.min(20, Math.max(1, c.uniqueCount))),
    0
  );

  let autoKey: SupervisedModelKey;
  let reason = '';
  const extraSignal =
    supervisedTask === 'classification'
      ? `${dataset.columnStats.find((c) => c.name === targetColumn)?.uniqueCount ?? 2} classes`
      : 'Continuous target';

  if (supervisedTask === 'regression') {
    if (rowCount < 45) {
      autoKey = 'ridge';
      reason = `Small regression sample size (${rowCount} rows, ${featCount} features) — selected Ridge Regression (L2) to prevent overfitting while capturing linear trends.`;
    } else if (estEncodedDims > rowCount * 0.32) {
      autoKey = 'lasso';
      reason = `High feature-to-row dimensionality (~${estEncodedDims} encoded dims for ${rowCount} rows) — selected Lasso Regression (L1) for automatic feature sparsity.`;
    } else if (catCount === 0 && numCount <= 4 && rowCount <= 150) {
      autoKey = 'svr';
      reason = `Compact all-numeric regression dataset (${rowCount} rows, ${numCount} numeric features) — selected Support Vector Regression (SVR) for margin-robust fitting.`;
    } else if (catCount > numCount && rowCount >= 80) {
      autoKey = 'random_forest';
      reason = `Categorical-heavy regression dataset (${catCount} categorical vs ${numCount} numeric features, ${rowCount} rows) — selected Random Forest Regressor for robust split ensembles.`;
    } else {
      autoKey = 'gradient_boosting';
      reason = `Tabular regression dataset (${rowCount} rows, ${numCount} numeric & ${catCount} categorical features) — selected Gradient Boosting Regressor to capture non-linear feature interactions.`;
    }

    const validKeys = new Set(SUPERVISED_REGRESSION_CATALOG.map((m) => m.key));
    const chosenKey = (
      manualOverrideModel && validKeys.has(manualOverrideModel)
        ? manualOverrideModel
        : autoKey
    ) as SupervisedModelKey;
    const isManualOverride = chosenKey !== autoKey;

    return {
      learningMode: 'supervised',
      taskType: 'regression',
      selectedModelKey: chosenKey,
      autoRecommendedModelKey: autoKey,
      isManualOverride,
      reason: isManualOverride
        ? `Manual override active. Auto-selector originally recommended ${
            SUPERVISED_REGRESSION_CATALOG.find((m) => m.key === autoKey)?.label ?? autoKey
          }: ${reason}`
        : reason,
      signals: [
        `Target: ${targetColumn}`,
        extraSignal,
        `${rowCount} rows`,
        `${numCount} num · ${catCount} cat`,
      ],
    };
  }

  // Classification
  const targetStat = dataset.columnStats.find((c) => c.name === targetColumn);
  const classCount = targetStat?.uniqueCount ?? 2;

  if (catCount >= Math.max(2, numCount * 2) && rowCount <= 400) {
    autoKey = 'naive_bayes';
    reason = `Predominantly categorical classification features (${catCount} categorical vs ${numCount} numeric, ${classCount} classes) — selected Naive Bayes for fast conditional probability modeling.`;
  } else if (rowCount < 65 && catCount === 0) {
    autoKey = classCount === 2 ? 'svm' : 'lda';
    reason = `Small all-numeric classification dataset (${rowCount} rows, ${classCount} classes) — selected ${
      autoKey === 'svm' ? 'SVM' : 'Linear Discriminant Analysis (LDA)'
    } for low-variance decision boundaries.`;
  } else if (catCount === 0 && numCount <= 6 && rowCount <= 180) {
    autoKey = classCount > 2 ? 'lda' : 'knn';
    reason = `Compact numeric classification dataset (${rowCount} rows, ${numCount} numeric features, ${classCount} classes) — selected ${
      autoKey === 'lda' ? 'LDA' : 'KNN Classifier'
    }.`;
  } else if (rowCount < 110) {
    autoKey = 'logistic';
    reason = `Moderate-size mixed classification dataset (${rowCount} rows, ${featCount} features) — selected Logistic Regression for calibrated linear probabilities.`;
  } else {
    autoKey = 'gradient_boosting';
    reason = `Rich mixed classification dataset (${rowCount} rows, ${numCount} numeric & ${catCount} categorical features, ${classCount} classes) — selected Gradient Boosting Classifier for strong non-linear accuracy.`;
  }

  const validClassKeys = new Set(SUPERVISED_CLASSIFICATION_CATALOG.map((m) => m.key));
  const chosenKey = (
    manualOverrideModel && validClassKeys.has(manualOverrideModel)
      ? manualOverrideModel
      : autoKey
  ) as SupervisedModelKey;
  const isManualOverride = chosenKey !== autoKey;

  return {
    learningMode: 'supervised',
    taskType: 'classification',
    selectedModelKey: chosenKey,
    autoRecommendedModelKey: autoKey,
    isManualOverride,
    reason: isManualOverride
      ? `Manual override active. Auto-selector originally recommended ${
          SUPERVISED_CLASSIFICATION_CATALOG.find((m) => m.key === autoKey)?.label ?? autoKey
        }: ${reason}`
      : reason,
    signals: [
      `Target: ${targetColumn}`,
      extraSignal,
      `${rowCount} rows`,
      `${numCount} num · ${catCount} cat`,
    ],
  };
}
