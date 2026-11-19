export type ColumnType = 'numeric' | 'categorical' | 'id';
export type LearningMode = 'supervised' | 'unsupervised';
export type SupervisedTaskType = 'regression' | 'classification';
export type UnsupervisedTaskType = 'pca' | 'clustering' | 'association';
export type TaskType = SupervisedTaskType | UnsupervisedTaskType;

export interface ColumnStat {
  name: string;
  inferredType: ColumnType;
  missingCount: number;
  uniqueCount: number;
  min?: number;
  max?: number;
  mean?: number;
  median?: number;
  step?: number;
  mode?: string;
  categories?: string[];
  hasOtherBucket?: boolean;
  sampleValues: string[];
  isLikelyId: boolean;
  idReason?: string;
}

export interface DroppedColumnInfo {
  name: string;
  reason: 'sparse' | 'constant';
  detail: string;
}

export interface CleaningReport {
  originalRowCount: number;
  cleanedRowCount: number;
  sampledRowCount: number;
  wasSampled: boolean;
  originalColumnCount: number;
  cleanedColumnCount: number;
  strippedSubHeaderRows: number;
  subHeaderSampleText?: string;
  sentinelValuesConverted: number;
  sentinelColumns: string[];
  droppedColumns: DroppedColumnInfo[];
  droppedSparseCount: number;
  droppedConstantCount: number;
  cappedCategoricalColumns: string[];
}

export interface ParsedDataset {
  id: string;
  fileName: string;
  label: string;
  headers: string[];
  rows: Record<string, string>[];
  columnStats: ColumnStat[];
  totalRows: number;
  totalColumns: number;
  missingCells: number;
  numericCount: number;
  categoricalCount: number;
  idCount: number;
  defaultTarget: string;
  cleaningReport: CleaningReport;
}

export interface FeatureEncoderSpec {
  rawName: string;
  type: 'numeric' | 'categorical';
  // For numeric
  median?: number;
  mean?: number;
  std?: number;
  min?: number;
  max?: number;
  step?: number;
  // For categorical
  mode?: string;
  categories?: string[];
  hasOtherBucket?: boolean;
  encodedStartIndex: number;
  encodedLength: number;
}

export interface PreprocessorState {
  taskType: SupervisedTaskType;
  targetColumn: string;
  droppedIdColumns: string[];
  featureNames: string[];
  encodedFeatureCount: number;
  specs: FeatureEncoderSpec[];
  encodedColumnToRaw: string[];
  classLabels?: string[];
  targetMin?: number;
  targetMax?: number;
  targetMean?: number;
  trainCount: number;
  testCount: number;
}

export interface UnsupervisedPreprocessorState {
  droppedIdColumns: string[];
  featureNames: string[];
  encodedFeatureCount: number;
  specs: FeatureEncoderSpec[];
  encodedColumnToRaw: string[];
  rowCount: number;
  isTransactional: boolean;
  transactionItemsCatalog: string[];
}

export interface FeatureImportanceItem {
  feature: string;
  importance: number; // 0..100 percentage
}

export interface RegressionPoint {
  actual: number;
  predicted: number;
  rowIndex: number;
}

export interface ConfusionMatrixData {
  labels: string[];
  matrix: number[][]; // [actualIdx][predIdx]
}

export type ExistingSupervisedModelKey =
  | 'ridge'
  | 'logistic'
  | 'decision_tree'
  | 'random_forest'
  | 'gradient_boosting';

export type ExpandedSupervisedModelKey =
  | 'linear_regression'
  | 'lasso'
  | 'svr'
  | 'knn'
  | 'svm'
  | 'naive_bayes'
  | 'lda'
  | 'adaboost';

export type SupervisedModelKey =
  | ExistingSupervisedModelKey
  | ExpandedSupervisedModelKey;

export type UnsupervisedModelKey =
  | 'pca'
  | 'kmeans'
  | 'hierarchical'
  | 'gmm'
  | 'apriori';

export type ModelAlgorithmKey = SupervisedModelKey | UnsupervisedModelKey;

export interface SerializedTreeNode {
  isLeaf: boolean;
  value?: number; // regression prediction or binary log-odds / class prob
  probs?: number[]; // multiclass probabilities
  featureIndex?: number;
  threshold?: number;
  left?: SerializedTreeNode;
  right?: SerializedTreeNode;
}

export type SerializedModelWeights =
  | {
      kind: 'linear';
      weights: number[]; // for regression or binary logistic / SVM
      bias: number;
      multiWeights?: number[][]; // [classIdx][featIdx] for multiclass
      multiBias?: number[];
    }
  | {
      kind: 'tree';
      tree: SerializedTreeNode;
    }
  | {
      kind: 'forest';
      trees: SerializedTreeNode[];
    }
  | {
      kind: 'gbdt';
      learningRate: number;
      baseScore: number;
      trees?: SerializedTreeNode[]; // regression or binary classification
      multiBaseScores?: number[];
      multiTrees?: SerializedTreeNode[][]; // [round][class]
    }
  | {
      kind: 'knn';
      k: number;
      XTrain: number[][];
      yTrain: number[];
    }
  | {
      kind: 'naive_bayes';
      variant: 'gaussian' | 'multinomial';
      priors: number[];
      means?: number[][]; // [class][feat]
      variances?: number[][]; // [class][feat]
      logProbs?: number[][]; // [class][feat] for multinomial
    }
  | {
      kind: 'adaboost';
      numClasses: number;
      stumps: SerializedTreeNode[];
      alphas: number[];
    };

export interface ModelMetrics {
  // Regression
  r2?: number;
  rmse?: number;
  mae?: number;
  // Classification
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1?: number;
}

export interface TrainedModelResult {
  key: SupervisedModelKey;
  name: string;
  sklearnName: string;
  family: string;
  trainingTimeMs: number;
  primaryScore: number; // R2 for regression, Accuracy for classification
  metrics: ModelMetrics;
  featureImportances: FeatureImportanceItem[];
  regressionPoints?: RegressionPoint[];
  confusionMatrix?: ConfusionMatrixData;
  weights: SerializedModelWeights;
  isWinner: boolean;
}

export interface PCAComponentInfo {
  component: string; // e.g., "PC1"
  eigenvalue: number;
  explainedVarianceRatio: number; // 0..1
  cumulativeVarianceRatio: number; // 0..1
  topLoadings: { feature: string; loading: number }[];
}

export interface ProjectionPoint {
  x: number;
  y: number;
  rowIndex: number;
  cluster?: number;
  probability?: number;
  label?: string;
}

export interface ClusterProfile {
  clusterId: number;
  label: string;
  count: number;
  percentage: number;
  centroid: number[];
  topFeatures: { feature: string; meanLabel: string; zScore: number }[];
}

export interface KDiagnosticPoint {
  k: number;
  silhouette: number;
  inertia?: number;
  bic?: number;
}

export interface HierarchicalMergeStep {
  step: number;
  size: number;
  distance: number;
}

export interface FrequentItemset {
  items: string[];
  support: number; // 0..1
  count: number;
}

export interface AssociationRule {
  antecedents: string[];
  consequents: string[];
  support: number; // 0..1
  confidence: number; // 0..1
  lift: number;
  conviction: number;
}

export interface UnsupervisedModelResult {
  key: UnsupervisedModelKey;
  name: string;
  sklearnName: string;
  family: string;
  unsupervisedTask: UnsupervisedTaskType;
  trainingTimeMs: number;
  // PCA
  pcaComponents?: PCAComponentInfo[];
  pcaVectors?: number[][]; // [componentIdx][featIdx]
  pcaMeans?: number[];
  totalVarianceExplained2D?: number;
  // Clustering (KMeans, Hierarchical, GMM)
  numClusters?: number;
  silhouetteScore?: number;
  inertia?: number;
  bic?: number;
  aic?: number;
  logLikelihood?: number;
  clusterProfiles?: ClusterProfile[];
  kDiagnostics?: KDiagnosticPoint[];
  mergeHistory?: HierarchicalMergeStep[];
  gmmWeights?: number[];
  gmmVariances?: number[][];
  // Shared 2D projection points (for PCA & Clustering)
  projectionPoints?: ProjectionPoint[];
  projectionAxisLabels?: [string, string];
  // Apriori
  totalTransactions?: number;
  uniqueItemsCount?: number;
  avgBasketSize?: number;
  minSupportUsed?: number;
  frequentItemsets?: FrequentItemset[];
  associationRules?: AssociationRule[];
}

export interface ModelSelectionInfo {
  learningMode: LearningMode;
  taskType: TaskType;
  selectedModelKey: ModelAlgorithmKey;
  autoRecommendedModelKey: ModelAlgorithmKey;
  isManualOverride: boolean;
  reason: string;
  signals: string[];
}

export interface AutoMLRunResult {
  runId: string;
  learningMode: 'supervised';
  datasetName: string;
  fileName: string;
  taskType: SupervisedTaskType;
  targetColumn: string;
  preprocessor: PreprocessorState;
  models: TrainedModelResult[];
  winner: TrainedModelResult;
  selection: ModelSelectionInfo;
  totalDurationMs: number;
  createdAt: string;
}

export interface UnsupervisedRunResult {
  runId: string;
  learningMode: 'unsupervised';
  datasetName: string;
  fileName: string;
  taskType: UnsupervisedTaskType;
  preprocessor: UnsupervisedPreprocessorState;
  model: UnsupervisedModelResult;
  selection: ModelSelectionInfo;
  totalDurationMs: number;
  createdAt: string;
}

export type AnyRunResult = AutoMLRunResult | UnsupervisedRunResult;

export interface PredictionOutput {
  taskType: SupervisedTaskType;
  rawPrediction: string | number;
  formattedPrediction: string;
  confidence?: number;
  classProbabilities?: { label: string; probability: number }[];
}

export interface UnsupervisedPredictionOutput {
  modelKey: UnsupervisedModelKey;
  // PCA
  projectedCoords?: { component: string; value: number }[];
  // Clustering
  assignedCluster?: number;
  assignedClusterLabel?: string;
  clusterProbabilities?: { clusterId: number; label: string; probability: number }[];
  centroidDistances?: { clusterId: number; label: string; distance: number }[];
}
