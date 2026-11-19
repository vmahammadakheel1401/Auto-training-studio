import { AnyRunResult, AutoMLRunResult, UnsupervisedRunResult } from './types';

function formatPyList(arr: string[]): string {
  return arr.length === 0
    ? '[]'
    : `[\n    ${arr.map((s) => JSON.stringify(s)).join(',\n    ')}\n]`;
}

function generateSupervisedPythonScript(run: AutoMLRunResult): string {
  const { preprocessor, winner, fileName, targetColumn, taskType } = run;

  const numericFeatures = preprocessor.specs
    .filter((s) => s.type === 'numeric')
    .map((s) => s.rawName);

  const categoricalFeatures = preprocessor.specs
    .filter((s) => s.type === 'categorical')
    .map((s) => s.rawName);

  const droppedIds = preprocessor.droppedIdColumns;

  const sklearnModelImport = (() => {
    switch (winner.sklearnName) {
      case 'LinearRegression':
        return 'from sklearn.linear_model import LinearRegression';
      case 'Ridge':
        return 'from sklearn.linear_model import Ridge';
      case 'Lasso':
        return 'from sklearn.linear_model import Lasso';
      case 'SVR':
        return 'from sklearn.svm import SVR';
      case 'SVC':
        return 'from sklearn.svm import SVC';
      case 'KNeighborsRegressor':
        return 'from sklearn.neighbors import KNeighborsRegressor';
      case 'KNeighborsClassifier':
        return 'from sklearn.neighbors import KNeighborsClassifier';
      case 'GaussianNB':
        return 'from sklearn.naive_bayes import GaussianNB';
      case 'MultinomialNB':
        return 'from sklearn.naive_bayes import MultinomialNB';
      case 'LinearDiscriminantAnalysis':
        return 'from sklearn.discriminant_analysis import LinearDiscriminantAnalysis';
      case 'LogisticRegression':
        return 'from sklearn.linear_model import LogisticRegression';
      case 'DecisionTreeRegressor':
        return 'from sklearn.tree import DecisionTreeRegressor';
      case 'DecisionTreeClassifier':
        return 'from sklearn.tree import DecisionTreeClassifier';
      case 'RandomForestRegressor':
        return 'from sklearn.ensemble import RandomForestRegressor';
      case 'RandomForestClassifier':
        return 'from sklearn.ensemble import RandomForestClassifier';
      case 'AdaBoostClassifier':
        return 'from sklearn.ensemble import AdaBoostClassifier';
      case 'GradientBoostingRegressor':
        return 'from sklearn.ensemble import GradientBoostingRegressor';
      case 'GradientBoostingClassifier':
      default:
        return 'from sklearn.ensemble import GradientBoostingClassifier';
    }
  })();

  const modelInitExpr = (() => {
    switch (winner.sklearnName) {
      case 'LinearRegression':
        return 'LinearRegression()';
      case 'Ridge':
        return 'Ridge(alpha=1.5, random_state=42)';
      case 'Lasso':
        return 'Lasso(alpha=0.05, max_iter=2000, random_state=42)';
      case 'SVR':
        return 'SVR(kernel="linear", C=1.0, epsilon=0.1)';
      case 'SVC':
        return 'SVC(kernel="linear", C=1.0, probability=True, random_state=42)';
      case 'KNeighborsRegressor':
        return 'KNeighborsRegressor(n_neighbors=5, weights="distance")';
      case 'KNeighborsClassifier':
        return 'KNeighborsClassifier(n_neighbors=5, weights="distance")';
      case 'GaussianNB':
        return 'GaussianNB()';
      case 'MultinomialNB':
        return 'MultinomialNB(alpha=1.0)';
      case 'LinearDiscriminantAnalysis':
        return 'LinearDiscriminantAnalysis()';
      case 'LogisticRegression':
        return 'LogisticRegression(max_iter=1000, C=1.0, random_state=42)';
      case 'DecisionTreeRegressor':
        return 'DecisionTreeRegressor(max_depth=6, min_samples_leaf=2, random_state=42)';
      case 'DecisionTreeClassifier':
        return 'DecisionTreeClassifier(max_depth=6, min_samples_leaf=2, random_state=42)';
      case 'RandomForestRegressor':
        return 'RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)';
      case 'RandomForestClassifier':
        return 'RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)';
      case 'AdaBoostClassifier':
        return 'AdaBoostClassifier(n_estimators=30, learning_rate=0.8, random_state=42)';
      case 'GradientBoostingRegressor':
        return 'GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)';
      case 'GradientBoostingClassifier':
      default:
        return 'GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)';
    }
  })();

  const metricsImport =
    taskType === 'regression'
      ? 'from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error'
      : 'from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report';

  const stratifyLine = taskType === 'classification' ? '    stratify=y,\n' : '';
  const denseNote =
    winner.sklearnName === 'GaussianNB' || winner.sklearnName === 'LinearDiscriminantAnalysis'
      ? 'OneHotEncoder(handle_unknown="ignore", sparse_output=False)'
      : 'OneHotEncoder(handle_unknown="ignore")';

  const evalBlock =
    taskType === 'regression'
      ? `y_pred = pipeline.predict(X_test)
print(f"R2 Score : {r2_score(y_test, y_pred):.4f}")
print(f"RMSE     : {root_mean_squared_error(y_test, y_pred):.2f}")
print(f"MAE      : {mean_absolute_error(y_test, y_pred):.2f}")`
      : `y_pred = pipeline.predict(X_test)
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
print("\\nClassification Report:\\n", classification_report(y_test, y_pred))`;

  return `"""
AutoML Studio — Generated Scikit-Learn Pipeline (Supervised)
Dataset: ${fileName}
Task: ${taskType === 'regression' ? 'Regression' : 'Classification'}
Target Column: ${targetColumn}
Auto-Selected Model: ${winner.name} (${winner.sklearnName})
"""

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
${sklearnModelImport}
${metricsImport}

# 1. Load dataset and drop missing targets / identifier columns
df = pd.read_csv(${JSON.stringify(fileName)})
TARGET_COL = ${JSON.stringify(targetColumn)}
DROPPED_ID_COLS = ${formatPyList(droppedIds)}

df = df.dropna(subset=[TARGET_COL])
X = df.drop(columns=[c for c in [TARGET_COL, *DROPPED_ID_COLS] if c in df.columns])
y = df[TARGET_COL]

# 2. Define feature schema
NUMERIC_FEATURES = ${formatPyList(numericFeatures)}
CATEGORICAL_FEATURES = ${formatPyList(categoricalFeatures)}

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", ${denseNote}),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ]
)

# 3. Build end-to-end training pipeline (${winner.name})
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", ${modelInitExpr}),
    ]
)

# 4. 80/20 Train/Test Split & Fit
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
${stratifyLine})

pipeline.fit(X_train, y_train)

# 5. Evaluate on holdout test split
${evalBlock}

# 6. Persist trained pipeline for production inference
joblib.dump(pipeline, "automl_selected_pipeline.joblib")
print("Saved trained pipeline to automl_selected_pipeline.joblib")
`;
}

function generateUnsupervisedPythonScript(run: UnsupervisedRunResult): string {
  const { preprocessor, model, fileName } = run;
  const droppedIds = preprocessor.droppedIdColumns;

  if (model.key === 'apriori') {
    const minSup = model.minSupportUsed ?? 0.1;
    return `"""
AutoML Studio — Generated Unsupervised Association Rule Mining Script
Dataset: ${fileName}
Task: Association Rule Mining (Apriori)
Auto-Selected Model: ${model.name}
Requires: pip install pandas mlxtend
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv(${JSON.stringify(fileName)})
DROPPED_ID_COLS = ${formatPyList(droppedIds)}
df_features = df.drop(columns=[c for c in DROPPED_ID_COLS if c in df.columns])

# Convert binary/boolean indicators or categorical columns into one-hot basket matrix
basket = pd.get_dummies(df_features, prefix_sep="=").astype(bool)

frequent_itemsets = apriori(basket, min_support=${minSup}, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.05)
rules = rules.sort_values(by=["lift", "confidence"], ascending=False)

print(f"Discovered {len(frequent_itemsets)} frequent itemsets and {len(rules)} rules.")
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(15))
`;
  }

  const numericFeatures = preprocessor.specs
    .filter((s) => s.type === 'numeric')
    .map((s) => s.rawName);
  const categoricalFeatures = preprocessor.specs
    .filter((s) => s.type === 'categorical')
    .map((s) => s.rawName);

  const k = model.numClusters ?? 3;
  const nComp = Math.min(6, model.pcaComponents?.length ?? 2);

  const modelImport =
    model.key === 'pca'
      ? 'from sklearn.decomposition import PCA'
      : model.key === 'kmeans'
      ? 'from sklearn.cluster import KMeans\nfrom sklearn.metrics import silhouette_score'
      : model.key === 'hierarchical'
      ? 'from sklearn.cluster import AgglomerativeClustering\nfrom sklearn.metrics import silhouette_score'
      : 'from sklearn.mixture import GaussianMixture\nfrom sklearn.metrics import silhouette_score';

  const modelBlock =
    model.key === 'pca'
      ? `model = PCA(n_components=${nComp}, random_state=42)
X_pca = model.fit_transform(X_processed)
print("Explained variance ratio:", model.explained_variance_ratio_)
print("Cumulative variance:", model.explained_variance_ratio_.cumsum())`
      : model.key === 'kmeans'
      ? `model = KMeans(n_clusters=${k}, n_init="auto", random_state=42)
labels = model.fit_predict(X_processed)
print(f"Silhouette Score (K=${k}): {silhouette_score(X_processed, labels):.4f}")
df["cluster"] = labels`
      : model.key === 'hierarchical'
      ? `model = AgglomerativeClustering(n_clusters=${k}, linkage="ward")
labels = model.fit_predict(X_processed)
print(f"Silhouette Score (K=${k}): {silhouette_score(X_processed, labels):.4f}")
df["cluster"] = labels`
      : `model = GaussianMixture(n_components=${k}, covariance_type="diag", random_state=42)
labels = model.fit_predict(X_processed)
print(f"BIC: {model.bic(X_processed):.1f} | Silhouette: {silhouette_score(X_processed, labels):.4f}")
df["cluster"] = labels`;

  return `"""
AutoML Studio — Generated Unsupervised Learning Script
Dataset: ${fileName}
Task: ${model.family}
Auto-Selected Model: ${model.name} (${model.sklearnName})
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
${modelImport}

df = pd.read_csv(${JSON.stringify(fileName)})
DROPPED_ID_COLS = ${formatPyList(droppedIds)}
X = df.drop(columns=[c for c in DROPPED_ID_COLS if c in df.columns])

NUMERIC_FEATURES = ${formatPyList(numericFeatures)}
CATEGORICAL_FEATURES = ${formatPyList(categoricalFeatures)}

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUMERIC_FEATURES),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL_FEATURES),
    ]
)

X_processed = preprocessor.fit_transform(X)

${modelBlock}
`;
}

export function generatePythonScript(run: AnyRunResult): string {
  if (run.learningMode === 'unsupervised') {
    return generateUnsupervisedPythonScript(run);
  }
  return generateSupervisedPythonScript(run);
}

export function generateModelStateJson(run: AnyRunResult): string {
  if (run.learningMode === 'unsupervised') {
    const artifact = {
      schemaVersion: '2.0.0',
      generatedBy: 'AutoML Studio (Client-Side Engine)',
      createdAt: run.createdAt,
      learningMode: 'unsupervised',
      selection: run.selection,
      dataset: {
        name: run.datasetName,
        fileName: run.fileName,
        taskType: run.taskType,
        rows: run.preprocessor.rowCount,
        droppedIdColumns: run.preprocessor.droppedIdColumns,
      },
      model: run.model,
      preprocessing: {
        encodedFeatureCount: run.preprocessor.encodedFeatureCount,
        isTransactional: run.preprocessor.isTransactional,
        features: run.preprocessor.specs,
      },
    };
    return JSON.stringify(artifact, null, 2);
  }

  const artifact = {
    schemaVersion: '2.0.0',
    generatedBy: 'AutoML Studio (Client-Side Engine)',
    createdAt: run.createdAt,
    learningMode: 'supervised',
    selection: run.selection,
    dataset: {
      name: run.datasetName,
      fileName: run.fileName,
      taskType: run.taskType,
      targetColumn: run.targetColumn,
      trainRows: run.preprocessor.trainCount,
      testRows: run.preprocessor.testCount,
      droppedIdColumns: run.preprocessor.droppedIdColumns,
    },
    selectedModel: {
      key: run.winner.key,
      name: run.winner.name,
      sklearnEquivalent: run.winner.sklearnName,
      primaryScore: run.winner.primaryScore,
      metrics: run.winner.metrics,
      trainingTimeMs: run.winner.trainingTimeMs,
      featureImportances: run.winner.featureImportances,
      weights: run.winner.weights,
    },
    preprocessing: {
      encodedFeatureCount: run.preprocessor.encodedFeatureCount,
      classLabels: run.preprocessor.classLabels ?? null,
      features: run.preprocessor.specs,
    },
  };

  return JSON.stringify(artifact, null, 2);
}
