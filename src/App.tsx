import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Check, ChevronDown, Cpu, Moon, Sun } from 'lucide-react';
import { StepModelOutput } from './components/StepModelOutput';
import { StepPredictExport } from './components/StepPredictExport';
import { StepUploadAudit } from './components/StepUploadAudit';
import { trainSingleAlgorithm } from './ml/algorithms';
import { buildDatasetFromCSV } from './ml/csvParser';
import {
  inferInitialLearningMode,
  inspectAndSelectBestModel,
} from './ml/modelSelector';
import { prepareTrainingData } from './ml/preprocessor';
import {
  loadSampleDataset,
  SAMPLE_DATASET_META,
  SampleDatasetKey,
} from './ml/sampleDatasets';
import {
  AnyRunResult,
  LearningMode,
  ModelAlgorithmKey,
  ParsedDataset,
  SupervisedModelKey,
  UnsupervisedModelKey,
} from './ml/types';
import {
  prepareUnsupervisedData,
  trainUnsupervisedAlgorithm,
} from './ml/unsupervised';

type WorkflowStep = 1 | 2 | 3;

const STEPS: { step: WorkflowStep; label: string }[] = [
  { step: 1, label: '1. Upload & Audit' },
  { step: 2, label: '2. Auto-Model & Results' },
  { step: 3, label: '3. Predict & Export' },
];

const SAMPLE_KEYS: SampleDatasetKey[] = [
  'house_prices',
  'customer_churn',
  'customer_segmentation',
  'market_basket',
];

export default function App() {
  const initialDataset = useMemo(() => loadSampleDataset('house_prices'), []);
  const [isDarkMode, setIsDarkMode] = useState(() => {
    const savedTheme = window.localStorage.getItem('automl-theme');
    return savedTheme
      ? savedTheme === 'dark'
      : window.matchMedia('(prefers-color-scheme: dark)').matches;
  });

  useEffect(() => {
    document.documentElement.classList.toggle('dark', isDarkMode);
    window.localStorage.setItem('automl-theme', isDarkMode ? 'dark' : 'light');
  }, [isDarkMode]);

  const [activeStep, setActiveStep] = useState<WorkflowStep>(1);
  const [dataset, setDataset] = useState<ParsedDataset | null>(initialDataset);
  const [activeSampleKey, setActiveSampleKey] = useState<SampleDatasetKey | null>(
    'house_prices'
  );
  const [parseError, setParseError] = useState<string | null>(null);

  const [learningMode, setLearningMode] = useState<LearningMode>('supervised');
  const [targetColumn, setTargetColumn] = useState<string>(
    initialDataset.defaultTarget
  );
  const [droppedIdColumns, setDroppedIdColumns] = useState<string[]>(() =>
    initialDataset.columnStats
      .filter((c) => c.isLikelyId && c.name !== initialDataset.defaultTarget)
      .map((c) => c.name)
  );
  const [manualOverrideModel, setManualOverrideModel] =
    useState<ModelAlgorithmKey | null>(null);

  const [isTraining, setIsTraining] = useState(false);
  const [trainError, setTrainError] = useState<string | null>(null);
  const [runResult, setRunResult] = useState<AnyRunResult | null>(null);

  const [sampleMenuOpen, setSampleMenuOpen] = useState(false);
  const sampleMenuRef = useRef<HTMLDivElement>(null);
  const runSeqRef = useRef(0);

  const executeSingleBestModel = useCallback(
    async (params: {
      ds: ParsedDataset;
      mode: LearningMode;
      target: string;
      droppedIds: string[];
      overrideModel: ModelAlgorithmKey | null;
    }) => {
      const { ds, mode, target, droppedIds, overrideModel } = params;
      const seq = ++runSeqRef.current;
      setIsTraining(true);
      setTrainError(null);
      const overallStart = performance.now();

      try {
        const selection = inspectAndSelectBestModel({
          dataset: ds,
          learningMode: mode,
          targetColumn: target,
          droppedIdColumns: droppedIds,
          manualOverrideModel: overrideModel,
        });

        if (selection.learningMode === 'unsupervised') {
          const cleanIds = droppedIds;
          const prepared = prepareUnsupervisedData(ds, cleanIds);
          const modelRes = await trainUnsupervisedAlgorithm(
            selection.selectedModelKey as UnsupervisedModelKey,
            prepared
          );
          if (seq !== runSeqRef.current) return;

          const totalDurationMs = Math.max(
            1,
            Math.round(performance.now() - overallStart)
          );

          setRunResult({
            runId: `run-${Date.now()}`,
            learningMode: 'unsupervised',
            datasetName: ds.label,
            fileName: ds.fileName,
            taskType: modelRes.unsupervisedTask,
            preprocessor: prepared.preprocessor,
            model: modelRes,
            selection,
            totalDurationMs,
            createdAt: new Date().toISOString(),
          });
        } else {
          const cleanIds = droppedIds.filter((c) => c !== target);
          const supervisedTask =
            selection.taskType === 'classification' ? 'classification' : 'regression';
          const prepared = prepareTrainingData(ds, target, supervisedTask, cleanIds);
          const trained = await trainSingleAlgorithm(
            selection.selectedModelKey as SupervisedModelKey,
            prepared
          );
          if (seq !== runSeqRef.current) return;

          const winner = { ...trained, isWinner: true };
          const totalDurationMs = Math.max(
            1,
            Math.round(performance.now() - overallStart)
          );

          setRunResult({
            runId: `run-${Date.now()}`,
            learningMode: 'supervised',
            datasetName: ds.label,
            fileName: ds.fileName,
            taskType: supervisedTask,
            targetColumn: target,
            preprocessor: prepared.preprocessor,
            models: [winner],
            winner,
            selection,
            totalDurationMs,
            createdAt: new Date().toISOString(),
          });
        }
      } catch (err) {
        if (seq !== runSeqRef.current) return;
        setRunResult(null);
        setTrainError(
          err instanceof Error ? err.message : 'Model execution failed unexpectedly.'
        );
      } finally {
        if (seq === runSeqRef.current) {
          setIsTraining(false);
        }
      }
    },
    []
  );

  // Train initial dataset automatically on mount
  useEffect(() => {
    void executeSingleBestModel({
      ds: initialDataset,
      mode: 'supervised',
      target: initialDataset.defaultTarget,
      droppedIds: initialDataset.columnStats
        .filter((c) => c.isLikelyId && c.name !== initialDataset.defaultTarget)
        .map((c) => c.name),
      overrideModel: null,
    });
  }, [executeSingleBestModel, initialDataset]);

  const applyNewDataset = useCallback(
    (
      nextDataset: ParsedDataset,
      sampleKey: SampleDatasetKey | null,
      explicitMode?: LearningMode,
      jumpToResults = true
    ) => {
      const nextMode = inferInitialLearningMode(nextDataset, explicitMode);
      const nextTarget = nextDataset.defaultTarget;
      const autoDrops = nextDataset.columnStats
        .filter(
          (c) =>
            c.isLikelyId && (nextMode === 'unsupervised' || c.name !== nextTarget)
        )
        .map((c) => c.name);

      setDataset(nextDataset);
      setActiveSampleKey(sampleKey);
      setParseError(null);
      setLearningMode(nextMode);
      setTargetColumn(nextTarget);
      setDroppedIdColumns(autoDrops);
      setManualOverrideModel(null);

      if (jumpToResults) {
        setActiveStep(2);
      }

      void executeSingleBestModel({
        ds: nextDataset,
        mode: nextMode,
        target: nextTarget,
        droppedIds: autoDrops,
        overrideModel: null,
      });
    },
    [executeSingleBestModel]
  );

  const handleSelectSample = useCallback(
    (key: SampleDatasetKey) => {
      const meta = SAMPLE_DATASET_META[key];
      const loaded = loadSampleDataset(key);
      applyNewDataset(loaded, key, meta.learningMode, true);
      setSampleMenuOpen(false);
    },
    [applyNewDataset]
  );

  const handleUploadFile = useCallback(
    (file: File) => {
      if (!file.name.toLowerCase().endsWith('.csv') && file.type !== 'text/csv') {
        setParseError('Please upload a valid .csv file.');
        return;
      }
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const text = String(reader.result ?? '');
          const parsed = buildDatasetFromCSV(text, file.name);
          applyNewDataset(parsed, null, undefined, true);
        } catch (err) {
          setParseError(
            err instanceof Error ? err.message : 'Unable to parse CSV file.'
          );
        }
      };
      reader.onerror = () => {
        setParseError('Failed to read the selected CSV file.');
      };
      reader.readAsText(file);
    },
    [applyNewDataset]
  );

  const handleChangeLearningMode = useCallback(
    (nextMode: LearningMode) => {
      if (!dataset || nextMode === learningMode) return;
      const nextDrops = dataset.columnStats
        .filter(
          (c) =>
            c.isLikelyId &&
            (nextMode === 'unsupervised' || c.name !== targetColumn)
        )
        .map((c) => c.name);

      setLearningMode(nextMode);
      setDroppedIdColumns(nextDrops);
      setManualOverrideModel(null);

      void executeSingleBestModel({
        ds: dataset,
        mode: nextMode,
        target: targetColumn,
        droppedIds: nextDrops,
        overrideModel: null,
      });
    },
    [dataset, learningMode, targetColumn, executeSingleBestModel]
  );

  const handleChangeTarget = useCallback(
    (nextTarget: string) => {
      if (!dataset) return;
      const nextDrops = droppedIdColumns.filter((c) => c !== nextTarget);
      setTargetColumn(nextTarget);
      setDroppedIdColumns(nextDrops);
      setManualOverrideModel(null);

      void executeSingleBestModel({
        ds: dataset,
        mode: learningMode,
        target: nextTarget,
        droppedIds: nextDrops,
        overrideModel: null,
      });
    },
    [dataset, droppedIdColumns, learningMode, executeSingleBestModel]
  );

  const handleToggleDropColumn = useCallback(
    (colName: string) => {
      if (!dataset) return;
      const nextDrops = droppedIdColumns.includes(colName)
        ? droppedIdColumns.filter((c) => c !== colName)
        : [...droppedIdColumns, colName];
      setDroppedIdColumns(nextDrops);

      void executeSingleBestModel({
        ds: dataset,
        mode: learningMode,
        target: targetColumn,
        droppedIds: nextDrops,
        overrideModel: manualOverrideModel,
      });
    },
    [
      dataset,
      droppedIdColumns,
      learningMode,
      targetColumn,
      manualOverrideModel,
      executeSingleBestModel,
    ]
  );

  const handleSelectModelOverride = useCallback(
    (key: ModelAlgorithmKey | null) => {
      if (!dataset) return;
      setManualOverrideModel(key);
      void executeSingleBestModel({
        ds: dataset,
        mode: learningMode,
        target: targetColumn,
        droppedIds: droppedIdColumns,
        overrideModel: key,
      });
    },
    [dataset, learningMode, targetColumn, droppedIdColumns, executeSingleBestModel]
  );

  const canAccessStep = (step: WorkflowStep): boolean => {
    if (step === 1) return true;
    if (step === 2) return dataset !== null;
    if (step === 3) return runResult !== null;
    return false;
  };

  const autoSelectedModelName = runResult
    ? runResult.learningMode === 'supervised'
      ? runResult.winner.name
      : runResult.model.name
    : undefined;

  return (
    <div className="app-shell min-h-screen flex flex-col">
      {/* Top Navigation Bar */}
      <header className="sticky top-0 z-30 bg-white/95 backdrop-blur-xs border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between gap-6">
          <div className="flex items-center gap-3 min-w-0">
            <div className="w-8 h-8 rounded-lg bg-indigo-600 text-white flex items-center justify-center shrink-0">
              <Cpu className="w-4 h-4" />
            </div>
            <div className="flex items-center gap-3 min-w-0">
              <span className="text-base sm:text-lg font-bold tracking-tight text-slate-900 whitespace-nowrap shrink-0">
                AutoML Studio
              </span>
              <span className="hidden md:inline-flex items-center px-2.5 py-0.5 rounded-md border border-slate-200 bg-slate-50 text-xs font-medium text-slate-600 whitespace-nowrap shrink-0">
                Supervised + Unsupervised · Auto-Select
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <button
              type="button"
              onClick={() => setIsDarkMode((dark) => !dark)}
              aria-pressed={isDarkMode}
              aria-label="Toggle dark mode"
              className="inline-flex items-center gap-2 px-3 py-2 rounded-lg border border-slate-200 bg-white hover:bg-slate-50 text-xs sm:text-sm font-medium text-slate-800 transition-colors whitespace-nowrap cursor-pointer"
            >
              {isDarkMode ? (
                <Sun className="w-4 h-4 text-amber-500" />
              ) : (
                <Moon className="w-4 h-4 text-slate-500" />
              )}
              <span className="hidden sm:inline">Dark mode</span>
            </button>

            <div className="relative shrink-0" ref={sampleMenuRef}>
              <button
                type="button"
                disabled={isTraining}
                onClick={() => setSampleMenuOpen((o) => !o)}
                className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg border border-slate-200 bg-white hover:bg-slate-50 text-xs sm:text-sm font-medium text-slate-800 transition-colors whitespace-nowrap shrink-0 disabled:opacity-50 cursor-pointer"
              >
                <span>Load Sample Data</span>
                <ChevronDown className="w-4 h-4 text-slate-500" />
              </button>

              {sampleMenuOpen && (
                <>
                  <div
                    className="fixed inset-0 z-10"
                    onClick={() => setSampleMenuOpen(false)}
                  />
                  <div className="absolute right-0 mt-2 w-80 rounded-xl border border-slate-200 bg-white shadow-lg py-1.5 z-20">
                    <div className="px-3.5 py-1.5 text-[11px] font-semibold text-slate-400">
                      Sample Datasets (Supervised & Unsupervised)
                    </div>
                    {SAMPLE_KEYS.map((key) => {
                      const meta = SAMPLE_DATASET_META[key];
                      const isCurrent = activeSampleKey === key;
                      return (
                        <button
                          key={key}
                          type="button"
                          onClick={() => handleSelectSample(key)}
                          className="w-full text-left px-3.5 py-2.5 hover:bg-slate-50 flex items-center justify-between gap-3 transition-colors cursor-pointer"
                        >
                          <div>
                            <div className="text-xs font-semibold text-slate-900">
                              {meta.dropdownLabel}
                            </div>
                            <div className="text-[11px] text-slate-500 font-mono tabular-nums mt-0.5">
                              {meta.rowCountHint} rows ·{' '}
                              {meta.learningMode === 'unsupervised'
                                ? 'No Target (Unlabeled)'
                                : `Target: ${meta.targetColumn}`}
                            </div>
                          </div>
                          {isCurrent && (
                            <Check className="w-4 h-4 text-indigo-600 shrink-0" />
                          )}
                        </button>
                      );
                    })}
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Workspace Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-6 space-y-6">
        {/* Horizontal 3-Step Workflow Stepper */}
        <nav
          aria-label="AutoML workflow steps"
          className="bg-white border border-slate-200 rounded-xl p-1.5 flex items-center gap-1.5 overflow-x-auto"
        >
          {STEPS.map((item) => {
            const enabled = canAccessStep(item.step);
            const isActive = activeStep === item.step;
            const isCompleted =
              (item.step === 1 && dataset !== null && activeStep > 1) ||
              (item.step === 2 && runResult !== null && activeStep > 2);

            return (
              <button
                key={item.step}
                type="button"
                disabled={!enabled}
                onClick={() => {
                  if (enabled) setActiveStep(item.step);
                }}
                className={`flex-1 min-w-[180px] inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-xs sm:text-sm font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer disabled:cursor-not-allowed ${
                  isActive
                    ? 'bg-indigo-600 text-white'
                    : enabled
                    ? 'text-slate-700 hover:bg-slate-100 hover:text-slate-900'
                    : 'text-slate-400 opacity-60'
                }`}
              >
                {isCompleted && !isActive && (
                  <Check className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
                )}
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        {activeStep === 1 && (
          <StepUploadAudit
            dataset={dataset}
            activeSampleKey={activeSampleKey}
            learningMode={learningMode}
            parseError={parseError}
            autoSelectedModelName={autoSelectedModelName}
            onUploadFile={handleUploadFile}
            onSelectSample={handleSelectSample}
            onContinue={() => setActiveStep(2)}
          />
        )}

        {activeStep === 2 && dataset && (
          <StepModelOutput
            dataset={dataset}
            learningMode={learningMode}
            targetColumn={targetColumn}
            droppedIdColumns={droppedIdColumns}
            manualOverrideModel={manualOverrideModel}
            isTraining={isTraining}
            trainError={trainError}
            run={runResult}
            onChangeLearningMode={handleChangeLearningMode}
            onChangeTarget={handleChangeTarget}
            onToggleDropColumn={handleToggleDropColumn}
            onSelectModelOverride={handleSelectModelOverride}
            onProceedToPredict={() => setActiveStep(3)}
          />
        )}

        {activeStep === 3 && runResult && <StepPredictExport run={runResult} />}
      </main>
    </div>
  );
}
