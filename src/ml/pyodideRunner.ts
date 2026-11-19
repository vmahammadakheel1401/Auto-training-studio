import { ParsedDataset } from './types';

const PYODIDE_INDEX_URL = 'https://cdn.jsdelivr.net/pyodide/v0.27.2/full/';
const PYODIDE_SCRIPT_URL = `${PYODIDE_INDEX_URL}pyodide.js`;

interface PyodideFS {
  writeFile(path: string, data: string | Uint8Array): void;
  readFile(path: string): Uint8Array;
  unlink(path: string): void;
  analyzePath(path: string): { exists: boolean };
}

interface PyodideInstance {
  loadPackage(names: string | string[]): Promise<void>;
  pyimport(name: string): { install(pkg: string | string[]): Promise<void> };
  runPythonAsync(code: string): Promise<unknown>;
  FS: PyodideFS;
}

declare global {
  interface Window {
    loadPyodide?: (config: { indexURL: string }) => Promise<PyodideInstance>;
  }
}

let scriptPromise: Promise<void> | null = null;
let pyodidePromise: Promise<PyodideInstance> | null = null;
let basePackagesLoaded = false;
let mlxtendInstalled = false;

function loadPyodideScript(): Promise<void> {
  if (typeof window.loadPyodide === 'function') {
    return Promise.resolve();
  }
  if (scriptPromise) {
    return scriptPromise;
  }

  scriptPromise = new Promise<void>((resolve, reject) => {
    const existing = document.querySelector<HTMLScriptElement>(
      `script[src="${PYODIDE_SCRIPT_URL}"]`
    );
    if (existing) {
      if (typeof window.loadPyodide === 'function') {
        resolve();
        return;
      }
      existing.addEventListener('load', () => resolve(), { once: true });
      existing.addEventListener(
        'error',
        () => {
          scriptPromise = null;
          reject(new Error('Failed to load Pyodide runtime script from CDN.'));
        },
        { once: true }
      );
      return;
    }

    const script = document.createElement('script');
    script.src = PYODIDE_SCRIPT_URL;
    script.async = true;
    script.crossOrigin = 'anonymous';
    script.onload = () => resolve();
    script.onerror = () => {
      scriptPromise = null;
      reject(new Error('Failed to load Pyodide runtime script from CDN.'));
    };
    document.head.appendChild(script);
  });

  return scriptPromise;
}

async function getPyodide(
  onStatus: (status: string) => void
): Promise<PyodideInstance> {
  if (!pyodidePromise) {
    onStatus('Initializing Python runtime & building .pkl...');
    pyodidePromise = (async () => {
      await loadPyodideScript();
      if (typeof window.loadPyodide !== 'function') {
        throw new Error('Pyodide loader did not initialize properly.');
      }
      return window.loadPyodide({ indexURL: PYODIDE_INDEX_URL });
    })().catch((err) => {
      pyodidePromise = null;
      throw err;
    });
  } else {
    onStatus('Preparing Python runtime...');
  }

  return pyodidePromise;
}

function escapeCsvCell(value: string | undefined): string {
  const str = value ?? '';
  if (/[",\r\n]/.test(str)) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

export function datasetToCsv(dataset: ParsedDataset): string {
  const headerLine = dataset.headers.map((h) => escapeCsvCell(h)).join(',');
  const rowLines = dataset.rows.map((row) =>
    dataset.headers.map((h) => escapeCsvCell(row[h])).join(',')
  );
  return [headerLine, ...rowLines].join('\n');
}

export async function generatePickleViaPyodide(params: {
  dataset: ParsedDataset;
  fileName: string;
  pythonScript: string;
  needsMlxtend: boolean;
  onStatus: (status: string) => void;
}): Promise<Uint8Array> {
  const { dataset, fileName, pythonScript, needsMlxtend, onStatus } = params;

  const pyodide = await getPyodide(onStatus);

  if (!basePackagesLoaded || (needsMlxtend && !mlxtendInstalled)) {
    const pkgsToLoad = ['pandas', 'scikit-learn'];
    if (needsMlxtend) {
      pkgsToLoad.push('micropip');
    }
    onStatus(
      needsMlxtend
        ? 'Loading pandas, scikit-learn & micropip...'
        : 'Loading pandas & scikit-learn...'
    );
    await pyodide.loadPackage(pkgsToLoad);
    basePackagesLoaded = true;
  }

  if (needsMlxtend && !mlxtendInstalled) {
    onStatus('Installing mlxtend for Apriori...');
    const micropip = pyodide.pyimport('micropip');
    await micropip.install('mlxtend');
    mlxtendInstalled = true;
  }

  onStatus('Executing script & saving trained_model.pkl...');

  const safeRemove = (path: string) => {
    try {
      if (pyodide.FS.analyzePath(path).exists) {
        pyodide.FS.unlink(path);
      }
    } catch {
      // Ignore cleanup errors
    }
  };

  safeRemove(fileName);
  safeRemove('trained_model.pkl');

  const csvText = datasetToCsv(dataset);
  pyodide.FS.writeFile(fileName, csvText);

  try {
    await pyodide.runPythonAsync(pythonScript);
    if (!pyodide.FS.analyzePath('trained_model.pkl').exists) {
      throw new Error('Python execution completed without producing trained_model.pkl.');
    }
    const rawBytes = pyodide.FS.readFile('trained_model.pkl');
    return new Uint8Array(rawBytes);
  } finally {
    safeRemove(fileName);
    safeRemove('trained_model.pkl');
  }
}
