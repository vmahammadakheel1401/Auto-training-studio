# AutoML Studio

AutoML Studio is a browser-based machine-learning workspace for exploring CSV data, training a suitable model, and trying predictions. It supports both supervised learning and unsupervised analysis, with a guided workflow from data audit to results and export.

## Features

- Upload a CSV file or start with one of four built-in sample datasets.
- Review column types, missing values, and dataset cleaning details.
- Automatically identify a supervised task (regression or classification) or use unsupervised analysis.
- Train an automatically selected model, or choose another supported model in the results step.
- Explore live predictions, projections, clusters, or association rules, depending on the task.
- Export the trained model state as JSON or download a Python training script. A browser-generated pickle export is also available for supported models.
- Switch between light and dark themes.

## Sample datasets

| Dataset | Learning task |
| --- | --- |
| House Prices | Supervised regression |
| Customer Churn | Supervised classification |
| Customer Segmentation | Unsupervised clustering |
| Market Basket Transactions | Unsupervised association-rule mining |

The House Prices dataset loads automatically when the app starts. Use **Load Sample Data** in the top navigation to switch datasets.

## Run locally

### Requirements

- Node.js
- npm

### Install and start

```bash
npm install
npm run dev
```

Open the local URL printed by Vite (by default, [http://localhost:3000](http://localhost:3000)). If that port is already in use, Vite may choose another available port and show it in the terminal.

## Available scripts

| Command | Description |
| --- | --- |
| `npm run dev` | Start the Vite development server |
| `npm run build` | Create a production build in `dist/` |
| `npm run preview` | Preview the production build locally |
| `npm run lint` | Run the TypeScript compiler without emitting files |

## Typical workflow

1. Start with the default House Prices sample, choose another sample, or upload a `.csv` file in **Upload & Audit**.
2. Review the dataset audit and configure the learning mode, target column, excluded ID columns, or model in **Auto-Model & Results**.
3. Inspect the results, then continue to **Predict & Export** to try what-if inputs or explore the unsupervised output.
4. Download the available JSON model state, Python script, or pickle export.

## Technology

The interface is built with React, TypeScript, and Vite. Data parsing, preprocessing, model training, and predictions for the interactive workflow run in the browser. The generated Python scripts use the Python machine-learning tooling specified in each export.
