# GMM for Incomplete Data

This repository implements a Gaussian Mixture Model (GMM) workflow for clustering and imputing incomplete data. It combines:

- a Python implementation of the alternating GMM-imputation algorithm;
- reproducible notebooks for step-by-step explanation, reproduction, and application;
- a modular LaTeX report with figures, experiments, and case studies.

The project is centered on the paper *Gaussian Mixture Model Clustering with Incomplete Data* and extends it with local implementation details, reproduction experiments, and application-driven artifacts.

## Repository Layout

```text
GMM_4_missing_data/
├── data/
│   └── CC GENERAL.csv
├── docs/
│   └── Gaussian Mixture Model Clustering with Incomplete Data.pdf
├── latex_composition/
│   ├── main.tex
│   ├── main.pdf
│   ├── figures/
│   └── sections/
├── notebooks/
│   ├── step_by_step_experiment.ipynb
│   ├── GMM_Nhat.ipynb
│   ├── airplane_crashes_gmm_imputation.ipynb
│   └── artifacts/
├── src/
│   ├── gmm_missing.py
│   ├── utils.py
│   └── airplane_crashes_pipeline.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Main Components

### Core implementation

- `src/gmm_missing.py`: main `GMMMissing` estimator.
- `src/utils.py`: synthetic data generation and basic imputers.
- `src/airplane_crashes_pipeline.py`: end-to-end application pipeline for the airplane crashes case study.

### Notebooks

- `notebooks/step_by_step_experiment.ipynb`: pedagogical walkthrough of the EM-style update cycle on synthetic data.
- `notebooks/GMM_Nhat.ipynb`: reproduction experiment on the `CC GENERAL` dataset.
- `notebooks/airplane_crashes_gmm_imputation.ipynb`: real-data application with generated figures and RMSE-based imputation evaluation.

### Report

- `latex_composition/main.tex`: source of the report.
- `latex_composition/main.pdf`: compiled paper.
- `latex_composition/figures/`: figures used in the report, including notebook-exported artifacts.

## Installation

The project targets Python `>=3.12`.

Using `uv`:

```bash
uv sync
```

Using `pip`:

```bash
pip install -r requirements.txt
```

## Quick Start

Run the notebooks:

```bash
uv run jupyter lab
```

Recommended order:

1. `notebooks/step_by_step_experiment.ipynb`
2. `notebooks/GMM_Nhat.ipynb`
3. `notebooks/airplane_crashes_gmm_imputation.ipynb`

Compile the LaTeX report:

```bash
cd latex_composition
latexmk -pdf -interaction=nonstopmode main.tex
```

## Reproducibility Scope

This repository currently includes three complementary layers of evidence:

- algorithm-level inspection through the step-by-step notebook;
- reproduction on `CC GENERAL`;
- application-level evaluation on the airplane crashes dataset, with exported figures and RMSE on masked entries.

Artifacts generated in notebooks are intended to be promoted into `latex_composition/figures/` so that the report remains tied to actual outputs rather than illustrative placeholders.

## Outputs

Important outputs already tracked in the project include:

- `latex_composition/main.pdf`
- figures under `latex_composition/figures/applications/`
- figures under `latex_composition/figures/notebook_exports/`

## Notes

- The LaTeX report uses a custom style file at `latex_composition/setting/khang_paper.sty`.
- The current font setup keeps the original `lmodern` and `microtype` configuration, which may produce non-blocking T5-related warnings during compilation.
