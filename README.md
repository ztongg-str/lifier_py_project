# OOP Triage Prediction

This workspace contains two implementations of a triage prediction pipeline:

- `triage_prediction.ipynb`: a clean, non-OOP notebook with a full pipeline.
- `oop_triage.ipynb`: an OOP-oriented notebook and supporting modules under `modules/`.

Quick start

1. Create and activate a Python environment (recommended Python 3.8+).
2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the OOP notebook in Jupyter or execute `oop_triage.ipynb` cells.

Files added in this change:

- `modules/oop_bases.py` — abstract base classes
- `modules/preprocessors_oop.py` — numerical, categorical, text preprocessors
- `modules/feature_builder.py` — feature composition and ESI word scoring
- `modules/oop_models.py` — model wrappers (Logistic, RF, SVM, XGBoost)
- `modules/oop_trainer.py` — `TriagePipeline` orchestrator with dunder methods
- `modules/oop_evaluator.py` — evaluator + visualizer for OOP pipeline
- `oop_triage.ipynb` — minimal notebook demonstrating the OOP pipeline

Notes

- The OOP modules were implemented to be importable from the project root (use `import modules.<name>`).
- Some of the existing non-OOP modules in `modules/` still import `modules.config`; if you run notebooks from different working directories, ensure project root is on `sys.path`.
