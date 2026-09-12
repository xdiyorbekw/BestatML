# Contributing to BestatML

Thanks for contributing to BestatML.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Checks

```bash
pytest
ruff check .
python -m build
twine check dist/*
```

## Code expectations

Keep changes modular and compatible with sklearn conventions. Preserve leakage-safe preprocessing and OOF invariants. Avoid broad exception swallowing, unnecessary public API expansion, unsupported feature claims, and hidden changes to fitted state.

New behavior should include focused tests, especially when it affects estimator state, validation, preprocessing, ensemble construction, or persistence.

## Pull requests

Explain the behavior change, include tests, and keep README/documentation claims consistent with the implementation. Do not commit build artifacts, caches, coverage output, or local model artifacts.

## Bug reports

Include the BestatML version, Python version, scikit-learn version, a minimal reproducible example, and the complete BestatML error message. Remove sensitive or proprietary data before sharing examples.
