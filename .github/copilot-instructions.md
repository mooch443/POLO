# Copilot instructions for this repo

## Fork context (POLO)
- This repo is a fork of Ultralytics (POLO) that adds localization modes for YOLO (point-only detections).
  - New task name is `locate` (see `TASKS`/`TASK2MODEL` in [ultralytics/cfg/__init__.py](ultralytics/cfg/__init__.py)).
  - Model head and loss are specialized for points: `Locate`/`LocalizationModel` and `v8LocalizationLoss` in [ultralytics/nn/tasks.py](ultralytics/nn/tasks.py).
  - Datasets carry point labels via `use_locations` and `v8_transforms_loc` in [ultralytics/data/dataset.py](ultralytics/data/dataset.py).
  - Validation for localization lives in [ultralytics/models/yolo/locate/val.py](ultralytics/models/yolo/locate/val.py).

## Big-picture architecture
- Core package lives in [ultralytics](ultralytics): entrypoint module, task-specific models, data, engine, and utilities.
- Public API is exposed via lazy imports in [ultralytics/__init__.py](ultralytics/__init__.py) (e.g., `YOLO`, `RTDETR`).
- CLI is implemented in [ultralytics/cfg/__init__.py](ultralytics/cfg/__init__.py) via `entrypoint()` and config parsing (`TASKS`, `MODES`, `DEFAULT_CFG_DICT`).
- Training/prediction flows are orchestrated by engine base classes:
  - `Model` in [ultralytics/engine/model.py](ultralytics/engine/model.py) routes to train/val/predict/export.
  - `BaseTrainer` in [ultralytics/engine/trainer.py](ultralytics/engine/trainer.py) handles datasets, checkpoints, DDP, and run directories.
  - `BasePredictor` in [ultralytics/engine/predictor.py](ultralytics/engine/predictor.py) handles inference sources and multi-format backends.
- Network definitions and losses are centralized in [ultralytics/nn/tasks.py](ultralytics/nn/tasks.py) (model assembly, loss dispatch, fusion helpers).
- Dataset loading and caching lives in [ultralytics/data/dataset.py](ultralytics/data/dataset.py) (YOLO-style labels, `.cache` versioning, task flags).

## Developer workflows (discoverable commands)
- Package metadata and CLI entrypoints are defined in [pyproject.toml](pyproject.toml) (scripts: `yolo`, `ultralytics`).
- Tests run with `pytest`; markers include `slow` (see [pyproject.toml](pyproject.toml)).
- Docs build/preview for the docs site use `mkdocs` (see [docs/README.md](docs/README.md)).

## Project-specific conventions
- CLI arguments use key/value parsing with `arg=value` pairs and infer task/mode (`TASKS`, `MODES`) in [ultralytics/cfg/__init__.py](ultralytics/cfg/__init__.py).
- Model file names are normalized with `check_model_file_from_stem()` before load (see [ultralytics/engine/model.py](ultralytics/engine/model.py)).
- Dataset caching is versioned via `DATASET_CACHE_VERSION`; updates must maintain backwards compatibility in [ultralytics/data/dataset.py](ultralytics/data/dataset.py).
- Run artifacts are organized under a save directory with `weights/last.pt` and `weights/best.pt` per trainer run (see [ultralytics/engine/trainer.py](ultralytics/engine/trainer.py)).

## Integration points
- Multi-format inference backends are wired through `AutoBackend` in [ultralytics/engine/predictor.py](ultralytics/engine/predictor.py).
- Optional dependencies for export, logging, and solutions are enumerated in [pyproject.toml](pyproject.toml); use the matching extra group when adding integrations.
