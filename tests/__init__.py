# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

import numpy as np
from PIL import Image

from ultralytics.cfg import TASK2DATA, TASK2MODEL, TASKS
from ultralytics.utils import ASSETS, WEIGHTS_DIR, checks


def _make_locate_dataset() -> Path:
    root = Path(__file__).resolve().parent / "tmp/locate"
    yaml_path = root / "locate-synth.yaml"
    if yaml_path.exists():
        return yaml_path

    images_dir = root / "images"
    labels_dir = root / "labels"
    for split in ("train", "val"):
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    radii = {0: 5.0, 1: 8.0}

    for split, count in (("train", 4), ("val", 2)):
        for i in range(count):
            img = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
            img_path = images_dir / split / f"img_{i}.jpg"
            Image.fromarray(img).save(img_path)

            cls = int(rng.integers(0, len(radii)))
            radius = radii[cls]
            x = float(rng.random() * 0.8 + 0.1)
            y = float(rng.random() * 0.8 + 0.1)
            label_path = labels_dir / split / f"img_{i}.txt"
            label_path.write_text(f"{cls} {radius:.1f} {x:.6f} {y:.6f}\n")

    yaml_path.write_text(
        "\n".join(
            [
                f"path: {root}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: class0",
                "  1: class1",
                "radii:",
                "  0: 5.0",
                "  1: 8.0",
                "",
            ]
        )
    )
    return yaml_path


TASK2DATA["locate"] = str(_make_locate_dataset())


def _task_model_path(task: str) -> Path:
    if task == "locate":
        return Path("ultralytics/cfg/models/v8/polov8.yaml")
    return WEIGHTS_DIR / TASK2MODEL[task]

# Constants used in tests
MODEL = WEIGHTS_DIR / "path with spaces" / "yolo26n.pt"  # test spaces in path
CFG = "yolo26n.yaml"
SOURCE = ASSETS / "bus.jpg"
SOURCES_LIST = [ASSETS / "bus.jpg", ASSETS, ASSETS / "*", ASSETS / "**/*.jpg"]
CUDA_IS_AVAILABLE = checks.cuda_is_available()
CUDA_DEVICE_COUNT = checks.cuda_device_count()
TASK_MODEL_DATA = [(task, _task_model_path(task), TASK2DATA[task]) for task in TASKS]
MODELS = frozenset([*[v for k, v in TASK2MODEL.items() if k != "locate"], "yolo11n-grayscale.pt"])

__all__ = (
    "CFG",
    "CUDA_DEVICE_COUNT",
    "CUDA_IS_AVAILABLE",
    "MODEL",
    "SOURCE",
    "SOURCES_LIST",
)
