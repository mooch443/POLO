# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

POLO_PATCH_MARKER_START = "# >>> ultralytics-polo-sahi-patch >>>"
POLO_PATCH_MARKER_END = "# <<< ultralytics-polo-sahi-patch <<<"


def apply_sahi_polo_monkeypatch(
    model_type_aliases: tuple[str, ...] = ("polo", "polo26", "polov8", "locate"),
    install_permanently: bool = False,
) -> bool:
    """Monkeypatch SAHI's Ultralytics backend to support POLO/locate outputs.

    This patch is runtime-only and avoids forking SAHI. Call it once before creating SAHI models.

    Args:
        model_type_aliases (tuple[str, ...]): Additional model_type names that should map to SAHI's
            Ultralytics backend.
        install_permanently (bool): If True, also installs a persistent SAHI hook so the patch auto-applies
            on future ``import sahi`` calls.

    Returns:
        (bool): True if patch was applied in this call, False if patch was already active.
    """
    try:
        import sahi.auto_model as auto_model
        import sahi.models.ultralytics as sahi_ultra
        from sahi.prediction import ObjectPrediction
        from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
    except Exception as exc:  # pragma: no cover
        raise ImportError("SAHI is required for apply_sahi_polo_monkeypatch().") from exc

    if getattr(sahi_ultra, "_polo_monkeypatch_applied", False):
        if install_permanently:
            install_sahi_polo_patch_permanently()
        return False

    cls = sahi_ultra.UltralyticsDetectionModel

    original_init = cls.__init__
    original_set_model = cls.set_model
    original_perform_inference = cls.perform_inference
    original_create_predictions = cls._create_object_prediction_list_from_original_predictions
    original_has_mask = cls.has_mask
    original_is_obb = cls.is_obb
    original_aliases = tuple(getattr(auto_model, "ULTRALYTICS_MODEL_NAMES", []))

    def _normalize_polo_radii(radii: Any) -> dict[int, float]:
        if not isinstance(radii, dict):
            return {}

        normalized: dict[int, float] = {}
        for class_id, value in radii.items():
            try:
                class_id_int = int(class_id)
            except (TypeError, ValueError):
                continue

            radius = value
            if isinstance(radius, dict):
                radius = radius.get("radius", radius.get("value", radius.get("r", None)))
            try:
                normalized[class_id_int] = float(radius)
            except (TypeError, ValueError):
                continue
        return normalized

    def _location_to_bbox(self, x: float, y: float, category_id: int) -> list[float]:
        radius = float(self.polo_radii.get(category_id, 1.0))
        half = max(radius * self.polo_box_padding, 1e-3)
        return [x - half, y - half, x + half, y + half]

    def __init__(self, *args, **kwargs):
        self.polo_box_padding: float = float(kwargs.pop("polo_box_padding", 1.0))
        self.polo_dor: float | None = kwargs.pop("dor", None)
        self.polo_radii: dict[int, float] = _normalize_polo_radii(kwargs.pop("radii", None))
        return original_init(self, *args, **kwargs)

    def set_model(self, model: Any, **kwargs):
        original_set_model(self, model, **kwargs)
        if not self.polo_radii:
            model_radii = getattr(model, "radii", None)
            if model_radii is None:
                predictor = getattr(model, "predictor", None)
                model_radii = getattr(predictor, "radii", None)
            self.polo_radii = _normalize_polo_radii(model_radii)

    def perform_inference(self, image):
        if not self.is_locate:
            return original_perform_inference(self, image)

        import torch

        if self.model is None:
            raise ValueError("Model is not loaded, load it by calling .load_model()")

        kwargs = {"cfg": self.config_path, "verbose": False, "conf": self.confidence_threshold, "device": self.device}
        if self.image_size is not None:
            kwargs["imgsz"] = self.image_size
        if self.polo_radii:
            kwargs["radii"] = self.polo_radii
        if self.polo_dor is not None:
            kwargs["dor"] = self.polo_dor

        prediction_result = self.model(image[:, :, ::-1], **kwargs)  # YOLO expects numpy arrays in BGR order
        device = getattr(self.model, "device", "cpu")
        self._original_predictions = [
            result.locations.data if getattr(result, "locations", None) is not None else torch.empty((0, 4), device=device)
            for result in prediction_result
        ]
        self._original_shape = image.shape

    def _task_name(self) -> str | None:
        if hasattr(self.model, "overrides") and isinstance(self.model.overrides, dict):
            task = self.model.overrides.get("task")
            if task is not None:
                return str(task).lower()

        if hasattr(self.model, "task") and getattr(self.model, "task", None) is not None:
            return str(self.model.task).lower()

        if self.model_path and isinstance(self.model_path, str):
            model_path = self.model_path.lower()
            if "obb" in model_path:
                return "obb"
            if "seg" in model_path:
                return "segment"
            if "polo" in model_path or "locate" in model_path or "-loc" in model_path:
                return "locate"
        return None

    def has_mask(self):
        return self._task_name == "segment"

    def is_obb(self):
        return self._task_name == "obb"

    def is_locate(self):
        return self._task_name == "locate"

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: list[list[int]] | None = [[0, 0]],
        full_shape_list: list[list[int]] | None = None,
    ):
        if not self.is_locate:
            return original_create_predictions(
                self,
                shift_amount_list=shift_amount_list,
                full_shape_list=full_shape_list,
            )

        original_predictions = self._original_predictions
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        object_prediction_list_per_image = []
        for image_ind, image_predictions in enumerate(original_predictions):
            shift_amount = shift_amount_list[image_ind]
            full_shape = None if full_shape_list is None else full_shape_list[image_ind]
            object_prediction_list = []

            boxes = image_predictions.cpu().detach().numpy()
            for prediction in boxes:
                x = float(prediction[0])
                y = float(prediction[1])
                score = float(prediction[-2])
                category_id = int(prediction[-1])
                category_name = self.category_mapping.get(str(category_id), str(category_id))
                bbox = [max(0.0, coord) for coord in _location_to_bbox(self, x=x, y=y, category_id=category_id)]

                if full_shape is not None:
                    bbox[0] = min(full_shape[1], bbox[0])
                    bbox[1] = min(full_shape[0], bbox[1])
                    bbox[2] = min(full_shape[1], bbox[2])
                    bbox[3] = min(full_shape[0], bbox[3])

                if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
                    continue

                object_prediction_list.append(
                    ObjectPrediction(
                        bbox=bbox,
                        category_id=category_id,
                        score=score,
                        segmentation=None,
                        category_name=category_name,
                        shift_amount=shift_amount,
                        full_shape=self._original_shape[:2] if full_shape is None else full_shape,
                    )
                )

            object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image

    cls.__init__ = __init__
    cls.set_model = set_model
    cls.perform_inference = perform_inference
    cls._task_name = property(_task_name)
    cls.has_mask = property(has_mask)
    cls.is_obb = property(is_obb)
    cls.is_locate = property(is_locate)
    cls._normalize_polo_radii = staticmethod(_normalize_polo_radii)
    cls._location_to_bbox = _location_to_bbox
    cls._create_object_prediction_list_from_original_predictions = _create_object_prediction_list_from_original_predictions

    aliases = list(original_aliases)
    for alias in model_type_aliases:
        if alias not in aliases:
            aliases.append(alias)
    auto_model.ULTRALYTICS_MODEL_NAMES = aliases

    sahi_ultra._polo_monkeypatch_state = {
        "class": cls,
        "original_init": original_init,
        "original_set_model": original_set_model,
        "original_perform_inference": original_perform_inference,
        "original_create_predictions": original_create_predictions,
        "original_has_mask": original_has_mask,
        "original_is_obb": original_is_obb,
        "original_aliases": original_aliases,
    }
    sahi_ultra._polo_monkeypatch_applied = True
    if install_permanently:
        install_sahi_polo_patch_permanently()
    return True


def remove_sahi_polo_monkeypatch() -> bool:
    """Revert :func:`apply_sahi_polo_monkeypatch` runtime changes."""
    import sahi.auto_model as auto_model
    import sahi.models.ultralytics as sahi_ultra

    if not getattr(sahi_ultra, "_polo_monkeypatch_applied", False):
        return False

    state = getattr(sahi_ultra, "_polo_monkeypatch_state", None)
    if not state:
        return False

    cls = state["class"]
    cls.__init__ = state["original_init"]
    cls.set_model = state["original_set_model"]
    cls.perform_inference = state["original_perform_inference"]
    cls._create_object_prediction_list_from_original_predictions = state["original_create_predictions"]
    cls.has_mask = state["original_has_mask"]
    cls.is_obb = state["original_is_obb"]
    for attr_name in ("is_locate", "_task_name", "_normalize_polo_radii", "_location_to_bbox"):
        if hasattr(cls, attr_name):
            delattr(cls, attr_name)

    auto_model.ULTRALYTICS_MODEL_NAMES = list(state["original_aliases"])

    sahi_ultra._polo_monkeypatch_applied = False
    sahi_ultra._polo_monkeypatch_state = None
    return True


def install_sahi_polo_patch_permanently() -> bool:
    """Install a persistent SAHI bootstrap hook for POLO monkeypatching.

    This writes one small helper module under SAHI and appends a guarded import block to ``sahi/__init__.py``.

    Returns:
        (bool): True if any file changed, False if already installed.
    """
    import sahi

    sahi_dir = Path(sahi.__file__).resolve().parent
    init_file = sahi_dir / "__init__.py"
    bootstrap_file = sahi_dir / "_ultralytics_polo_patch.py"
    changed = False

    bootstrap_contents = (
        "from ultralytics.utils.sahi import apply_sahi_polo_monkeypatch\n"
        "\n"
        "\n"
        "def apply() -> bool:\n"
        "    \"\"\"Apply Ultralytics POLO patch to SAHI at import time.\"\"\"\n"
        "    return apply_sahi_polo_monkeypatch()\n"
    )
    if not bootstrap_file.exists() or bootstrap_file.read_text(encoding="utf-8") != bootstrap_contents:
        bootstrap_file.write_text(bootstrap_contents, encoding="utf-8")
        changed = True

    init_text = init_file.read_text(encoding="utf-8")
    if POLO_PATCH_MARKER_START not in init_text:
        snippet = (
            f"\n{POLO_PATCH_MARKER_START}\n"
            "try:\n"
            "    from sahi._ultralytics_polo_patch import apply as _apply_ultralytics_polo_patch\n"
            "    _apply_ultralytics_polo_patch()\n"
            "except Exception:\n"
            "    pass\n"
            f"{POLO_PATCH_MARKER_END}\n"
        )
        init_file.write_text(init_text + snippet, encoding="utf-8")
        changed = True

    return changed


def uninstall_sahi_polo_patch_permanently() -> bool:
    """Remove persistent SAHI bootstrap hook installed by ``install_sahi_polo_patch_permanently``."""
    import sahi

    sahi_dir = Path(sahi.__file__).resolve().parent
    init_file = sahi_dir / "__init__.py"
    bootstrap_file = sahi_dir / "_ultralytics_polo_patch.py"
    changed = False

    init_text = init_file.read_text(encoding="utf-8")
    if POLO_PATCH_MARKER_START in init_text and POLO_PATCH_MARKER_END in init_text:
        start = init_text.index(POLO_PATCH_MARKER_START)
        end = init_text.index(POLO_PATCH_MARKER_END) + len(POLO_PATCH_MARKER_END)
        while end < len(init_text) and init_text[end] == "\n":
            end += 1
        init_file.write_text(init_text[:start] + init_text[end:], encoding="utf-8")
        changed = True

    if bootstrap_file.exists():
        bootstrap_file.unlink()
        changed = True

    return changed
