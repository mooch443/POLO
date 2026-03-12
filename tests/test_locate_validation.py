from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from ultralytics.engine.results import Results
from ultralytics.models.yolo.locate import LocalizationPredictor, LocalizationValidator
from ultralytics.utils import ops
from ultralytics.utils.loss import v8LocalizationLoss
from ultralytics.utils.plotting import Annotator


def _locate_prediction(*anchors):
    """Build a single-image locate prediction tensor from anchor tuples."""
    rows = list(zip(*anchors))
    return torch.tensor([rows], dtype=torch.float32)


def test_locate_validator_defaults_to_fast_mode_and_consistent_conf(monkeypatch):
    """Locate validation should default to the fast path and reuse one resolved conf threshold."""
    call = {}

    def fake_nms(**kwargs):
        call.update(kwargs)
        return [torch.zeros((0, 4), dtype=torch.float32)]

    monkeypatch.setattr(ops, "non_max_suppression_loc", fake_nms)
    validator = LocalizationValidator(args={"max_det": 5})
    validator.data = {"radii": {0: 5.0, 1: 8.0}, "val": "images/val"}
    validator.init_metrics(SimpleNamespace(names={0: "class0", 1: "class1"}))

    preds = _locate_prediction((10.0, 20.0, 0.9, 0.1), (30.0, 40.0, 0.2, 0.8))
    validator.postprocess(preds)

    assert validator.args.locate_val_mode == "fast"
    assert validator.args.conf == 0.25
    assert validator.confusion_matrix.conf == 0.25
    assert call["conf_thres"] == 0.25
    assert call["multi_label"] is False


def test_locate_validator_legacy_mode_preserves_multi_label(monkeypatch):
    """Legacy locate validation mode should keep the previous multi-label behavior."""
    call = {}

    def fake_nms(**kwargs):
        call.update(kwargs)
        return [torch.zeros((0, 4), dtype=torch.float32)]

    monkeypatch.setattr(ops, "non_max_suppression_loc", fake_nms)
    validator = LocalizationValidator(args={"locate_val_mode": "legacy", "max_det": 5})
    validator.data = {"radii": {0: 5.0, 1: 8.0}, "val": "images/val"}

    preds = _locate_prediction((10.0, 20.0, 0.9, 0.1), (30.0, 40.0, 0.2, 0.8))
    validator.postprocess(preds)

    assert call["multi_label"] is True


def test_locate_validator_uses_explicit_conf_everywhere(monkeypatch):
    """Explicit locate conf should flow through validator postprocess and confusion-matrix setup unchanged."""
    call = {}

    def fake_nms(**kwargs):
        call.update(kwargs)
        return [torch.zeros((0, 4), dtype=torch.float32)]

    monkeypatch.setattr(ops, "non_max_suppression_loc", fake_nms)
    validator = LocalizationValidator(args={"conf": 0.42, "max_det": 5})
    validator.data = {"radii": {0: 5.0, 1: 8.0}, "val": "images/val"}
    validator.init_metrics(SimpleNamespace(names={0: "class0", 1: "class1"}))

    preds = _locate_prediction((10.0, 20.0, 0.9, 0.1), (30.0, 40.0, 0.2, 0.8))
    validator.postprocess(preds)

    assert validator.args.conf == 0.42
    assert validator.confusion_matrix.conf == 0.42
    assert call["conf_thres"] == 0.42


def test_locate_predictor_and_validator_share_default_candidate_policy(monkeypatch):
    """Default locate predictor and validator should both use the fast single-label candidate policy."""
    calls = []

    def fake_nms(**kwargs):
        calls.append(kwargs)
        return [torch.zeros((0, 4), dtype=torch.float32)]

    monkeypatch.setattr(ops, "non_max_suppression_loc", fake_nms)

    validator = LocalizationValidator(args={"max_det": 5})
    validator.data = {"radii": {0: 5.0, 1: 8.0}, "val": "images/val"}
    validator.postprocess(_locate_prediction((10.0, 20.0, 0.9, 0.1),))

    predictor = LocalizationPredictor(overrides={"max_det": 5})
    predictor.radii = {0: 5.0, 1: 8.0}
    predictor.batch = (["image.jpg"], None, None)
    predictor.model = SimpleNamespace(names={0: "class0", 1: "class1"})
    predictor.postprocess(
        _locate_prediction((10.0, 20.0, 0.9, 0.1),),
        img=torch.zeros((1, 3, 32, 32), dtype=torch.float32),
        orig_imgs=[np.zeros((32, 32, 3), dtype=np.uint8)],
    )

    assert len(calls) == 2
    assert calls[0]["multi_label"] is False
    assert calls[1].get("multi_label", False) is False
    assert calls[0]["conf_thres"] == 0.25
    assert calls[1]["conf_thres"] == 0.25


def test_locate_nms_ignores_low_confidence_anchors_in_fast_mode():
    """Low-confidence anchors should not affect locate NMS outputs in the fast single-label path."""
    base = _locate_prediction(
        (10.0, 20.0, 0.95, 0.10),
        (40.0, 50.0, 0.05, 0.90),
    )
    noisy = _locate_prediction(
        (10.0, 20.0, 0.95, 0.10),
        (40.0, 50.0, 0.05, 0.90),
        (100.0, 120.0, 0.24, 0.23),
        (130.0, 150.0, 0.01, 0.02),
    )

    out_base = ops.non_max_suppression_loc(
        base, conf_thres=0.25, dor_thres=0.3, radii={0: 5.0, 1: 8.0}, multi_label=False
    )
    out_noisy = ops.non_max_suppression_loc(
        noisy, conf_thres=0.25, dor_thres=0.3, radii={0: 5.0, 1: 8.0}, multi_label=False
    )

    assert len(out_base) == len(out_noisy) == 1
    torch.testing.assert_close(out_base[0], out_noisy[0])


def test_locate_results_plot_respects_conf_threshold_and_uses_radii(monkeypatch):
    """Locate result plotting should skip sub-threshold points and draw with stored per-location radii."""
    calls = []

    def fake_loc_label(self, loc, label="", color=(128, 128, 128), txt_color=(255, 255, 255), loc_radius=4):
        calls.append({"loc": tuple(loc), "label": label, "loc_radius": loc_radius})

    monkeypatch.setattr(Annotator, "loc_label", fake_loc_label)
    result = Results(
        orig_img=np.zeros((32, 32, 3), dtype=np.uint8),
        path="image.jpg",
        names={0: "class0", 1: "class1"},
        locations=torch.tensor([[8.0, 10.0, 0.91, 0.0], [16.0, 20.0, 0.24, 1.0]], dtype=torch.float32),
        location_radii=torch.tensor([[7.0], [11.0]], dtype=torch.float32),
    )

    result.plot(conf=True, conf_thres=0.25)

    assert len(calls) == 1
    assert calls[0]["label"] == "class0 0.91"
    assert calls[0]["loc_radius"] == 7


def test_locate_validator_prediction_plots_pass_predicted_radii(monkeypatch):
    """Locate validation plots should forward per-class radii for predicted circles."""
    call = {}

    def fake_plot_images(**kwargs):
        call.update(kwargs)

    monkeypatch.setattr("ultralytics.models.yolo.locate.val.plot_images", fake_plot_images)
    validator = LocalizationValidator(args={"max_det": 5})
    validator.data = {"radii": {0: 5.0, 1: 8.0}, "val": "images/val"}
    validator.names = {0: "class0", 1: "class1"}

    preds = [torch.tensor([[10.0, 12.0, 0.9, 0.0], [20.0, 22.0, 0.8, 1.0]], dtype=torch.float32)]
    batch = {
        "img": torch.zeros((1, 3, 32, 32), dtype=torch.float32),
        "im_file": ["image.jpg"],
    }
    validator.plot_predictions(batch, preds, 0)

    assert "radii" in call
    np.testing.assert_allclose(np.asarray(call["radii"]).reshape(-1), np.array([5.0, 8.0], dtype=np.float32))


def test_locate_loss_sanitizes_non_finite_logits():
    """Locate loss should keep non-finite logits from immediately poisoning the AMP-sensitive loss path."""

    class _FakeHead:
        stride = torch.tensor([8.0])
        nc = 2
        no = 4
        reg_max = 1

    class _FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))
            self.args = SimpleNamespace(loc_loss="mse", loc=0.5, cls=0.5)
            self.model = [_FakeHead()]
            self.radii = {0: 5.0, 1: 8.0}

    criterion = v8LocalizationLoss(_FakeModel())
    criterion.assigner.topk = 1
    feat = torch.tensor([[[[0.0, 0.0]], [[0.0, 0.0]], [[float("nan"), 0.5]], [[0.1, 0.2]]]], dtype=torch.float32)
    batch = {
        "batch_idx": torch.tensor([0.0], dtype=torch.float32),
        "cls": torch.tensor([[0.0]], dtype=torch.float32),
        "radii": torch.tensor([[5.0]], dtype=torch.float32),
        "locations": torch.tensor([[0.5, 0.5]], dtype=torch.float32),
    }

    loss, loss_items = criterion([feat], batch)

    assert torch.isfinite(loss)
    assert torch.isfinite(loss_items).all()
