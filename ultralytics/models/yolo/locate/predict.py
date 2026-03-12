# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops
from ultralytics.data.utils import check_det_dataset


class LocalizationPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a localization model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import LocalizationPredictor

        args = dict(model='yolov8n.pt', source=ASSETS)
        predictor = LocalizationPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes the SegmentationPredictor with the provided configuration, overrides, and callbacks."""
        overrides = {} if overrides is None else {**overrides}
        overrides["task"] = "locate"
        radii = overrides.pop("radii", None)

        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "locate"
        self.args.conf = ops.resolve_locate_conf(self.args.conf)

        if radii is None:
            if self.data:
                try:
                    data = check_det_dataset(dataset=self.data)
                    radii = data["radii"]
                except FileNotFoundError as exc:
                    LOGGER.warning(
                        "WARNING ⚠️ dataset config not found for locate predictor; falling back to model radii. "
                        f"({exc})"
                    )
                    radii = getattr(self.model, "radii", None) or {}
            else:
                radii = getattr(self.model, "radii", None) or {}

        self.radii = radii

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression_loc(
            prediction=preds,
            conf_thres=self.args.conf,
            dor_thres=self.args.dor,
            radii=self.radii,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :2] = ops.scale_locations(img.shape[2:], pred[:, :2], orig_img.shape, remove_clipped=False)
            outside_img = (pred[:, 0] == 0) & (pred[:, 1] == 0)
            pred = pred[~outside_img]
            radii = ops.generate_radii_t(self.radii, pred[:, 3:4]) if len(pred) else pred.new_zeros((0, 1))

            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, locations=pred, location_radii=radii))
        return results
