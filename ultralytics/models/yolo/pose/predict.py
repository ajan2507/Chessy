<<<<<<< HEAD
# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops


class PosePredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a pose model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.pose import PosePredictor

        args = dict(model='yolov8n-pose.pt', source=ASSETS)
        predictor = PosePredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes PosePredictor, sets task to 'pose' and logs a warning for using 'mps' as device."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = 'pose'
        if isinstance(self.args.device, str) and self.args.device.lower() == 'mps':
            LOGGER.warning("WARNING ⚠️ Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                           'See https://github.com/ultralytics/ultralytics/issues/4031.')

    def postprocess(self, preds, img, orig_imgs):
        """Return detection results for a given input image or list of images."""
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes,
                                        nc=len(self.model.names))

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape).round()
            pred_kpts = pred[:, 6:].view(len(pred), *self.model.kpt_shape) if len(pred) else pred[:, 6:]
            pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
            img_path = self.batch[0][i]
            results.append(
                Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], keypoints=pred_kpts))
        return results
=======
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops


class PosePredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a pose model.

    This class specializes in pose estimation, handling keypoints detection alongside standard object detection
    capabilities inherited from DetectionPredictor.

    Attributes:
        args (namespace): Configuration arguments for the predictor.
        model (torch.nn.Module): The loaded YOLO pose model with keypoint detection capabilities.

    Methods:
        construct_result: Construct the result object from the prediction, including keypoints.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.pose import PosePredictor
        >>> args = dict(model="yolo11n-pose.pt", source=ASSETS)
        >>> predictor = PosePredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize PosePredictor for pose estimation tasks.

        Sets up a PosePredictor instance, configuring it for pose detection tasks and handling device-specific
        warnings for Apple MPS.

        Args:
            cfg (Any): Configuration for the predictor.
            overrides (dict, optional): Configuration overrides that take precedence over cfg.
            _callbacks (list, optional): List of callback functions to be invoked during prediction.

        Examples:
            >>> from ultralytics.utils import ASSETS
            >>> from ultralytics.models.yolo.pose import PosePredictor
            >>> args = dict(model="yolo11n-pose.pt", source=ASSETS)
            >>> predictor = PosePredictor(overrides=args)
            >>> predictor.predict_cli()
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "pose"
        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def construct_result(self, pred, img, orig_img, img_path):
        """
        Construct the result object from the prediction, including keypoints.

        Extends the parent class implementation by extracting keypoint data from predictions and adding them to the
        result object.

        Args:
            pred (torch.Tensor): The predicted bounding boxes, scores, and keypoints with shape (N, 6+K*D) where N is
                the number of detections, K is the number of keypoints, and D is the keypoint dimension.
            img (torch.Tensor): The processed input image tensor with shape (B, C, H, W).
            orig_img (np.ndarray): The original unprocessed image as a numpy array.
            img_path (str): The path to the original image file.

        Returns:
            (Results): The result object containing the original image, image path, class names, bounding boxes, and
                keypoints.
        """
        result = super().construct_result(pred, img, orig_img, img_path)
        # Extract keypoints from prediction and reshape according to model's keypoint shape
        pred_kpts = pred[:, 6:].view(len(pred), *self.model.kpt_shape)
        # Scale keypoints coordinates to match the original image dimensions
        pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
        result.update(keypoints=pred_kpts)
        return result
>>>>>>> 0e451281e978a7c5f581cc5d9bb3d0f62e0bc632
