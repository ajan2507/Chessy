<<<<<<< HEAD
# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops

__all__ = ['NASValidator']


class NASValidator(DetectionValidator):
    """
    Ultralytics YOLO NAS Validator for object detection.

    Extends `DetectionValidator` from the Ultralytics models package and is designed to post-process the raw predictions
    generated by YOLO NAS models. It performs non-maximum suppression to remove overlapping and low-confidence boxes,
    ultimately producing the final detections.

    Attributes:
        args (Namespace): Namespace containing various configurations for post-processing, such as confidence and IoU thresholds.
        lb (torch.Tensor): Optional tensor for multilabel NMS.

    Example:
        ```python
        from ultralytics import NAS

        model = NAS('yolo_nas_s')
        validator = model.validator
        # Assumes that raw_preds are available
        final_preds = validator.postprocess(raw_preds)
        ```

    Note:
        This class is generally not instantiated directly but is used internally within the `NAS` class.
    """

    def postprocess(self, preds_in):
        """Apply Non-maximum suppression to prediction outputs."""
        boxes = ops.xyxy2xywh(preds_in[0][0])
        preds = torch.cat((boxes, preds_in[0][1]), -1).permute(0, 2, 1)
        return ops.non_max_suppression(preds,
                                       self.args.conf,
                                       self.args.iou,
                                       labels=self.lb,
                                       multi_label=False,
                                       agnostic=self.args.single_cls,
                                       max_det=self.args.max_det,
                                       max_time_img=0.5)
=======
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops

__all__ = ["NASValidator"]


class NASValidator(DetectionValidator):
    """
    Ultralytics YOLO NAS Validator for object detection.

    Extends DetectionValidator from the Ultralytics models package and is designed to post-process the raw predictions
    generated by YOLO NAS models. It performs non-maximum suppression to remove overlapping and low-confidence boxes,
    ultimately producing the final detections.

    Attributes:
        args (Namespace): Namespace containing various configurations for post-processing, such as confidence and IoU
            thresholds.
        lb (torch.Tensor): Optional tensor for multilabel NMS.

    Examples:
        >>> from ultralytics import NAS
        >>> model = NAS("yolo_nas_s")
        >>> validator = model.validator
        >>> # Assumes that raw_preds are available
        >>> final_preds = validator.postprocess(raw_preds)

    Notes:
        This class is generally not instantiated directly but is used internally within the NAS class.
    """

    def postprocess(self, preds_in):
        """Apply Non-maximum suppression to prediction outputs."""
        boxes = ops.xyxy2xywh(preds_in[0][0])  # Convert bounding box format from xyxy to xywh
        preds = torch.cat((boxes, preds_in[0][1]), -1).permute(0, 2, 1)  # Concatenate boxes with scores and permute
        return super().postprocess(preds)
>>>>>>> 0e451281e978a7c5f581cc5d9bb3d0f62e0bc632
