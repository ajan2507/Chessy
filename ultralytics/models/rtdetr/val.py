<<<<<<< HEAD
# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

import torch

from ultralytics.data import YOLODataset
from ultralytics.data.augment import Compose, Format, v8_transforms
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import colorstr, ops

__all__ = 'RTDETRValidator',  # tuple or list


class RTDETRDataset(YOLODataset):
    """
    Real-Time DEtection and TRacking (RT-DETR) dataset class extending the base YOLODataset class.

    This specialized dataset class is designed for use with the RT-DETR object detection model and is optimized for
    real-time detection and tracking tasks.
    """

    def __init__(self, *args, data=None, **kwargs):
        """Initialize the RTDETRDataset class by inheriting from the YOLODataset class."""
        super().__init__(*args, data=data, use_segments=False, use_keypoints=False, **kwargs)

    # NOTE: add stretch version load_image for RTDETR mosaic
    def load_image(self, i, rect_mode=False):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        return super().load_image(i=i, rect_mode=rect_mode)

    def build_transforms(self, hyp=None):
        """Temporary, only for evaluation."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp, stretch=True)
        else:
            # transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), auto=False, scaleFill=True)])
            transforms = Compose([])
        transforms.append(
            Format(bbox_format='xywh',
                   normalize=True,
                   return_mask=self.use_segments,
                   return_keypoint=self.use_keypoints,
                   batch_idx=True,
                   mask_ratio=hyp.mask_ratio,
                   mask_overlap=hyp.overlap_mask))
        return transforms


class RTDETRValidator(DetectionValidator):
    """
    RTDETRValidator extends the DetectionValidator class to provide validation capabilities specifically tailored for
    the RT-DETR (Real-Time DETR) object detection model.

    The class allows building of an RTDETR-specific dataset for validation, applies Non-maximum suppression for
    post-processing, and updates evaluation metrics accordingly.

    Example:
        ```python
        from ultralytics.models.rtdetr import RTDETRValidator

        args = dict(model='rtdetr-l.pt', data='coco8.yaml')
        validator = RTDETRValidator(args=args)
        validator()
        ```

    Note:
        For further details on the attributes and methods, refer to the parent DetectionValidator class.
    """

    def build_dataset(self, img_path, mode='val', batch=None):
        """
        Build an RTDETR Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,  # no augmentation
            hyp=self.args,
            rect=False,  # no rect
            cache=self.args.cache or None,
            prefix=colorstr(f'{mode}: '),
            data=self.data)

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        bs, _, nd = preds[0].shape
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)
        bboxes *= self.args.imgsz
        outputs = [torch.zeros((0, 6), device=bboxes.device)] * bs
        for i, bbox in enumerate(bboxes):  # (300, 4)
            bbox = ops.xywh2xyxy(bbox)
            score, cls = scores[i].max(-1)  # (300, )
            # Do not need threshold for evaluation as only got 300 boxes here
            # idx = score > self.args.conf
            pred = torch.cat([bbox, score[..., None], cls[..., None]], dim=-1)  # filter
            # Sort by confidence to correctly get internal metrics
            pred = pred[score.argsort(descending=True)]
            outputs[i] = pred  # [idx]

        return outputs

    def update_metrics(self, preds, batch):
        """Metrics."""
        for si, pred in enumerate(preds):
            idx = batch['batch_idx'] == si
            cls = batch['cls'][idx]
            bbox = batch['bboxes'][idx]
            nl, npr = cls.shape[0], pred.shape[0]  # number of labels, predictions
            shape = batch['ori_shape'][si]
            correct_bboxes = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)  # init
            self.seen += 1

            if npr == 0:
                if nl:
                    self.stats.append((correct_bboxes, *torch.zeros((2, 0), device=self.device), cls.squeeze(-1)))
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, labels=cls.squeeze(-1))
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            predn[..., [0, 2]] *= shape[1] / self.args.imgsz  # native-space pred
            predn[..., [1, 3]] *= shape[0] / self.args.imgsz  # native-space pred

            # Evaluate
            if nl:
                tbox = ops.xywh2xyxy(bbox)  # target boxes
                tbox[..., [0, 2]] *= shape[1]  # native-space pred
                tbox[..., [1, 3]] *= shape[0]  # native-space pred
                labelsn = torch.cat((cls, tbox), 1)  # native-space labels
                # NOTE: To get correct metrics, the inputs of `_process_batch` should always be float32 type.
                correct_bboxes = self._process_batch(predn.float(), labelsn)
                # TODO: maybe remove these `self.` arguments as they already are member variable
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, labelsn)
            self.stats.append((correct_bboxes, pred[:, 4], pred[:, 5], cls.squeeze(-1)))  # (conf, pcls, tcls)

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch['im_file'][si])
            if self.args.save_txt:
                file = self.save_dir / 'labels' / f'{Path(batch["im_file"][si]).stem}.txt'
                self.save_one_txt(predn, self.args.save_conf, shape, file)
=======
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import torch

from ultralytics.data import YOLODataset
from ultralytics.data.augment import Compose, Format, v8_transforms
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import colorstr, ops

__all__ = ("RTDETRValidator",)  # tuple or list


class RTDETRDataset(YOLODataset):
    """
    Real-Time DEtection and TRacking (RT-DETR) dataset class extending the base YOLODataset class.

    This specialized dataset class is designed for use with the RT-DETR object detection model and is optimized for
    real-time detection and tracking tasks.

    Attributes:
        augment (bool): Whether to apply data augmentation.
        rect (bool): Whether to use rectangular training.
        use_segments (bool): Whether to use segmentation masks.
        use_keypoints (bool): Whether to use keypoint annotations.
        imgsz (int): Target image size for training.

    Methods:
        load_image: Load one image from dataset index.
        build_transforms: Build transformation pipeline for the dataset.

    Examples:
        Initialize an RT-DETR dataset
        >>> dataset = RTDETRDataset(img_path="path/to/images", imgsz=640)
        >>> image, hw = dataset.load_image(0)
    """

    def __init__(self, *args, data=None, **kwargs):
        """
        Initialize the RTDETRDataset class by inheriting from the YOLODataset class.

        This constructor sets up a dataset specifically optimized for the RT-DETR (Real-Time DEtection and TRacking)
        model, building upon the base YOLODataset functionality.

        Args:
            *args (Any): Variable length argument list passed to the parent YOLODataset class.
            data (dict | None): Dictionary containing dataset information. If None, default values will be used.
            **kwargs (Any): Additional keyword arguments passed to the parent YOLODataset class.
        """
        super().__init__(*args, data=data, **kwargs)

    def load_image(self, i, rect_mode=False):
        """
        Load one image from dataset index 'i'.

        Args:
            i (int): Index of the image to load.
            rect_mode (bool, optional): Whether to use rectangular mode for batch inference.

        Returns:
            im (torch.Tensor): The loaded image.
            resized_hw (tuple): Height and width of the resized image with shape (2,).

        Examples:
            Load an image from the dataset
            >>> dataset = RTDETRDataset(img_path="path/to/images")
            >>> image, hw = dataset.load_image(0)
        """
        return super().load_image(i=i, rect_mode=rect_mode)

    def build_transforms(self, hyp=None):
        """
        Build transformation pipeline for the dataset.

        Args:
            hyp (dict, optional): Hyperparameters for transformations.

        Returns:
            (Compose): Composition of transformation functions.
        """
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            hyp.cutmix = hyp.cutmix if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp, stretch=True)
        else:
            # transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), auto=False, scale_fill=True)])
            transforms = Compose([])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
            )
        )
        return transforms


class RTDETRValidator(DetectionValidator):
    """
    RTDETRValidator extends the DetectionValidator class to provide validation capabilities specifically tailored for
    the RT-DETR (Real-Time DETR) object detection model.

    The class allows building of an RTDETR-specific dataset for validation, applies Non-maximum suppression for
    post-processing, and updates evaluation metrics accordingly.

    Attributes:
        args (Namespace): Configuration arguments for validation.
        data (dict): Dataset configuration dictionary.

    Methods:
        build_dataset: Build an RTDETR Dataset for validation.
        postprocess: Apply Non-maximum suppression to prediction outputs.

    Examples:
        Initialize and run RT-DETR validation
        >>> from ultralytics.models.rtdetr import RTDETRValidator
        >>> args = dict(model="rtdetr-l.pt", data="coco8.yaml")
        >>> validator = RTDETRValidator(args=args)
        >>> validator()

    Notes:
        For further details on the attributes and methods, refer to the parent DetectionValidator class.
    """

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build an RTDETR Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str, optional): `train` mode or `val` mode, users are able to customize different augmentations for
                each mode.
            batch (int, optional): Size of batches, this is for `rect`.

        Returns:
            (RTDETRDataset): Dataset configured for RT-DETR validation.
        """
        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,  # no augmentation
            hyp=self.args,
            rect=False,  # no rect
            cache=self.args.cache or None,
            prefix=colorstr(f"{mode}: "),
            data=self.data,
        )

    def postprocess(
        self, preds: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Apply Non-maximum suppression to prediction outputs.

        Args:
            preds (torch.Tensor | List | Tuple): Raw predictions from the model. If tensor, should have shape
                (batch_size, num_predictions, num_classes + 4) where last dimension contains bbox coords and class scores.

        Returns:
            (List[Dict[str, torch.Tensor]]): List of dictionaries for each image, each containing:
                - 'bboxes': Tensor of shape (N, 4) with bounding box coordinates
                - 'conf': Tensor of shape (N,) with confidence scores
                - 'cls': Tensor of shape (N,) with class indices
        """
        if not isinstance(preds, (list, tuple)):  # list for PyTorch inference but list[0] Tensor for export inference
            preds = [preds, None]

        bs, _, nd = preds[0].shape
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)
        bboxes *= self.args.imgsz
        outputs = [torch.zeros((0, 6), device=bboxes.device)] * bs
        for i, bbox in enumerate(bboxes):  # (300, 4)
            bbox = ops.xywh2xyxy(bbox)
            score, cls = scores[i].max(-1)  # (300, )
            pred = torch.cat([bbox, score[..., None], cls[..., None]], dim=-1)  # filter
            # Sort by confidence to correctly get internal metrics
            pred = pred[score.argsort(descending=True)]
            outputs[i] = pred[score > self.args.conf]

        return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5]} for x in outputs]

    def pred_to_json(self, predn: Dict[str, torch.Tensor], pbatch: Dict[str, Any]) -> None:
        """
        Serialize YOLO predictions to COCO json format.

        Args:
            predn (Dict[str, torch.Tensor]): Predictions dictionary containing 'bboxes', 'conf', and 'cls' keys
                with bounding box coordinates, confidence scores, and class predictions.
            pbatch (Dict[str, Any]): Batch dictionary containing 'imgsz', 'ori_shape', 'ratio_pad', and 'im_file'.
        """
        stem = Path(pbatch["im_file"]).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = predn["bboxes"].clone()
        box[..., [0, 2]] *= pbatch["ori_shape"][1] / self.args.imgsz  # native-space pred
        box[..., [1, 3]] *= pbatch["ori_shape"][0] / self.args.imgsz  # native-space pred
        box = ops.xyxy2xywh(box)  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for b, s, c in zip(box.tolist(), predn["conf"].tolist(), predn["cls"].tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(c)],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(s, 5),
                }
            )
>>>>>>> 0e451281e978a7c5f581cc5d9bb3d0f62e0bc632
