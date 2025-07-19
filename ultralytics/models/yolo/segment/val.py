<<<<<<< HEAD
# Ultralytics YOLO 🚀, AGPL-3.0 license

from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, NUM_THREADS, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import SegmentMetrics, box_iou, mask_iou
from ultralytics.utils.plotting import output_to_target, plot_images


class SegmentationValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on a segmentation model.

    Example:
        ```python
        from ultralytics.models.yolo.segment import SegmentationValidator

        args = dict(model='yolov8n-seg.pt', data='coco8-seg.yaml')
        validator = SegmentationValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize SegmentationValidator and set task to 'segment', metrics to SegmentMetrics."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.plot_masks = None
        self.process = None
        self.args.task = 'segment'
        self.metrics = SegmentMetrics(save_dir=self.save_dir, on_plot=self.on_plot)

    def preprocess(self, batch):
        """Preprocesses batch by converting masks to float and sending to device."""
        batch = super().preprocess(batch)
        batch['masks'] = batch['masks'].to(self.device).float()
        return batch

    def init_metrics(self, model):
        """Initialize metrics and select mask processing function based on save_json flag."""
        super().init_metrics(model)
        self.plot_masks = []
        if self.args.save_json:
            check_requirements('pycocotools>=2.0.6')
            self.process = ops.process_mask_upsample  # more accurate
        else:
            self.process = ops.process_mask  # faster

    def get_desc(self):
        """Return a formatted description of evaluation metrics."""
        return ('%22s' + '%11s' * 10) % ('Class', 'Images', 'Instances', 'Box(P', 'R', 'mAP50', 'mAP50-95)', 'Mask(P',
                                         'R', 'mAP50', 'mAP50-95)')

    def postprocess(self, preds):
        """Post-processes YOLO predictions and returns output detections with proto."""
        p = ops.non_max_suppression(preds[0],
                                    self.args.conf,
                                    self.args.iou,
                                    labels=self.lb,
                                    multi_label=True,
                                    agnostic=self.args.single_cls,
                                    max_det=self.args.max_det,
                                    nc=self.nc)
        proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # second output is len 3 if pt, but only 1 if exported
        return p, proto

    def update_metrics(self, preds, batch):
        """Metrics."""
        for si, (pred, proto) in enumerate(zip(preds[0], preds[1])):
            idx = batch['batch_idx'] == si
            cls = batch['cls'][idx]
            bbox = batch['bboxes'][idx]
            nl, npr = cls.shape[0], pred.shape[0]  # number of labels, predictions
            shape = batch['ori_shape'][si]
            correct_masks = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)  # init
            correct_bboxes = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)  # init
            self.seen += 1

            if npr == 0:
                if nl:
                    self.stats.append((correct_bboxes, correct_masks, *torch.zeros(
                        (2, 0), device=self.device), cls.squeeze(-1)))
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, labels=cls.squeeze(-1))
                continue

            # Masks
            midx = [si] if self.args.overlap_mask else idx
            gt_masks = batch['masks'][midx]
            pred_masks = self.process(proto, pred[:, 6:], pred[:, :4], shape=batch['img'][si].shape[1:])

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            ops.scale_boxes(batch['img'][si].shape[1:], predn[:, :4], shape,
                            ratio_pad=batch['ratio_pad'][si])  # native-space pred

            # Evaluate
            if nl:
                height, width = batch['img'].shape[2:]
                tbox = ops.xywh2xyxy(bbox) * torch.tensor(
                    (width, height, width, height), device=self.device)  # target boxes
                ops.scale_boxes(batch['img'][si].shape[1:], tbox, shape,
                                ratio_pad=batch['ratio_pad'][si])  # native-space labels
                labelsn = torch.cat((cls, tbox), 1)  # native-space labels
                correct_bboxes = self._process_batch(predn, labelsn)
                # TODO: maybe remove these `self.` arguments as they already are member variable
                correct_masks = self._process_batch(predn,
                                                    labelsn,
                                                    pred_masks,
                                                    gt_masks,
                                                    overlap=self.args.overlap_mask,
                                                    masks=True)
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, labelsn)

            # Append correct_masks, correct_boxes, pconf, pcls, tcls
            self.stats.append((correct_bboxes, correct_masks, pred[:, 4], pred[:, 5], cls.squeeze(-1)))

            pred_masks = torch.as_tensor(pred_masks, dtype=torch.uint8)
            if self.args.plots and self.batch_i < 3:
                self.plot_masks.append(pred_masks[:15].cpu())  # filter top 15 to plot

            # Save
            if self.args.save_json:
                pred_masks = ops.scale_image(pred_masks.permute(1, 2, 0).contiguous().cpu().numpy(),
                                             shape,
                                             ratio_pad=batch['ratio_pad'][si])
                self.pred_to_json(predn, batch['im_file'][si], pred_masks)
            # if self.args.save_txt:
            #    save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')

    def finalize_metrics(self, *args, **kwargs):
        """Sets speed and confusion matrix for evaluation metrics."""
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def _process_batch(self, detections, labels, pred_masks=None, gt_masks=None, overlap=False, masks=False):
        """
        Return correct prediction matrix.

        Args:
            detections (array[N, 6]), x1, y1, x2, y2, conf, class
            labels (array[M, 5]), class, x1, y1, x2, y2

        Returns:
            correct (array[N, 10]), for 10 IoU levels
        """
        if masks:
            if overlap:
                nl = len(labels)
                index = torch.arange(nl, device=gt_masks.device).view(nl, 1, 1) + 1
                gt_masks = gt_masks.repeat(nl, 1, 1)  # shape(1,640,640) -> (n,640,640)
                gt_masks = torch.where(gt_masks == index, 1.0, 0.0)
            if gt_masks.shape[1:] != pred_masks.shape[1:]:
                gt_masks = F.interpolate(gt_masks[None], pred_masks.shape[1:], mode='bilinear', align_corners=False)[0]
                gt_masks = gt_masks.gt_(0.5)
            iou = mask_iou(gt_masks.view(gt_masks.shape[0], -1), pred_masks.view(pred_masks.shape[0], -1))
        else:  # boxes
            iou = box_iou(labels[:, 1:], detections[:, :4])

        return self.match_predictions(detections[:, 5], labels[:, 0], iou)

    def plot_val_samples(self, batch, ni):
        """Plots validation samples with bounding box labels."""
        plot_images(batch['img'],
                    batch['batch_idx'],
                    batch['cls'].squeeze(-1),
                    batch['bboxes'],
                    batch['masks'],
                    paths=batch['im_file'],
                    fname=self.save_dir / f'val_batch{ni}_labels.jpg',
                    names=self.names,
                    on_plot=self.on_plot)

    def plot_predictions(self, batch, preds, ni):
        """Plots batch predictions with masks and bounding boxes."""
        plot_images(
            batch['img'],
            *output_to_target(preds[0], max_det=15),  # not set to self.args.max_det due to slow plotting speed
            torch.cat(self.plot_masks, dim=0) if len(self.plot_masks) else self.plot_masks,
            paths=batch['im_file'],
            fname=self.save_dir / f'val_batch{ni}_pred.jpg',
            names=self.names,
            on_plot=self.on_plot)  # pred
        self.plot_masks.clear()

    def pred_to_json(self, predn, filename, pred_masks):
        """Save one JSON result."""
        # Example result = {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
        from pycocotools.mask import encode  # noqa

        def single_encode(x):
            """Encode predicted masks as RLE and append results to jdict."""
            rle = encode(np.asarray(x[:, :, None], order='F', dtype='uint8'))[0]
            rle['counts'] = rle['counts'].decode('utf-8')
            return rle

        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        pred_masks = np.transpose(pred_masks, (2, 0, 1))
        with ThreadPool(NUM_THREADS) as pool:
            rles = pool.map(single_encode, pred_masks)
        for i, (p, b) in enumerate(zip(predn.tolist(), box.tolist())):
            self.jdict.append({
                'image_id': image_id,
                'category_id': self.class_map[int(p[5])],
                'bbox': [round(x, 3) for x in b],
                'score': round(p[4], 5),
                'segmentation': rles[i]})

    def eval_json(self, stats):
        """Return COCO-style object detection evaluation metrics."""
        if self.args.save_json and self.is_coco and len(self.jdict):
            anno_json = self.data['path'] / 'annotations/instances_val2017.json'  # annotations
            pred_json = self.save_dir / 'predictions.json'  # predictions
            LOGGER.info(f'\nEvaluating pycocotools mAP using {pred_json} and {anno_json}...')
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                check_requirements('pycocotools>=2.0.6')
                from pycocotools.coco import COCO  # noqa
                from pycocotools.cocoeval import COCOeval  # noqa

                for x in anno_json, pred_json:
                    assert x.is_file(), f'{x} file not found'
                anno = COCO(str(anno_json))  # init annotations api
                pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
                for i, eval in enumerate([COCOeval(anno, pred, 'bbox'), COCOeval(anno, pred, 'segm')]):
                    if self.is_coco:
                        eval.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # im to eval
                    eval.evaluate()
                    eval.accumulate()
                    eval.summarize()
                    idx = i * 4 + 2
                    stats[self.metrics.keys[idx + 1]], stats[
                        self.metrics.keys[idx]] = eval.stats[:2]  # update mAP50-95 and mAP50
            except Exception as e:
                LOGGER.warning(f'pycocotools unable to run: {e}')
        return stats
=======
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, NUM_THREADS, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import SegmentMetrics, mask_iou


class SegmentationValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on a segmentation model.

    This validator handles the evaluation of segmentation models, processing both bounding box and mask predictions
    to compute metrics such as mAP for both detection and segmentation tasks.

    Attributes:
        plot_masks (list): List to store masks for plotting.
        process (callable): Function to process masks based on save_json and save_txt flags.
        args (namespace): Arguments for the validator.
        metrics (SegmentMetrics): Metrics calculator for segmentation tasks.
        stats (dict): Dictionary to store statistics during validation.

    Examples:
        >>> from ultralytics.models.yolo.segment import SegmentationValidator
        >>> args = dict(model="yolo11n-seg.pt", data="coco8-seg.yaml")
        >>> validator = SegmentationValidator(args=args)
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """
        Initialize SegmentationValidator and set task to 'segment', metrics to SegmentMetrics.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to use for validation.
            save_dir (Path, optional): Directory to save results.
            args (namespace, optional): Arguments for the validator.
            _callbacks (list, optional): List of callback functions.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.process = None
        self.args.task = "segment"
        self.metrics = SegmentMetrics()

    def preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess batch of images for YOLO segmentation validation.

        Args:
            batch (Dict[str, Any]): Batch containing images and annotations.

        Returns:
            (Dict[str, Any]): Preprocessed batch.
        """
        batch = super().preprocess(batch)
        batch["masks"] = batch["masks"].to(self.device).float()
        return batch

    def init_metrics(self, model: torch.nn.Module) -> None:
        """
        Initialize metrics and select mask processing function based on save_json flag.

        Args:
            model (torch.nn.Module): Model to validate.
        """
        super().init_metrics(model)
        if self.args.save_json:
            check_requirements("faster-coco-eval>=1.6.7")
        # More accurate vs faster
        self.process = ops.process_mask_native if self.args.save_json or self.args.save_txt else ops.process_mask

    def get_desc(self) -> str:
        """Return a formatted description of evaluation metrics."""
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Mask(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def postprocess(self, preds: List[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """
        Post-process YOLO predictions and return output detections with proto.

        Args:
            preds (List[torch.Tensor]): Raw predictions from the model.

        Returns:
            List[Dict[str, torch.Tensor]]: Processed detection predictions with masks.
        """
        proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # second output is len 3 if pt, but only 1 if exported
        preds = super().postprocess(preds[0])
        imgsz = [4 * x for x in proto.shape[2:]]  # get image size from proto
        for i, pred in enumerate(preds):
            coefficient = pred.pop("extra")
            pred["masks"] = (
                self.process(proto[i], coefficient, pred["bboxes"], shape=imgsz)
                if len(coefficient)
                else torch.zeros(
                    (0, *(imgsz if self.process is ops.process_mask_native else proto.shape[2:])),
                    dtype=torch.uint8,
                    device=pred["bboxes"].device,
                )
            )
        return preds

    def _prepare_batch(self, si: int, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare a batch for training or inference by processing images and targets.

        Args:
            si (int): Batch index.
            batch (Dict[str, Any]): Batch data containing images and annotations.

        Returns:
            (Dict[str, Any]): Prepared batch with processed annotations.
        """
        prepared_batch = super()._prepare_batch(si, batch)
        midx = [si] if self.args.overlap_mask else batch["batch_idx"] == si
        prepared_batch["masks"] = batch["masks"][midx]
        return prepared_batch

    def _process_batch(self, preds: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Compute correct prediction matrix for a batch based on bounding boxes and optional masks.

        Args:
            preds (Dict[str, torch.Tensor]): Dictionary containing predictions with keys like 'cls' and 'masks'.
            batch (Dict[str, Any]): Dictionary containing batch data with keys like 'cls' and 'masks'.

        Returns:
            (Dict[str, np.ndarray]): A dictionary containing correct prediction matrices including 'tp_m' for mask IoU.

        Notes:
            - If `masks` is True, the function computes IoU between predicted and ground truth masks.
            - If `overlap` is True and `masks` is True, overlapping masks are taken into account when computing IoU.

        Examples:
            >>> preds = {"cls": torch.tensor([1, 0]), "masks": torch.rand(2, 640, 640), "bboxes": torch.rand(2, 4)}
            >>> batch = {"cls": torch.tensor([1, 0]), "masks": torch.rand(2, 640, 640), "bboxes": torch.rand(2, 4)}
            >>> correct_preds = validator._process_batch(preds, batch)
        """
        tp = super()._process_batch(preds, batch)
        gt_cls, gt_masks = batch["cls"], batch["masks"]
        if len(gt_cls) == 0 or len(preds["cls"]) == 0:
            tp_m = np.zeros((len(preds["cls"]), self.niou), dtype=bool)
        else:
            pred_masks = preds["masks"]
            if self.args.overlap_mask:
                nl = len(gt_cls)
                index = torch.arange(nl, device=gt_masks.device).view(nl, 1, 1) + 1
                gt_masks = gt_masks.repeat(nl, 1, 1)  # shape(1,640,640) -> (n,640,640)
                gt_masks = torch.where(gt_masks == index, 1.0, 0.0)
            if gt_masks.shape[1:] != pred_masks.shape[1:]:
                gt_masks = F.interpolate(gt_masks[None], pred_masks.shape[1:], mode="bilinear", align_corners=False)[0]
                gt_masks = gt_masks.gt_(0.5)
            iou = mask_iou(gt_masks.view(gt_masks.shape[0], -1), pred_masks.view(pred_masks.shape[0], -1))
            tp_m = self.match_predictions(preds["cls"], gt_cls, iou).cpu().numpy()
        tp.update({"tp_m": tp_m})  # update tp with mask IoU
        return tp

    def plot_predictions(self, batch: Dict[str, Any], preds: List[Dict[str, torch.Tensor]], ni: int) -> None:
        """
        Plot batch predictions with masks and bounding boxes.

        Args:
            batch (Dict[str, Any]): Batch containing images and annotations.
            preds (List[Dict[str, torch.Tensor]]): List of predictions from the model.
            ni (int): Batch index.
        """
        for p in preds:
            masks = p["masks"]
            if masks.shape[0] > 50:
                LOGGER.warning("Limiting validation plots to first 50 items per image for speed...")
            p["masks"] = torch.as_tensor(masks[:50], dtype=torch.uint8).cpu()
        super().plot_predictions(batch, preds, ni, max_det=50)  # plot bboxes

    def save_one_txt(self, predn: torch.Tensor, save_conf: bool, shape: Tuple[int, int], file: Path) -> None:
        """
        Save YOLO detections to a txt file in normalized coordinates in a specific format.

        Args:
            predn (torch.Tensor): Predictions in the format (x1, y1, x2, y2, conf, class).
            save_conf (bool): Whether to save confidence scores.
            shape (Tuple[int, int]): Shape of the original image.
            file (Path): File path to save the detections.
        """
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
            masks=torch.as_tensor(predn["masks"], dtype=torch.uint8),
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn: Dict[str, torch.Tensor], pbatch: Dict[str, Any]) -> None:
        """
        Save one JSON result for COCO evaluation.

        Args:
            predn (Dict[str, torch.Tensor]): Predictions containing bboxes, masks, confidence scores, and classes.
            pbatch (Dict[str, Any]): Batch dictionary containing 'imgsz', 'ori_shape', 'ratio_pad', and 'im_file'.

        Examples:
             >>> result = {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
        """
        from faster_coco_eval.core.mask import encode  # noqa

        def single_encode(x):
            """Encode predicted masks as RLE and append results to jdict."""
            rle = encode(np.asarray(x[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            return rle

        coco_masks = torch.as_tensor(predn["masks"], dtype=torch.uint8)
        coco_masks = ops.scale_image(
            coco_masks.permute(1, 2, 0).contiguous().cpu().numpy(),
            pbatch["ori_shape"],
            ratio_pad=pbatch["ratio_pad"],
        )
        pred_masks = np.transpose(coco_masks, (2, 0, 1))
        with ThreadPool(NUM_THREADS) as pool:
            rles = pool.map(single_encode, pred_masks)
        super().pred_to_json(predn, pbatch)
        for i, r in enumerate(rles):
            self.jdict[-len(rles) + i]["segmentation"] = r  # segmentation

    def eval_json(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Return COCO-style instance segmentation evaluation metrics."""
        pred_json = self.save_dir / "predictions.json"  # predictions
        anno_json = (
            self.data["path"]
            / "annotations"
            / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
        )  # annotations
        return super().coco_evaluate(stats, pred_json, anno_json, ["bbox", "segm"], suffix=["Box", "Mask"])
>>>>>>> 0e451281e978a7c5f581cc5d9bb3d0f62e0bc632
