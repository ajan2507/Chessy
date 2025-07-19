<<<<<<< HEAD
# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.engine.results import Results
from ultralytics.models.fastsam.utils import bbox_iou
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class FastSAMPredictor(DetectionPredictor):
    """
    FastSAMPredictor is specialized for fast SAM (Segment Anything Model) segmentation prediction tasks in Ultralytics
    YOLO framework.

    This class extends the DetectionPredictor, customizing the prediction pipeline specifically for fast SAM.
    It adjusts post-processing steps to incorporate mask prediction and non-max suppression while optimizing
    for single-class segmentation.

    Attributes:
        cfg (dict): Configuration parameters for prediction.
        overrides (dict, optional): Optional parameter overrides for custom behavior.
        _callbacks (dict, optional): Optional list of callback functions to be invoked during prediction.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the FastSAMPredictor class, inheriting from DetectionPredictor and setting the task to 'segment'.

        Args:
            cfg (dict): Configuration parameters for prediction.
            overrides (dict, optional): Optional parameter overrides for custom behavior.
            _callbacks (dict, optional): Optional list of callback functions to be invoked during prediction.
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = 'segment'

    def postprocess(self, preds, img, orig_imgs):
        """
        Perform post-processing steps on predictions, including non-max suppression and scaling boxes to original image
        size, and returns the final results.

        Args:
            preds (list): The raw output predictions from the model.
            img (torch.Tensor): The processed image tensor.
            orig_imgs (list | torch.Tensor): The original image or list of images.

        Returns:
            (list): A list of Results objects, each containing processed boxes, masks, and other metadata.
        """
        p = ops.non_max_suppression(
            preds[0],
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=1,  # set to 1 class since SAM has no class predictions
            classes=self.args.classes)
        full_box = torch.zeros(p[0].shape[1], device=p[0].device)
        full_box[2], full_box[3], full_box[4], full_box[6:] = img.shape[3], img.shape[2], 1.0, 1.0
        full_box = full_box.view(1, -1)
        critical_iou_index = bbox_iou(full_box[0][:4], p[0][:, :4], iou_thres=0.9, image_shape=img.shape[2:])
        if critical_iou_index.numel() != 0:
            full_box[0][4] = p[0][critical_iou_index][:, 4]
            full_box[0][6:] = p[0][critical_iou_index][:, 6:]
            p[0][critical_iou_index] = full_box

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # second output is len 3 if pt, but only 1 if exported
        for i, pred in enumerate(p):
            orig_img = orig_imgs[i]
            img_path = self.batch[0][i]
            if not len(pred):  # save empty boxes
                masks = None
            elif self.args.retina_masks:
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
            else:
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks))
        return results
=======
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
from PIL import Image

from ultralytics.models.yolo.segment import SegmentationPredictor
from ultralytics.utils import DEFAULT_CFG, checks
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.ops import scale_masks

from .utils import adjust_bboxes_to_image_border


class FastSAMPredictor(SegmentationPredictor):
    """
    FastSAMPredictor is specialized for fast SAM (Segment Anything Model) segmentation prediction tasks.

    This class extends the SegmentationPredictor, customizing the prediction pipeline specifically for fast SAM. It
    adjusts post-processing steps to incorporate mask prediction and non-maximum suppression while optimizing for
    single-class segmentation.

    Attributes:
        prompts (dict): Dictionary containing prompt information for segmentation (bboxes, points, labels, texts).
        device (torch.device): Device on which model and tensors are processed.
        clip_model (Any, optional): CLIP model for text-based prompting, loaded on demand.
        clip_preprocess (Any, optional): CLIP preprocessing function for images, loaded on demand.

    Methods:
        postprocess: Apply postprocessing to FastSAM predictions and handle prompts.
        prompt: Perform image segmentation inference based on various prompt types.
        set_prompts: Set prompts to be used during inference.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize the FastSAMPredictor with configuration and callbacks.

        This initializes a predictor specialized for Fast SAM (Segment Anything Model) segmentation tasks. The predictor
        extends SegmentationPredictor with custom post-processing for mask prediction and non-maximum suppression
        optimized for single-class segmentation.

        Args:
            cfg (dict): Configuration for the predictor.
            overrides (dict, optional): Configuration overrides.
            _callbacks (list, optional): List of callback functions.
        """
        super().__init__(cfg, overrides, _callbacks)
        self.prompts = {}

    def postprocess(self, preds, img, orig_imgs):
        """
        Apply postprocessing to FastSAM predictions and handle prompts.

        Args:
            preds (List[torch.Tensor]): Raw predictions from the model.
            img (torch.Tensor): Input image tensor that was fed to the model.
            orig_imgs (List[np.ndarray]): Original images before preprocessing.

        Returns:
            (List[Results]): Processed results with prompts applied.
        """
        bboxes = self.prompts.pop("bboxes", None)
        points = self.prompts.pop("points", None)
        labels = self.prompts.pop("labels", None)
        texts = self.prompts.pop("texts", None)
        results = super().postprocess(preds, img, orig_imgs)
        for result in results:
            full_box = torch.tensor(
                [0, 0, result.orig_shape[1], result.orig_shape[0]], device=preds[0].device, dtype=torch.float32
            )
            boxes = adjust_bboxes_to_image_border(result.boxes.xyxy, result.orig_shape)
            idx = torch.nonzero(box_iou(full_box[None], boxes) > 0.9).flatten()
            if idx.numel() != 0:
                result.boxes.xyxy[idx] = full_box

        return self.prompt(results, bboxes=bboxes, points=points, labels=labels, texts=texts)

    def prompt(self, results, bboxes=None, points=None, labels=None, texts=None):
        """
        Perform image segmentation inference based on cues like bounding boxes, points, and text prompts.

        Args:
            results (Results | List[Results]): Original inference results from FastSAM models without any prompts.
            bboxes (np.ndarray | List, optional): Bounding boxes with shape (N, 4), in XYXY format.
            points (np.ndarray | List, optional): Points indicating object locations with shape (N, 2), in pixels.
            labels (np.ndarray | List, optional): Labels for point prompts, shape (N, ). 1 = foreground, 0 = background.
            texts (str | List[str], optional): Textual prompts, a list containing string objects.

        Returns:
            (List[Results]): Output results filtered and determined by the provided prompts.
        """
        if bboxes is None and points is None and texts is None:
            return results
        prompt_results = []
        if not isinstance(results, list):
            results = [results]
        for result in results:
            if len(result) == 0:
                prompt_results.append(result)
                continue
            masks = result.masks.data
            if masks.shape[1:] != result.orig_shape:
                masks = scale_masks(masks[None], result.orig_shape)[0]
            # bboxes prompt
            idx = torch.zeros(len(result), dtype=torch.bool, device=self.device)
            if bboxes is not None:
                bboxes = torch.as_tensor(bboxes, dtype=torch.int32, device=self.device)
                bboxes = bboxes[None] if bboxes.ndim == 1 else bboxes
                bbox_areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
                mask_areas = torch.stack([masks[:, b[1] : b[3], b[0] : b[2]].sum(dim=(1, 2)) for b in bboxes])
                full_mask_areas = torch.sum(masks, dim=(1, 2))

                union = bbox_areas[:, None] + full_mask_areas - mask_areas
                idx[torch.argmax(mask_areas / union, dim=1)] = True
            if points is not None:
                points = torch.as_tensor(points, dtype=torch.int32, device=self.device)
                points = points[None] if points.ndim == 1 else points
                if labels is None:
                    labels = torch.ones(points.shape[0])
                labels = torch.as_tensor(labels, dtype=torch.int32, device=self.device)
                assert len(labels) == len(points), (
                    f"Expected `labels` with same size as `point`, but got {len(labels)} and {len(points)}"
                )
                point_idx = (
                    torch.ones(len(result), dtype=torch.bool, device=self.device)
                    if labels.sum() == 0  # all negative points
                    else torch.zeros(len(result), dtype=torch.bool, device=self.device)
                )
                for point, label in zip(points, labels):
                    point_idx[torch.nonzero(masks[:, point[1], point[0]], as_tuple=True)[0]] = bool(label)
                idx |= point_idx
            if texts is not None:
                if isinstance(texts, str):
                    texts = [texts]
                crop_ims, filter_idx = [], []
                for i, b in enumerate(result.boxes.xyxy.tolist()):
                    x1, y1, x2, y2 = (int(x) for x in b)
                    if masks[i].sum() <= 100:
                        filter_idx.append(i)
                        continue
                    crop_ims.append(Image.fromarray(result.orig_img[y1:y2, x1:x2, ::-1]))
                similarity = self._clip_inference(crop_ims, texts)
                text_idx = torch.argmax(similarity, dim=-1)  # (M, )
                if len(filter_idx):
                    text_idx += (torch.tensor(filter_idx, device=self.device)[None] <= int(text_idx)).sum(0)
                idx[text_idx] = True

            prompt_results.append(result[idx])

        return prompt_results

    def _clip_inference(self, images, texts):
        """
        Perform CLIP inference to calculate similarity between images and text prompts.

        Args:
            images (List[PIL.Image]): List of source images, each should be PIL.Image with RGB channel order.
            texts (List[str]): List of prompt texts, each should be a string object.

        Returns:
            (torch.Tensor): Similarity matrix between given images and texts with shape (M, N).
        """
        try:
            import clip
        except ImportError:
            checks.check_requirements("git+https://github.com/ultralytics/CLIP.git")
            import clip
        if (not hasattr(self, "clip_model")) or (not hasattr(self, "clip_preprocess")):
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        images = torch.stack([self.clip_preprocess(image).to(self.device) for image in images])
        tokenized_text = clip.tokenize(texts).to(self.device)
        image_features = self.clip_model.encode_image(images)
        text_features = self.clip_model.encode_text(tokenized_text)
        image_features /= image_features.norm(dim=-1, keepdim=True)  # (N, 512)
        text_features /= text_features.norm(dim=-1, keepdim=True)  # (M, 512)
        return (image_features * text_features[:, None]).sum(-1)  # (M, N)

    def set_prompts(self, prompts):
        """Set prompts to be used during inference."""
        self.prompts = prompts
>>>>>>> 0e451281e978a7c5f581cc5d9bb3d0f62e0bc632
