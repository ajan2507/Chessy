<<<<<<< HEAD
---
comments: true
description: Learn about YOLOv8 Classify models for image classification. Get detailed information on List of Pretrained Models & how to Train, Validate, Predict & Export models.
keywords: Ultralytics, YOLOv8, Image Classification, Pretrained Models, YOLOv8n-cls, Training, Validation, Prediction, Model Export
---

# Image Classification

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418606-adf35c62-2e11-405d-84c6-b84e7d013804.png" alt="Image classification examples">

Image classification is the simplest of the three tasks and involves classifying an entire image into one of a set of predefined classes.

The output of an image classifier is a single class label and a confidence score. Image classification is useful when you need to know only what class an image belongs to and don't need to know where objects of that class are located or what their exact shape is.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/NAs-cfq9BDw?start=169"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Explore Ultralytics YOLO Tasks: Image Classification
</p>

!!! Tip "Tip"

    YOLOv8 Classify models use the `-cls` suffix, i.e. `yolov8n-cls.pt` and are pretrained on [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml).

## [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

YOLOv8 pretrained Classify models are shown here. Detect, Segment and Pose models are pretrained on the [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) dataset, while Classify models are pretrained on the [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) dataset.

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) download automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.

| Model                                                                                        | size<br><sup>(pixels) | acc<br><sup>top1 | acc<br><sup>top5 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) at 640 |
|----------------------------------------------------------------------------------------------|-----------------------|------------------|------------------|--------------------------------|-------------------------------------|--------------------|--------------------------|
| [YOLOv8n-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-cls.pt) | 224                   | 66.6             | 87.0             | 12.9                           | 0.31                                | 2.7                | 4.3                      |
| [YOLOv8s-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-cls.pt) | 224                   | 72.3             | 91.1             | 23.4                           | 0.35                                | 6.4                | 13.5                     |
| [YOLOv8m-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-cls.pt) | 224                   | 76.4             | 93.2             | 85.4                           | 0.62                                | 17.0               | 42.7                     |
| [YOLOv8l-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-cls.pt) | 224                   | 78.0             | 94.1             | 163.0                          | 0.87                                | 37.5               | 99.7                     |
| [YOLOv8x-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-cls.pt) | 224                   | 78.4             | 94.3             | 232.0                          | 1.01                                | 57.4               | 154.8                    |

- **acc** values are model accuracies on the [ImageNet](https://www.image-net.org/) dataset validation set.
  <br>Reproduce by `yolo val classify data=path/to/ImageNet device=0`
- **Speed** averaged over ImageNet val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/)
  instance.
  <br>Reproduce by `yolo val classify data=path/to/ImageNet batch=1 device=0|cpu`

## Train

Train YOLOv8n-cls on the MNIST160 dataset for 100 epochs at image size 64. For a full list of available arguments see the [Configuration](../usage/cfg.md) page.

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n-cls.yaml')  # build a new model from YAML
        model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)
        model = YOLO('yolov8n-cls.yaml').load('yolov8n-cls.pt')  # build from YAML and transfer weights

        # Train the model
        results = model.train(data='mnist160', epochs=100, imgsz=64)
        ```

    === "CLI"

        ```bash
        # Build a new model from YAML and start training from scratch
        yolo classify train data=mnist160 model=yolov8n-cls.yaml epochs=100 imgsz=64

        # Start training from a pretrained *.pt model
        yolo classify train data=mnist160 model=yolov8n-cls.pt epochs=100 imgsz=64

        # Build a new model from YAML, transfer pretrained weights to it and start training
        yolo classify train data=mnist160 model=yolov8n-cls.yaml pretrained=yolov8n-cls.pt epochs=100 imgsz=64
        ```

### Dataset format

YOLO classification dataset format can be found in detail in the [Dataset Guide](../datasets/classify/index.md).

## Val

Validate trained YOLOv8n-cls model accuracy on the MNIST160 dataset. No argument need to passed as the `model` retains it's training `data` and arguments as model attributes.

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n-cls.pt')  # load an official model
        model = YOLO('path/to/best.pt')  # load a custom model

        # Validate the model
        metrics = model.val()  # no arguments needed, dataset and settings remembered
        metrics.top1   # top1 accuracy
        metrics.top5   # top5 accuracy
        ```
    === "CLI"

        ```bash
        yolo classify val model=yolov8n-cls.pt  # val official model
        yolo classify val model=path/to/best.pt  # val custom model
        ```

## Predict

Use a trained YOLOv8n-cls model to run predictions on images.

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n-cls.pt')  # load an official model
        model = YOLO('path/to/best.pt')  # load a custom model

        # Predict with the model
        results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image
        ```
    === "CLI"

        ```bash
        yolo classify predict model=yolov8n-cls.pt source='https://ultralytics.com/images/bus.jpg'  # predict with official model
        yolo classify predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # predict with custom model
        ```

See full `predict` mode details in the [Predict](https://docs.ultralytics.com/modes/predict/) page.

## Export

Export a YOLOv8n-cls model to a different format like ONNX, CoreML, etc.

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n-cls.pt')  # load an official model
        model = YOLO('path/to/best.pt')  # load a custom trained model

        # Export the model
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n-cls.pt format=onnx  # export official model
        yolo export model=path/to/best.pt format=onnx  # export custom trained model
        ```

Available YOLOv8-cls export formats are in the table below. You can predict or validate directly on exported models, i.e. `yolo predict model=yolov8n-cls.onnx`. Usage examples are shown for your model after export completes.

| Format                                                             | `format` Argument | Model                         | Metadata | Arguments                                           |
|--------------------------------------------------------------------|-------------------|-------------------------------|----------|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -                 | `yolov8n-cls.pt`              | ✅        | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript`     | `yolov8n-cls.torchscript`     | ✅        | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`            | `yolov8n-cls.onnx`            | ✅        | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`        | `yolov8n-cls_openvino_model/` | ✅        | `imgsz`, `half`                                     |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`          | `yolov8n-cls.engine`          | ✅        | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`          | `yolov8n-cls.mlpackage`       | ✅        | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`     | `yolov8n-cls_saved_model/`    | ✅        | `imgsz`, `keras`                                    |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`              | `yolov8n-cls.pb`              | ❌        | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`          | `yolov8n-cls.tflite`          | ✅        | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`         | `yolov8n-cls_edgetpu.tflite`  | ✅        | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`            | `yolov8n-cls_web_model/`      | ✅        | `imgsz`, `half`, `int8`                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`          | `yolov8n-cls_paddle_model/`   | ✅        | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`            | `yolov8n-cls_ncnn_model/`     | ✅        | `imgsz`, `half`                                     |

See full `export` details in the [Export](https://docs.ultralytics.com/modes/export/) page.
=======
---
comments: true
description: Master image classification using YOLO11. Learn to train, validate, predict, and export models efficiently.
keywords: YOLO11, image classification, AI, machine learning, pretrained models, ImageNet, model export, predict, train, validate
model_name: yolo11n-cls
---

# Image Classification

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/image-classification-examples.avif" alt="Image classification examples">

[Image classification](https://www.ultralytics.com/glossary/image-classification) is the simplest of the three tasks and involves classifying an entire image into one of a set of predefined classes.

The output of an image classifier is a single class label and a confidence score. Image classification is useful when you need to know only what class an image belongs to and don't need to know where objects of that class are located or what their exact shape is.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/5BO0Il_YYAg"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Explore Ultralytics YOLO Tasks: Image Classification using Ultralytics HUB
</p>

!!! tip

    YOLO11 Classify models use the `-cls` suffix, i.e. `yolo11n-cls.pt` and are pretrained on [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml).

## [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/11)

YOLO11 pretrained Classify models are shown here. Detect, Segment and Pose models are pretrained on the [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) dataset, while Classify models are pretrained on the [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) dataset.

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) download automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.

{% include "macros/yolo-cls-perf.md" %}

- **acc** values are model accuracies on the [ImageNet](https://www.image-net.org/) dataset validation set. <br>Reproduce by `yolo val classify data=path/to/ImageNet device=0`
- **Speed** averaged over ImageNet val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. <br>Reproduce by `yolo val classify data=path/to/ImageNet batch=1 device=0|cpu`

## Train

Train YOLO11n-cls on the MNIST160 dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) at image size 64. For a full list of available arguments see the [Configuration](../usage/cfg.md) page.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-cls.yaml")  # build a new model from YAML
        model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)
        model = YOLO("yolo11n-cls.yaml").load("yolo11n-cls.pt")  # build from YAML and transfer weights

        # Train the model
        results = model.train(data="mnist160", epochs=100, imgsz=64)
        ```

    === "CLI"

        ```bash
        # Build a new model from YAML and start training from scratch
        yolo classify train data=mnist160 model=yolo11n-cls.yaml epochs=100 imgsz=64

        # Start training from a pretrained *.pt model
        yolo classify train data=mnist160 model=yolo11n-cls.pt epochs=100 imgsz=64

        # Build a new model from YAML, transfer pretrained weights to it and start training
        yolo classify train data=mnist160 model=yolo11n-cls.yaml pretrained=yolo11n-cls.pt epochs=100 imgsz=64
        ```

!!! tip

    Ultralytics YOLO classification uses [`torchvision.transforms.RandomResizedCrop`](https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomResizedCrop.html) for training and [`torchvision.transforms.CenterCrop`](https://pytorch.org/vision/stable/generated/torchvision.transforms.CenterCrop.html) for validation and inference.
    These cropping-based transforms assume square inputs and may inadvertently crop out important regions from images with extreme aspect ratios, potentially causing loss of critical visual information during training.
    To preserve the full image while maintaining its proportions, consider using [`torchvision.transforms.Resize`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Resize.html) instead of cropping transforms.

    You can implement this by customizing your augmentation pipeline through a custom `ClassificationDataset` and `ClassificationTrainer`.


    ```python
    import torch
    import torchvision.transforms as T

    from ultralytics import YOLO
    from ultralytics.data.dataset import ClassificationDataset
    from ultralytics.models.yolo.classify import ClassificationTrainer


    class CustomizedDataset(ClassificationDataset):
        """A customized dataset class for image classification with enhanced data augmentation transforms."""

        def __init__(self, root: str, args, augment: bool = False, prefix: str = ""):
            """Initialize a customized classification dataset with enhanced data augmentation transforms."""
            super().__init__(root, args, augment, prefix)

            # Add your custom training transforms here
            train_transforms = T.Compose(
                [
                    T.Resize((args.imgsz, args.imgsz)),
                    T.RandomHorizontalFlip(p=args.fliplr),
                    T.RandomVerticalFlip(p=args.flipud),
                    T.RandAugment(interpolation=T.InterpolationMode.BILINEAR),
                    T.ColorJitter(brightness=args.hsv_v, contrast=args.hsv_v, saturation=args.hsv_s, hue=args.hsv_h),
                    T.ToTensor(),
                    T.Normalize(mean=torch.tensor(0), std=torch.tensor(1)),
                    T.RandomErasing(p=args.erasing, inplace=True),
                ]
            )

            # Add your custom validation transforms here
            val_transforms = T.Compose(
                [
                    T.Resize((args.imgsz, args.imgsz)),
                    T.ToTensor(),
                    T.Normalize(mean=torch.tensor(0), std=torch.tensor(1)),
                ]
            )
            self.torch_transforms = train_transforms if augment else val_transforms


    class CustomizedTrainer(ClassificationTrainer):
        """A customized trainer class for YOLO classification models with enhanced dataset handling."""

        def build_dataset(self, img_path: str, mode: str = "train", batch=None):
            """Build a customized dataset for classification training and the validation during training."""
            return CustomizedDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)


    class CustomizedValidator(ClassificationValidator):
        """A customized validator class for YOLO classification models with enhanced dataset handling."""

        def build_dataset(self, img_path: str, mode: str = "train"):
            """Build a customized dataset for classification standalone validation."""
            return CustomizedDataset(root=img_path, args=self.args, augment=mode == "train", prefix=self.args.split)


    model = YOLO("yolo11n-cls.pt")
    model.train(data="imagenet1000", trainer=CustomizedTrainer, epochs=10, imgsz=224, batch=64)
    model.val(data="imagenet1000", validator=CustomizedValidator, imgsz=224, batch=64)
    ```

### Dataset format

YOLO classification dataset format can be found in detail in the [Dataset Guide](../datasets/classify/index.md).

## Val

Validate trained YOLO11n-cls model [accuracy](https://www.ultralytics.com/glossary/accuracy) on the MNIST160 dataset. No arguments are needed as the `model` retains its training `data` and arguments as model attributes.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-cls.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Validate the model
        metrics = model.val()  # no arguments needed, dataset and settings remembered
        metrics.top1  # top1 accuracy
        metrics.top5  # top5 accuracy
        ```

    === "CLI"

        ```bash
        yolo classify val model=yolo11n-cls.pt  # val official model
        yolo classify val model=path/to/best.pt # val custom model
        ```

!!! tip

    As mentioned in the [training section](#train), you can handle extreme aspect ratios during training by using a custom `ClassificationTrainer`. You need to apply the same approach for consistent validation results by implementing a custom `ClassificationValidator` when calling the `val()` method. Refer to the complete code example in the [training section](#train) for implementation details.

## Predict

Use a trained YOLO11n-cls model to run predictions on images.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-cls.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Predict with the model
        results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
        ```

    === "CLI"

        ```bash
        yolo classify predict model=yolo11n-cls.pt source='https://ultralytics.com/images/bus.jpg'  # predict with official model
        yolo classify predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg' # predict with custom model
        ```

See full `predict` mode details in the [Predict](../modes/predict.md) page.

## Export

Export a YOLO11n-cls model to a different format like ONNX, CoreML, etc.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-cls.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom trained model

        # Export the model
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n-cls.pt format=onnx  # export official model
        yolo export model=path/to/best.pt format=onnx # export custom trained model
        ```

Available YOLO11-cls export formats are in the table below. You can export to any format using the `format` argument, i.e. `format='onnx'` or `format='engine'`. You can predict or validate directly on exported models, i.e. `yolo predict model=yolo11n-cls.onnx`. Usage examples are shown for your model after export completes.

{% include "macros/export-table.md" %}

See full `export` details in the [Export](../modes/export.md) page.

## FAQ

### What is the purpose of YOLO11 in image classification?

YOLO11 models, such as `yolo11n-cls.pt`, are designed for efficient image classification. They assign a single class label to an entire image along with a confidence score. This is particularly useful for applications where knowing the specific class of an image is sufficient, rather than identifying the location or shape of objects within the image.

### How do I train a YOLO11 model for image classification?

To train a YOLO11 model, you can use either Python or CLI commands. For example, to train a `yolo11n-cls` model on the MNIST160 dataset for 100 epochs at an image size of 64:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="mnist160", epochs=100, imgsz=64)
        ```

    === "CLI"

        ```bash
        yolo classify train data=mnist160 model=yolo11n-cls.pt epochs=100 imgsz=64
        ```

For more configuration options, visit the [Configuration](../usage/cfg.md) page.

### Where can I find pretrained YOLO11 classification models?

Pretrained YOLO11 classification models can be found in the [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/11) section. Models like `yolo11n-cls.pt`, `yolo11s-cls.pt`, `yolo11m-cls.pt`, etc., are pretrained on the [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) dataset and can be easily downloaded and used for various image classification tasks.

### How can I export a trained YOLO11 model to different formats?

You can export a trained YOLO11 model to various formats using Python or CLI commands. For instance, to export a model to ONNX format:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-cls.pt")  # load the trained model

        # Export the model to ONNX
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n-cls.pt format=onnx # export the trained model to ONNX format
        ```

For detailed export options, refer to the [Export](../modes/export.md) page.

### How do I validate a trained YOLO11 classification model?

To validate a trained model's accuracy on a dataset like MNIST160, you can use the following Python or CLI commands:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-cls.pt")  # load the trained model

        # Validate the model
        metrics = model.val()  # no arguments needed, uses the dataset and settings from training
        metrics.top1  # top1 accuracy
        metrics.top5  # top5 accuracy
        ```

    === "CLI"

        ```bash
        yolo classify val model=yolo11n-cls.pt # validate the trained model
        ```

For more information, visit the [Validate](#val) section.
>>>>>>> 0e451281e978a7c5f581cc5d9bb3d0f62e0bc632
