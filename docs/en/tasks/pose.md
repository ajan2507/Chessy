<<<<<<< HEAD
---
comments: true
description: Learn how to use Ultralytics YOLOv8 for pose estimation tasks. Find pretrained models, learn how to train, validate, predict, and export your own.
keywords: Ultralytics, YOLO, YOLOv8, pose estimation, keypoints detection, object detection, pre-trained models, machine learning, artificial intelligence
---

# Pose Estimation

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418616-9811ac0b-a4a7-452a-8aba-484ba32bb4a8.png" alt="Pose estimation examples">

Pose estimation is a task that involves identifying the location of specific points in an image, usually referred to as keypoints. The keypoints can represent various parts of the object such as joints, landmarks, or other distinctive features. The locations of the keypoints are usually represented as a set of 2D `[x, y]` or 3D `[x, y, visible]`
coordinates.

The output of a pose estimation model is a set of points that represent the keypoints on an object in the image, usually along with the confidence scores for each point. Pose estimation is a good choice when you need to identify specific parts of an object in a scene, and their location in relation to each other.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/Y28xXQmju64?si=pCY4ZwejZFu6Z4kZ"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Pose Estimation with Ultralytics YOLOv8.
</p>

!!! Tip "Tip"

    YOLOv8 _pose_ models use the `-pose` suffix, i.e. `yolov8n-pose.pt`. These models are trained on the [COCO keypoints](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml) dataset and are suitable for a variety of pose estimation tasks.

## [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

YOLOv8 pretrained Pose models are shown here. Detect, Segment and Pose models are pretrained on the [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) dataset, while Classify models are pretrained on the [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) dataset.

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) download automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.

| Model                                                                                                | size<br><sup>(pixels) | mAP<sup>pose<br>50-95 | mAP<sup>pose<br>50 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
|------------------------------------------------------------------------------------------------------|-----------------------|-----------------------|--------------------|--------------------------------|-------------------------------------|--------------------|-------------------|
| [YOLOv8n-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt)       | 640                   | 50.4                  | 80.1               | 131.8                          | 1.18                                | 3.3                | 9.2               |
| [YOLOv8s-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt)       | 640                   | 60.0                  | 86.2               | 233.2                          | 1.42                                | 11.6               | 30.2              |
| [YOLOv8m-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-pose.pt)       | 640                   | 65.0                  | 88.8               | 456.3                          | 2.00                                | 26.4               | 81.0              |
| [YOLOv8l-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-pose.pt)       | 640                   | 67.6                  | 90.0               | 784.5                          | 2.59                                | 44.4               | 168.6             |
| [YOLOv8x-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose.pt)       | 640                   | 69.2                  | 90.2               | 1607.1                         | 3.73                                | 69.4               | 263.2             |
| [YOLOv8x-pose-p6](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose-p6.pt) | 1280                  | 71.6                  | 91.2               | 4088.7                         | 10.04                               | 99.1               | 1066.4            |

- **mAP<sup>val</sup>** values are for single-model single-scale on [COCO Keypoints val2017](http://cocodataset.org)
  dataset.
  <br>Reproduce by `yolo val pose data=coco-pose.yaml device=0`
- **Speed** averaged over COCO val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/)
  instance.
  <br>Reproduce by `yolo val pose data=coco8-pose.yaml batch=1 device=0|cpu`

## Train

Train a YOLOv8-pose model on the COCO128-pose dataset.

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n-pose.yaml')  # build a new model from YAML
        model = YOLO('yolov8n-pose.pt')  # load a pretrained model (recommended for training)
        model = YOLO('yolov8n-pose.yaml').load('yolov8n-pose.pt')  # build from YAML and transfer weights

        # Train the model
        results = model.train(data='coco8-pose.yaml', epochs=100, imgsz=640)
        ```
    === "CLI"

        ```bash
        # Build a new model from YAML and start training from scratch
        yolo pose train data=coco8-pose.yaml model=yolov8n-pose.yaml epochs=100 imgsz=640

        # Start training from a pretrained *.pt model
        yolo pose train data=coco8-pose.yaml model=yolov8n-pose.pt epochs=100 imgsz=640

        # Build a new model from YAML, transfer pretrained weights to it and start training
        yolo pose train data=coco8-pose.yaml model=yolov8n-pose.yaml pretrained=yolov8n-pose.pt epochs=100 imgsz=640
        ```

### Dataset format

YOLO pose dataset format can be found in detail in the [Dataset Guide](../datasets/pose/index.md). To convert your existing dataset from other formats (like COCO etc.) to YOLO format, please use [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) tool by Ultralytics.

## Val

Validate trained YOLOv8n-pose model accuracy on the COCO128-pose dataset. No argument need to passed as the `model`
retains it's training `data` and arguments as model attributes.

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n-pose.pt')  # load an official model
        model = YOLO('path/to/best.pt')  # load a custom model

        # Validate the model
        metrics = model.val()  # no arguments needed, dataset and settings remembered
        metrics.box.map    # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps   # a list contains map50-95 of each category
        ```
    === "CLI"

        ```bash
        yolo pose val model=yolov8n-pose.pt  # val official model
        yolo pose val model=path/to/best.pt  # val custom model
        ```

## Predict

Use a trained YOLOv8n-pose model to run predictions on images.

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n-pose.pt')  # load an official model
        model = YOLO('path/to/best.pt')  # load a custom model

        # Predict with the model
        results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image
        ```
    === "CLI"

        ```bash
        yolo pose predict model=yolov8n-pose.pt source='https://ultralytics.com/images/bus.jpg'  # predict with official model
        yolo pose predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # predict with custom model
        ```

See full `predict` mode details in the [Predict](https://docs.ultralytics.com/modes/predict/) page.

## Export

Export a YOLOv8n Pose model to a different format like ONNX, CoreML, etc.

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n-pose.pt')  # load an official model
        model = YOLO('path/to/best.pt')  # load a custom trained model

        # Export the model
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n-pose.pt format=onnx  # export official model
        yolo export model=path/to/best.pt format=onnx  # export custom trained model
        ```

Available YOLOv8-pose export formats are in the table below. You can predict or validate directly on exported models, i.e. `yolo predict model=yolov8n-pose.onnx`. Usage examples are shown for your model after export completes.

| Format                                                             | `format` Argument | Model                          | Metadata | Arguments                                           |
|--------------------------------------------------------------------|-------------------|--------------------------------|----------|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -                 | `yolov8n-pose.pt`              | ✅        | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript`     | `yolov8n-pose.torchscript`     | ✅        | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`            | `yolov8n-pose.onnx`            | ✅        | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`        | `yolov8n-pose_openvino_model/` | ✅        | `imgsz`, `half`                                     |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`          | `yolov8n-pose.engine`          | ✅        | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`          | `yolov8n-pose.mlpackage`       | ✅        | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`     | `yolov8n-pose_saved_model/`    | ✅        | `imgsz`, `keras`                                    |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`              | `yolov8n-pose.pb`              | ❌        | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`          | `yolov8n-pose.tflite`          | ✅        | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`         | `yolov8n-pose_edgetpu.tflite`  | ✅        | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`            | `yolov8n-pose_web_model/`      | ✅        | `imgsz`, `half`, `int8`                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`          | `yolov8n-pose_paddle_model/`   | ✅        | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`            | `yolov8n-pose_ncnn_model/`     | ✅        | `imgsz`, `half`                                     |

See full `export` details in the [Export](https://docs.ultralytics.com/modes/export/) page.
=======
---
comments: true
description: Discover how to use YOLO11 for pose estimation tasks. Learn about model training, validation, prediction, and exporting in various formats.
keywords: pose estimation, YOLO11, Ultralytics, keypoints, model training, image recognition, deep learning, human pose detection, computer vision, real-time tracking
model_name: yolo11n-pose
---

# Pose Estimation

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/pose-estimation-examples.avif" alt="Pose estimation examples">

Pose estimation is a task that involves identifying the location of specific points in an image, usually referred to as keypoints. The keypoints can represent various parts of the object such as joints, landmarks, or other distinctive features. The locations of the keypoints are usually represented as a set of 2D `[x, y]` or 3D `[x, y, visible]` coordinates.

The output of a pose estimation model is a set of points that represent the keypoints on an object in the image, usually along with the confidence scores for each point. Pose estimation is a good choice when you need to identify specific parts of an object in a scene, and their location in relation to each other.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/AAkfToU3nAc"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Ultralytics YOLO11 Pose Estimation Tutorial | Real-Time Object Tracking and Human Pose Detection
</p>

!!! tip

    YOLO11 _pose_ models use the `-pose` suffix, i.e. `yolo11n-pose.pt`. These models are trained on the [COCO keypoints](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml) dataset and are suitable for a variety of pose estimation tasks.

    In the default YOLO11 pose model, there are 17 keypoints, each representing a different part of the human body. Here is the mapping of each index to its respective body joint:

    0. Nose
    1. Left Eye
    2. Right Eye
    3. Left Ear
    4. Right Ear
    5. Left Shoulder
    6. Right Shoulder
    7. Left Elbow
    8. Right Elbow
    9. Left Wrist
    10. Right Wrist
    11. Left Hip
    12. Right Hip
    13. Left Knee
    14. Right Knee
    15. Left Ankle
    16. Right Ankle

## [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/11)

Ultralytics YOLO11 pretrained Pose models are shown here. Detect, Segment and Pose models are pretrained on the [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) dataset, while Classify models are pretrained on the [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) dataset.

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) download automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.

{% include "macros/yolo-pose-perf.md" %}

- **mAP<sup>val</sup>** values are for single-model single-scale on [COCO Keypoints val2017](https://cocodataset.org/) dataset. <br>Reproduce by `yolo val pose data=coco-pose.yaml device=0`
- **Speed** averaged over COCO val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. <br>Reproduce by `yolo val pose data=coco-pose.yaml batch=1 device=0|cpu`

## Train

Train a YOLO11-pose model on the COCO8-pose dataset. The [COCO8-pose dataset](https://docs.ultralytics.com/datasets/pose/coco8-pose/) is a small sample dataset that's perfect for testing and debugging your pose estimation models.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-pose.yaml")  # build a new model from YAML
        model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)
        model = YOLO("yolo11n-pose.yaml").load("yolo11n-pose.pt")  # build from YAML and transfer weights

        # Train the model
        results = model.train(data="coco8-pose.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Build a new model from YAML and start training from scratch
        yolo pose train data=coco8-pose.yaml model=yolo11n-pose.yaml epochs=100 imgsz=640

        # Start training from a pretrained *.pt model
        yolo pose train data=coco8-pose.yaml model=yolo11n-pose.pt epochs=100 imgsz=640

        # Build a new model from YAML, transfer pretrained weights to it and start training
        yolo pose train data=coco8-pose.yaml model=yolo11n-pose.yaml pretrained=yolo11n-pose.pt epochs=100 imgsz=640
        ```

### Dataset format

YOLO pose dataset format can be found in detail in the [Dataset Guide](../datasets/pose/index.md). To convert your existing dataset from other formats (like [COCO](https://docs.ultralytics.com/datasets/pose/coco/) etc.) to YOLO format, please use the [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) tool by Ultralytics.

For custom pose estimation tasks, you can also explore specialized datasets like [Tiger-Pose](https://docs.ultralytics.com/datasets/pose/tiger-pose/) for animal pose estimation, [Hand Keypoints](https://docs.ultralytics.com/datasets/pose/hand-keypoints/) for hand tracking, or [Dog-Pose](https://docs.ultralytics.com/datasets/pose/dog-pose/) for canine pose analysis.

## Val

Validate trained YOLO11n-pose model [accuracy](https://www.ultralytics.com/glossary/accuracy) on the COCO8-pose dataset. No arguments are needed as the `model` retains its training `data` and arguments as model attributes.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-pose.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Validate the model
        metrics = model.val()  # no arguments needed, dataset and settings remembered
        metrics.box.map  # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps  # a list contains map50-95 of each category
        metrics.pose.map  # map50-95(P)
        metrics.pose.map50  # map50(P)
        metrics.pose.map75  # map75(P)
        metrics.pse.maps  # a list contains map50-95(P) of each category
        ```

    === "CLI"

        ```bash
        yolo pose val model=yolo11n-pose.pt # val official model
        yolo pose val model=path/to/best.pt # val custom model
        ```

## Predict

Use a trained YOLO11n-pose model to run predictions on images. The [predict mode](https://docs.ultralytics.com/modes/predict/) allows you to perform inference on images, videos, or real-time streams.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-pose.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Predict with the model
        results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

        # Access the results
        for result in results:
            xy = result.keypoints.xy  # x and y coordinates
            xyn = result.keypoints.xyn  # normalized
            kpts = result.keypoints.data  # x, y, visibility (if available)
        ```

    === "CLI"

        ```bash
        yolo pose predict model=yolo11n-pose.pt source='https://ultralytics.com/images/bus.jpg' # predict with official model
        yolo pose predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg' # predict with custom model
        ```

See full `predict` mode details in the [Predict](../modes/predict.md) page.

## Export

Export a YOLO11n Pose model to a different format like ONNX, CoreML, etc. This allows you to deploy your model on various platforms and devices for [real-time inference](https://www.ultralytics.com/glossary/real-time-inference).

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-pose.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom trained model

        # Export the model
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n-pose.pt format=onnx # export official model
        yolo export model=path/to/best.pt format=onnx # export custom trained model
        ```

Available YOLO11-pose export formats are in the table below. You can export to any format using the `format` argument, i.e. `format='onnx'` or `format='engine'`. You can predict or validate directly on exported models, i.e. `yolo predict model=yolo11n-pose.onnx`. Usage examples are shown for your model after export completes.

{% include "macros/export-table.md" %}

See full `export` details in the [Export](../modes/export.md) page.

## FAQ

### What is Pose Estimation with Ultralytics YOLO11 and how does it work?

Pose estimation with Ultralytics YOLO11 involves identifying specific points, known as keypoints, in an image. These keypoints typically represent joints or other important features of the object. The output includes the `[x, y]` coordinates and confidence scores for each point. YOLO11-pose models are specifically designed for this task and use the `-pose` suffix, such as `yolo11n-pose.pt`. These models are pre-trained on datasets like [COCO keypoints](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml) and can be used for various pose estimation tasks. For more information, visit the [Pose Estimation Page](#pose-estimation).

### How can I train a YOLO11-pose model on a custom dataset?

Training a YOLO11-pose model on a custom dataset involves loading a model, either a new model defined by a YAML file or a pre-trained model. You can then start the training process using your specified dataset and parameters.

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-pose.yaml")  # build a new model from YAML
model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="your-dataset.yaml", epochs=100, imgsz=640)
```

For comprehensive details on training, refer to the [Train Section](#train). You can also use [Ultralytics HUB](https://www.ultralytics.com/hub) for a no-code approach to training custom pose estimation models.

### How do I validate a trained YOLO11-pose model?

Validation of a YOLO11-pose model involves assessing its accuracy using the same dataset parameters retained during training. Here's an example:

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-pose.pt")  # load an official model
model = YOLO("path/to/best.pt")  # load a custom model

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
```

For more information, visit the [Val Section](#val).

### Can I export a YOLO11-pose model to other formats, and how?

Yes, you can export a YOLO11-pose model to various formats like ONNX, CoreML, TensorRT, and more. This can be done using either Python or the Command Line Interface (CLI).

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-pose.pt")  # load an official model
model = YOLO("path/to/best.pt")  # load a custom trained model

# Export the model
model.export(format="onnx")
```

Refer to the [Export Section](#export) for more details. Exported models can be deployed on edge devices for [real-time applications](https://www.ultralytics.com/blog/real-time-inferences-in-vision-ai-solutions-are-making-an-impact) like fitness tracking, sports analysis, or [robotics](https://www.ultralytics.com/blog/from-algorithms-to-automation-ais-role-in-robotics).

### What are the available Ultralytics YOLO11-pose models and their performance metrics?

Ultralytics YOLO11 offers various pretrained pose models such as YOLO11n-pose, YOLO11s-pose, YOLO11m-pose, among others. These models differ in size, accuracy (mAP), and speed. For instance, the YOLO11n-pose model achieves a mAP<sup>pose</sup>50-95 of 50.0 and an mAP<sup>pose</sup>50 of 81.0. For a complete list and performance details, visit the [Models Section](#models).
>>>>>>> 0e451281e978a7c5f581cc5d9bb3d0f62e0bc632
