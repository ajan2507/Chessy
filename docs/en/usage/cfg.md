<<<<<<< HEAD
---
comments: true
description: Master YOLOv8 settings and hyperparameters for improved model performance. Learn to use YOLO CLI commands, adjust training settings, and optimize YOLO tasks & modes.
keywords: YOLOv8, settings, hyperparameters, YOLO CLI commands, YOLO tasks, YOLO modes, Ultralytics documentation, model optimization, YOLOv8 training
---

YOLO settings and hyperparameters play a critical role in the model's performance, speed, and accuracy. These settings and hyperparameters can affect the model's behavior at various stages of the model development process, including training, validation, and prediction.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/GsXGnb-A4Kc?start=87"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Mastering Ultralytics YOLOv8: Configuration
</p>

Ultralytics commands use the following syntax:

!!! Example

    === "CLI"

        ```bash
        yolo TASK MODE ARGS
        ```

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLOv8 model from a pre-trained weights file
        model = YOLO('yolov8n.pt')

        # Run MODE mode using the custom arguments ARGS (guess TASK)
        model.MODE(ARGS)
        ```

Where:

- `TASK` (optional) is one of ([detect](../tasks/detect.md), [segment](../tasks/segment.md), [classify](../tasks/classify.md), [pose](../tasks/pose.md))
- `MODE` (required) is one of ([train](../modes/train.md), [val](../modes/val.md), [predict](../modes/predict.md), [export](../modes/export.md), [track](../modes/track.md))
- `ARGS` (optional) are `arg=value` pairs like `imgsz=640` that override defaults.

Default `ARG` values are defined on this page from the `cfg/defaults.yaml` [file](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml).

#### Tasks

YOLO models can be used for a variety of tasks, including detection, segmentation, classification and pose. These tasks differ in the type of output they produce and the specific problem they are designed to solve.

**Detect**: For identifying and localizing objects or regions of interest in an image or video.
**Segment**: For dividing an image or video into regions or pixels that correspond to different objects or classes.
**Classify**: For predicting the class label of an input image.
**Pose**: For identifying objects and estimating their keypoints in an image or video.

| Key    | Value      | Description                                     |
|--------|------------|-------------------------------------------------|
| `task` | `'detect'` | YOLO task, i.e. detect, segment, classify, pose |

[Tasks Guide](../tasks/index.md){ .md-button }

#### Modes

YOLO models can be used in different modes depending on the specific problem you are trying to solve. These modes include:

**Train**: For training a YOLOv8 model on a custom dataset.
**Val**: For validating a YOLOv8 model after it has been trained.
**Predict**: For making predictions using a trained YOLOv8 model on new images or videos.
**Export**: For exporting a YOLOv8 model to a format that can be used for deployment.
**Track**: For tracking objects in real-time using a YOLOv8 model.
**Benchmark**: For benchmarking YOLOv8 exports (ONNX, TensorRT, etc.) speed and accuracy.

| Key    | Value     | Description                                                   |
|--------|-----------|---------------------------------------------------------------|
| `mode` | `'train'` | YOLO mode, i.e. train, val, predict, export, track, benchmark |

[Modes Guide](../modes/index.md){ .md-button }

## Train

The training settings for YOLO models encompass various hyperparameters and configurations used during the training process. These settings influence the model's performance, speed, and accuracy. Key training settings include batch size, learning rate, momentum, and weight decay. Additionally, the choice of optimizer, loss function, and training dataset composition can impact the training process. Careful tuning and experimentation with these settings are crucial for optimizing performance.

| Key               | Value    | Description                                                                                    |
|-------------------|----------|------------------------------------------------------------------------------------------------|
| `model`           | `None`   | path to model file, i.e. yolov8n.pt, yolov8n.yaml                                              |
| `data`            | `None`   | path to data file, i.e. coco128.yaml                                                           |
| `epochs`          | `100`    | number of epochs to train for                                                                  |
| `patience`        | `50`     | epochs to wait for no observable improvement for early stopping of training                    |
| `batch`           | `16`     | number of images per batch (-1 for AutoBatch)                                                  |
| `imgsz`           | `640`    | size of input images as integer                                                                |
| `save`            | `True`   | save train checkpoints and predict results                                                     |
| `save_period`     | `-1`     | Save checkpoint every x epochs (disabled if < 1)                                               |
| `cache`           | `False`  | True/ram, disk or False. Use cache for data loading                                            |
| `device`          | `None`   | device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu                           |
| `workers`         | `8`      | number of worker threads for data loading (per RANK if DDP)                                    |
| `project`         | `None`   | project name                                                                                   |
| `name`            | `None`   | experiment name                                                                                |
| `exist_ok`        | `False`  | whether to overwrite existing experiment                                                       |
| `pretrained`      | `True`   | (bool or str) whether to use a pretrained model (bool) or a model to load weights from (str)   |
| `optimizer`       | `'auto'` | optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]              |
| `verbose`         | `False`  | whether to print verbose output                                                                |
| `seed`            | `0`      | random seed for reproducibility                                                                |
| `deterministic`   | `True`   | whether to enable deterministic mode                                                           |
| `single_cls`      | `False`  | train multi-class data as single-class                                                         |
| `rect`            | `False`  | rectangular training with each batch collated for minimum padding                              |
| `cos_lr`          | `False`  | use cosine learning rate scheduler                                                             |
| `close_mosaic`    | `10`     | (int) disable mosaic augmentation for final epochs (0 to disable)                              |
| `resume`          | `False`  | resume training from last checkpoint                                                           |
| `amp`             | `True`   | Automatic Mixed Precision (AMP) training, choices=[True, False]                                |
| `fraction`        | `1.0`    | dataset fraction to train on (default is 1.0, all images in train set)                         |
| `profile`         | `False`  | profile ONNX and TensorRT speeds during training for loggers                                   |
| `freeze`          | `None`   | (int or list, optional) freeze first n layers, or freeze list of layer indices during training |
| `lr0`             | `0.01`   | initial learning rate (i.e. SGD=1E-2, Adam=1E-3)                                               |
| `lrf`             | `0.01`   | final learning rate (lr0 * lrf)                                                                |
| `momentum`        | `0.937`  | SGD momentum/Adam beta1                                                                        |
| `weight_decay`    | `0.0005` | optimizer weight decay 5e-4                                                                    |
| `warmup_epochs`   | `3.0`    | warmup epochs (fractions ok)                                                                   |
| `warmup_momentum` | `0.8`    | warmup initial momentum                                                                        |
| `warmup_bias_lr`  | `0.1`    | warmup initial bias lr                                                                         |
| `box`             | `7.5`    | box loss gain                                                                                  |
| `cls`             | `0.5`    | cls loss gain (scale with pixels)                                                              |
| `dfl`             | `1.5`    | dfl loss gain                                                                                  |
| `pose`            | `12.0`   | pose loss gain (pose-only)                                                                     |
| `kobj`            | `2.0`    | keypoint obj loss gain (pose-only)                                                             |
| `label_smoothing` | `0.0`    | label smoothing (fraction)                                                                     |
| `nbs`             | `64`     | nominal batch size                                                                             |
| `overlap_mask`    | `True`   | masks should overlap during training (segment train only)                                      |
| `mask_ratio`      | `4`      | mask downsample ratio (segment train only)                                                     |
| `dropout`         | `0.0`    | use dropout regularization (classify train only)                                               |
| `val`             | `True`   | validate/test during training                                                                  |
| `plots`           | `False`  | save plots and images during train/val                                                         |

[Train Guide](../modes/train.md){ .md-button }

## Predict

The prediction settings for YOLO models encompass a range of hyperparameters and configurations that influence the model's performance, speed, and accuracy during inference on new data. Careful tuning and experimentation with these settings are essential to achieve optimal performance for a specific task. Key settings include the confidence threshold, Non-Maximum Suppression (NMS) threshold, and the number of classes considered. Additional factors affecting the prediction process are input data size and format, the presence of supplementary features such as masks or multiple labels per box, and the particular task the model is employed for.

Inference arguments:

| Name            | Type           | Default                | Description                                                                |
|-----------------|----------------|------------------------|----------------------------------------------------------------------------|
| `source`        | `str`          | `'ultralytics/assets'` | source directory for images or videos                                      |
| `conf`          | `float`        | `0.25`                 | object confidence threshold for detection                                  |
| `iou`           | `float`        | `0.7`                  | intersection over union (IoU) threshold for NMS                            |
| `imgsz`         | `int or tuple` | `640`                  | image size as scalar or (h, w) list, i.e. (640, 480)                       |
| `half`          | `bool`         | `False`                | use half precision (FP16)                                                  |
| `device`        | `None or str`  | `None`                 | device to run on, i.e. cuda device=0/1/2/3 or device=cpu                   |
| `max_det`       | `int`          | `300`                  | maximum number of detections per image                                     |
| `vid_stride`    | `bool`         | `False`                | video frame-rate stride                                                    |
| `stream_buffer` | `bool`         | `False`                | buffer all streaming frames (True) or return the most recent frame (False) |
| `visualize`     | `bool`         | `False`                | visualize model features                                                   |
| `augment`       | `bool`         | `False`                | apply image augmentation to prediction sources                             |
| `agnostic_nms`  | `bool`         | `False`                | class-agnostic NMS                                                         |
| `retina_masks`  | `bool`         | `False`                | use high-resolution segmentation masks                                     |
| `classes`       | `None or list` | `None`                 | filter results by class, i.e. classes=0, or classes=[0,2,3]                |

Visualization arguments:

| Name          | Type          | Default | Description                                                     |
|---------------|---------------|---------|-----------------------------------------------------------------|
| `show`        | `bool`        | `False` | show predicted images and videos if environment allows          |
| `save`        | `bool`        | `False` | save predicted images and videos                                |
| `save_frames` | `bool`        | `False` | save predicted individual video frames                          |
| `save_txt`    | `bool`        | `False` | save results as `.txt` file                                     |
| `save_conf`   | `bool`        | `False` | save results with confidence scores                             |
| `save_crop`   | `bool`        | `False` | save cropped images with results                                |
| `show_labels` | `bool`        | `True`  | show prediction labels, i.e. 'person'                           |
| `show_conf`   | `bool`        | `True`  | show prediction confidence, i.e. '0.99'                         |
| `show_boxes`  | `bool`        | `True`  | show prediction boxes                                           |
| `line_width`  | `None or int` | `None`  | line width of the bounding boxes. Scaled to image size if None. |

[Predict Guide](../modes/predict.md){ .md-button }

## Val

The val (validation) settings for YOLO models involve various hyperparameters and configurations used to evaluate the model's performance on a validation dataset. These settings influence the model's performance, speed, and accuracy. Common YOLO validation settings include batch size, validation frequency during training, and performance evaluation metrics. Other factors affecting the validation process include the validation dataset's size and composition, as well as the specific task the model is employed for. Careful tuning and experimentation with these settings are crucial to ensure optimal performance on the validation dataset and detect and prevent overfitting.

| Key           | Value   | Description                                                        |
|---------------|---------|--------------------------------------------------------------------|
| `data`        | `None`  | path to data file, i.e. coco128.yaml                               |
| `imgsz`       | `640`   | size of input images as integer                                    |
| `batch`       | `16`    | number of images per batch (-1 for AutoBatch)                      |
| `save_json`   | `False` | save results to JSON file                                          |
| `save_hybrid` | `False` | save hybrid version of labels (labels + additional predictions)    |
| `conf`        | `0.001` | object confidence threshold for detection                          |
| `iou`         | `0.6`   | intersection over union (IoU) threshold for NMS                    |
| `max_det`     | `300`   | maximum number of detections per image                             |
| `half`        | `True`  | use half precision (FP16)                                          |
| `device`      | `None`  | device to run on, i.e. cuda device=0/1/2/3 or device=cpu           |
| `dnn`         | `False` | use OpenCV DNN for ONNX inference                                  |
| `plots`       | `False` | save plots and images during train/val                             |
| `rect`        | `False` | rectangular val with each batch collated for minimum padding       |
| `split`       | `val`   | dataset split to use for validation, i.e. 'val', 'test' or 'train' |

[Val Guide](../modes/val.md){ .md-button }

## Export

Export settings for YOLO models encompass configurations and options related to saving or exporting the model for use in different environments or platforms. These settings can impact the model's performance, size, and compatibility with various systems. Key export settings include the exported model file format (e.g., ONNX, TensorFlow SavedModel), the target device (e.g., CPU, GPU), and additional features such as masks or multiple labels per box. The export process may also be affected by the model's specific task and the requirements or constraints of the destination environment or platform. It is crucial to thoughtfully configure these settings to ensure the exported model is optimized for the intended use case and functions effectively in the target environment.

| Key         | Value           | Description                                          |
|-------------|-----------------|------------------------------------------------------|
| `format`    | `'torchscript'` | format to export to                                  |
| `imgsz`     | `640`           | image size as scalar or (h, w) list, i.e. (640, 480) |
| `keras`     | `False`         | use Keras for TF SavedModel export                   |
| `optimize`  | `False`         | TorchScript: optimize for mobile                     |
| `half`      | `False`         | FP16 quantization                                    |
| `int8`      | `False`         | INT8 quantization                                    |
| `dynamic`   | `False`         | ONNX/TensorRT: dynamic axes                          |
| `simplify`  | `False`         | ONNX/TensorRT: simplify model                        |
| `opset`     | `None`          | ONNX: opset version (optional, defaults to latest)   |
| `workspace` | `4`             | TensorRT: workspace size (GB)                        |
| `nms`       | `False`         | CoreML: add NMS                                      |

[Export Guide](../modes/export.md){ .md-button }

## Augmentation

Augmentation settings for YOLO models refer to the various transformations and modifications applied to the training data to increase the diversity and size of the dataset. These settings can affect the model's performance, speed, and accuracy. Some common YOLO augmentation settings include the type and intensity of the transformations applied (e.g. random flips, rotations, cropping, color changes), the probability with which each transformation is applied, and the presence of additional features such as masks or multiple labels per box. Other factors that may affect the augmentation process include the size and composition of the original dataset and the specific task the model is being used for. It is important to carefully tune and experiment with these settings to ensure that the augmented dataset is diverse and representative enough to train a high-performing model.

| Key           | Value   | Description                                     |
|---------------|---------|-------------------------------------------------|
| `hsv_h`       | `0.015` | image HSV-Hue augmentation (fraction)           |
| `hsv_s`       | `0.7`   | image HSV-Saturation augmentation (fraction)    |
| `hsv_v`       | `0.4`   | image HSV-Value augmentation (fraction)         |
| `degrees`     | `0.0`   | image rotation (+/- deg)                        |
| `translate`   | `0.1`   | image translation (+/- fraction)                |
| `scale`       | `0.5`   | image scale (+/- gain)                          |
| `shear`       | `0.0`   | image shear (+/- deg)                           |
| `perspective` | `0.0`   | image perspective (+/- fraction), range 0-0.001 |
| `flipud`      | `0.0`   | image flip up-down (probability)                |
| `fliplr`      | `0.5`   | image flip left-right (probability)             |
| `mosaic`      | `1.0`   | image mosaic (probability)                      |
| `mixup`       | `0.0`   | image mixup (probability)                       |
| `copy_paste`  | `0.0`   | segment copy-paste (probability)                |

## Logging, checkpoints, plotting and file management

Logging, checkpoints, plotting, and file management are important considerations when training a YOLO model.

- Logging: It is often helpful to log various metrics and statistics during training to track the model's progress and diagnose any issues that may arise. This can be done using a logging library such as TensorBoard or by writing log messages to a file.
- Checkpoints: It is a good practice to save checkpoints of the model at regular intervals during training. This allows you to resume training from a previous point if the training process is interrupted or if you want to experiment with different training configurations.
- Plotting: Visualizing the model's performance and training progress can be helpful for understanding how the model is behaving and identifying potential issues. This can be done using a plotting library such as matplotlib or by generating plots using a logging library such as TensorBoard.
- File management: Managing the various files generated during the training process, such as model checkpoints, log files, and plots, can be challenging. It is important to have a clear and organized file structure to keep track of these files and make it easy to access and analyze them as needed.

Effective logging, checkpointing, plotting, and file management can help you keep track of the model's progress and make it easier to debug and optimize the training process.

| Key        | Value    | Description                                                                                    |
|------------|----------|------------------------------------------------------------------------------------------------|
| `project`  | `'runs'` | project name                                                                                   |
| `name`     | `'exp'`  | experiment name. `exp` gets automatically incremented if not specified, i.e, `exp`, `exp2` ... |
| `exist_ok` | `False`  | whether to overwrite existing experiment                                                       |
| `plots`    | `False`  | save plots during train/val                                                                    |
| `save`     | `False`  | save train checkpoints and predict results                                                     |
=======
---
comments: true
description: Optimize your Ultralytics YOLO model's performance with the right settings and hyperparameters. Learn about training, validation, and prediction configurations.
keywords: YOLO, hyperparameters, configuration, training, validation, prediction, model settings, Ultralytics, performance optimization, machine learning
---

# Configuration

YOLO settings and hyperparameters play a critical role in the model's performance, speed, and [accuracy](https://www.ultralytics.com/glossary/accuracy). These settings can affect the model's behavior at various stages, including training, validation, and prediction.

**Watch:** Mastering Ultralytics YOLO: Configuration

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/GsXGnb-A4Kc?start=87"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Mastering Ultralytics YOLO: Configuration
</p>

Ultralytics commands use the following syntax:

!!! example

    === "CLI"

        ```bash
        yolo TASK MODE ARGS
        ```

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO model from a pre-trained weights file
        model = YOLO("yolo11n.pt")

        # Run MODE mode using the custom arguments ARGS (guess TASK)
        model.MODE(ARGS)
        ```

Where:

- `TASK` (optional) is one of ([detect](../tasks/detect.md), [segment](../tasks/segment.md), [classify](../tasks/classify.md), [pose](../tasks/pose.md), [obb](../tasks/obb.md))
- `MODE` (required) is one of ([train](../modes/train.md), [val](../modes/val.md), [predict](../modes/predict.md), [export](../modes/export.md), [track](../modes/track.md), [benchmark](../modes/benchmark.md))
- `ARGS` (optional) are `arg=value` pairs like `imgsz=640` that override defaults.

Default `ARG` values are defined on this page and come from the `cfg/defaults.yaml` [file](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml).

## Tasks

Ultralytics YOLO models can perform a variety of computer vision tasks, including:

- **Detect**: [Object detection](https://docs.ultralytics.com/tasks/detect/) identifies and localizes objects within an image or video.
- **Segment**: [Instance segmentation](https://docs.ultralytics.com/tasks/segment/) divides an image or video into regions corresponding to different objects or classes.
- **Classify**: [Image classification](https://docs.ultralytics.com/tasks/classify/) predicts the class label of an input image.
- **Pose**: [Pose estimation](https://docs.ultralytics.com/tasks/pose/) identifies objects and estimates their keypoints in an image or video.
- **OBB**: [Oriented Bounding Boxes](https://docs.ultralytics.com/tasks/obb/) uses rotated bounding boxes, suitable for satellite or medical imagery.

| Argument | Default    | Description                                                                                                                                                                                                                                                                                                                        |
| -------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `task`   | `'detect'` | Specifies the YOLO task: `detect` for [object detection](https://www.ultralytics.com/glossary/object-detection), `segment` for segmentation, `classify` for classification, `pose` for pose estimation, and `obb` for oriented bounding boxes. Each task is tailored to specific outputs and problems in image and video analysis. |

[Tasks Guide](../tasks/index.md){ .md-button }

## Modes

Ultralytics YOLO models operate in different modes, each designed for a specific stage of the model lifecycle:

- **Train**: Train a YOLO model on a custom dataset.
- **Val**: Validate a trained YOLO model.
- **Predict**: Use a trained YOLO model to make predictions on new images or videos.
- **Export**: Export a YOLO model for deployment.
- **Track**: Track objects in real-time using a YOLO model.
- **Benchmark**: Benchmark the speed and accuracy of YOLO exports (ONNX, TensorRT, etc.).

| Argument | Default   | Description                                                                                                                                                                                                                                                                                                        |
| -------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `mode`   | `'train'` | Specifies the YOLO model's operating mode: `train` for model training, `val` for validation, `predict` for inference, `export` for converting to deployment formats, `track` for object tracking, and `benchmark` for performance evaluation. Each mode supports different stages, from development to deployment. |

[Modes Guide](../modes/index.md){ .md-button }

## Train Settings

Training settings for YOLO models include hyperparameters and configurations that affect the model's performance, speed, and [accuracy](https://www.ultralytics.com/glossary/accuracy). Key settings include [batch size](https://www.ultralytics.com/glossary/batch-size), [learning rate](https://www.ultralytics.com/glossary/learning-rate), momentum, and weight decay. The choice of optimizer, [loss function](https://www.ultralytics.com/glossary/loss-function), and dataset composition also impact training. Tuning and experimentation are crucial for optimal performance. For more details, see the [Ultralytics entrypoint function](../reference/cfg/__init__.md).

{% include "macros/train-args.md" %}

!!! info "Note on Batch-size Settings"

    The `batch` argument offers three configuration options:

    - **Fixed Batch Size**: Specify the number of images per batch with an integer (e.g., `batch=16`).
    - **Auto Mode (60% GPU Memory)**: Use `batch=-1` for automatic adjustment to approximately 60% CUDA memory utilization.
    - **Auto Mode with Utilization Fraction**: Set a fraction (e.g., `batch=0.70`) to adjust based on a specified GPU memory usage.

[Train Guide](../modes/train.md){ .md-button }

## Predict Settings

Prediction settings for YOLO models include hyperparameters and configurations that influence performance, speed, and [accuracy](https://www.ultralytics.com/glossary/accuracy) during inference. Key settings include the confidence threshold, [Non-Maximum Suppression (NMS)](https://www.ultralytics.com/glossary/non-maximum-suppression-nms) threshold, and the number of classes. Input data size, format, and supplementary features like masks also affect predictions. Tuning these settings is essential for optimal performance.

Inference arguments:

{% include "macros/predict-args.md" %}

Visualization arguments:

{% from "macros/visualization-args.md" import param_table %} {{ param_table() }}

[Predict Guide](../modes/predict.md){ .md-button }

## Validation Settings

Validation settings for YOLO models involve hyperparameters and configurations to evaluate performance on a [validation dataset](https://www.ultralytics.com/glossary/validation-data). These settings influence performance, speed, and [accuracy](https://www.ultralytics.com/glossary/accuracy). Common settings include batch size, validation frequency, and performance metrics. The validation dataset's size and composition, along with the specific task, also affect the process.

{% include "macros/validation-args.md" %}

Careful tuning and experimentation are crucial to ensure optimal performance and to detect and prevent [overfitting](https://www.ultralytics.com/glossary/overfitting).

[Val Guide](../modes/val.md){ .md-button }

## Export Settings

Export settings for YOLO models include configurations for saving or exporting the model for use in different environments. These settings impact performance, size, and compatibility. Key settings include the exported file format (e.g., ONNX, TensorFlow SavedModel), the target device (e.g., CPU, GPU), and features like masks. The model's task and the destination environment's constraints also affect the export process.

{% include "macros/export-args.md" %}

Thoughtful configuration ensures the exported model is optimized for its use case and functions effectively in the target environment.

[Export Guide](../modes/export.md){ .md-button }

## Solutions Settings

Ultralytics Solutions configuration settings offer flexibility to customize models for tasks like object counting, heatmap creation, workout tracking, data analysis, zone tracking, queue management, and region-based counting. These options allow easy adjustments for accurate and useful results tailored to specific needs.

{% from "macros/solutions-args.md" import param_table %} {{ param_table() }}

[Solutions Guide](../solutions/index.md){ .md-button }

## Augmentation Settings

[Data augmentation](https://www.ultralytics.com/glossary/data-augmentation) techniques are essential for improving YOLO model robustness and performance by introducing variability into the [training data](https://www.ultralytics.com/glossary/training-data), helping the model generalize better to unseen data. The following table outlines each augmentation argument's purpose and effect:

{% include "macros/augmentation-args.md" %}

Adjust these settings to meet dataset and task requirements. Experimenting with different values can help find the optimal augmentation strategy for the best model performance.

[Augmentation Guide](../guides/yolo-data-augmentation.md){ .md-button }

## Logging, Checkpoints and Plotting Settings

Logging, checkpoints, plotting, and file management are important when training a YOLO model:

- **Logging**: Track the model's progress and diagnose issues using libraries like [TensorBoard](https://docs.ultralytics.com/integrations/tensorboard/) or by writing to a file.
- **Checkpoints**: Save the model at regular intervals to resume training or experiment with different configurations.
- **Plotting**: Visualize performance and training progress using libraries like matplotlib or TensorBoard.
- **File management**: Organize files generated during training, such as checkpoints, log files, and plots, for easy access and analysis.

Effective management of these aspects helps track progress and makes debugging and optimization easier.

| Argument   | Default  | Description                                                                                                                                                                                                                                                                                               |
| ---------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `project`  | `'runs'` | Specifies the root directory for saving training runs. Each run is saved in a separate subdirectory.                                                                                                                                                                                                      |
| `name`     | `'exp'`  | Defines the experiment name. If unspecified, YOLO increments this name for each run (e.g., `exp`, `exp2`) to avoid overwriting.                                                                                                                                                                           |
| `exist_ok` | `False`  | Determines whether to overwrite an existing experiment directory. `True` allows overwriting; `False` prevents it.                                                                                                                                                                                         |
| `plots`    | `False`  | Controls the generation and saving of training and validation plots. Set to `True` to create plots like loss curves, [precision](https://www.ultralytics.com/glossary/precision)-[recall](https://www.ultralytics.com/glossary/recall) curves, and sample predictions for visual tracking of performance. |
| `save`     | `False`  | Enables saving training checkpoints and final model weights. Set to `True` to save model states periodically, allowing training resumption or model deployment.                                                                                                                                           |

## FAQ

### How do I improve my YOLO model's performance during training?

Improve performance by tuning hyperparameters like [batch size](https://www.ultralytics.com/glossary/batch-size), [learning rate](https://www.ultralytics.com/glossary/learning-rate), momentum, and weight decay. Adjust [data augmentation](https://www.ultralytics.com/glossary/data-augmentation) settings, select the right optimizer, and use techniques like early stopping or [mixed precision](https://www.ultralytics.com/glossary/mixed-precision). For details, see the [Train Guide](../modes/train.md).

### What are the key hyperparameters for YOLO model accuracy?

Key hyperparameters affecting accuracy include:

- **Batch Size (`batch`)**: Larger sizes can stabilize training but need more memory.
- **Learning Rate (`lr0`)**: Smaller rates offer fine adjustments but slower convergence.
- **Momentum (`momentum`)**: Accelerates gradient vectors, dampening oscillations.
- **Image Size (`imgsz`)**: Larger sizes improve accuracy but increase computational load.

Adjust these based on your dataset and hardware. Learn more in [Train Settings](#train-settings).

### How do I set the learning rate for training a YOLO model?

The learning rate (`lr0`) is crucial; start with `0.01` for SGD or `0.001` for [Adam optimizer](https://www.ultralytics.com/glossary/adam-optimizer). Monitor metrics and adjust as needed. Use cosine learning rate schedulers (`cos_lr`) or warmup (`warmup_epochs`, `warmup_momentum`). Details are in the [Train Guide](../modes/train.md).

### What are the default inference settings for YOLO models?

Default settings include:

- **Confidence Threshold (`conf=0.25`)**: Minimum confidence for detections.
- **IoU Threshold (`iou=0.7`)**: For [Non-Maximum Suppression (NMS)](https://www.ultralytics.com/glossary/non-maximum-suppression-nms).
- **Image Size (`imgsz=640`)**: Resizes input images.
- **Device (`device=None`)**: Selects CPU or GPU.

For a full overview, see [Predict Settings](#predict-settings) and the [Predict Guide](../modes/predict.md).

### Why use mixed precision training with YOLO models?

[Mixed precision](https://www.ultralytics.com/glossary/mixed-precision) training (`amp=True`) reduces memory usage and speeds up training using FP16 and FP32. It's beneficial for modern GPUs, allowing larger models and faster computations without significant accuracy loss. Learn more in the [Train Guide](../modes/train.md).
>>>>>>> 0e451281e978a7c5f581cc5d9bb3d0f62e0bc632
