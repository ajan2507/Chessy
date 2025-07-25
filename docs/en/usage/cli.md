<<<<<<< HEAD
---
comments: true
description: 'Learn how to use Ultralytics YOLO through Command Line: train models, run predictions and exports models to different formats easily using terminal commands.'
keywords: Ultralytics, YOLO, CLI, train, validation, prediction, command line interface, YOLO CLI, YOLO terminal, model training, prediction, exporting
---

# Command Line Interface Usage

The YOLO command line interface (CLI) allows for simple single-line commands without the need for a Python environment. CLI requires no customization or Python code. You can simply run all tasks from the terminal with the `yolo` command.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/GsXGnb-A4Kc?start=19"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Mastering Ultralytics YOLOv8: CLI
</p>

!!! Example

    === "Syntax"

        Ultralytics `yolo` commands use the following syntax:
        ```bash
        yolo TASK MODE ARGS

        Where   TASK (optional) is one of [detect, segment, classify]
                MODE (required) is one of [train, val, predict, export, track]
                ARGS (optional) are any number of custom 'arg=value' pairs like 'imgsz=320' that override defaults.
        ```
        See all ARGS in the full [Configuration Guide](cfg.md) or with `yolo cfg`

    === "Train"

        Train a detection model for 10 epochs with an initial learning_rate of 0.01
        ```bash
        yolo train data=coco128.yaml model=yolov8n.pt epochs=10 lr0=0.01
        ```

    === "Predict"

        Predict a YouTube video using a pretrained segmentation model at image size 320:
        ```bash
        yolo predict model=yolov8n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320
        ```

    === "Val"

        Val a pretrained detection model at batch-size 1 and image size 640:
        ```bash
        yolo val model=yolov8n.pt data=coco128.yaml batch=1 imgsz=640
        ```

    === "Export"

        Export a YOLOv8n classification model to ONNX format at image size 224 by 128 (no TASK required)
        ```bash
        yolo export model=yolov8n-cls.pt format=onnx imgsz=224,128
        ```

    === "Special"

        Run special commands to see version, view settings, run checks and more:
        ```bash
        yolo help
        yolo checks
        yolo version
        yolo settings
        yolo copy-cfg
        yolo cfg
        ```

Where:

- `TASK` (optional) is one of `[detect, segment, classify]`. If it is not passed explicitly YOLOv8 will try to guess the `TASK` from the model type.
- `MODE` (required) is one of `[train, val, predict, export, track]`
- `ARGS` (optional) are any number of custom `arg=value` pairs like `imgsz=320` that override defaults. For a full list of available `ARGS` see the [Configuration](cfg.md) page and `defaults.yaml`
  GitHub [source](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml).

!!! Warning "Warning"

    Arguments must be passed as `arg=val` pairs, split by an equals `=` sign and delimited by spaces ` ` between pairs. Do not use `--` argument prefixes or commas `,` between arguments.

    - `yolo predict model=yolov8n.pt imgsz=640 conf=0.25` &nbsp; ✅
    - `yolo predict model yolov8n.pt imgsz 640 conf 0.25` &nbsp; ❌
    - `yolo predict --model yolov8n.pt --imgsz 640 --conf 0.25` &nbsp; ❌

## Train

Train YOLOv8n on the COCO128 dataset for 100 epochs at image size 640. For a full list of available arguments see the [Configuration](cfg.md) page.

!!! Example "Example"

    === "Train"

        Start training YOLOv8n on COCO128 for 100 epochs at image-size 640.
        ```bash
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

    === "Resume"

        Resume an interrupted training.
        ```bash
        yolo detect train resume model=last.pt
        ```

## Val

Validate trained YOLOv8n model accuracy on the COCO128 dataset. No argument need to passed as the `model` retains it's training `data` and arguments as model attributes.

!!! Example "Example"

    === "Official"

        Validate an official YOLOv8n model.
        ```bash
        yolo detect val model=yolov8n.pt
        ```

    === "Custom"

        Validate a custom-trained model.
        ```bash
        yolo detect val model=path/to/best.pt
        ```

## Predict

Use a trained YOLOv8n model to run predictions on images.

!!! Example "Example"

    === "Official"

        Predict with an official YOLOv8n model.
        ```bash
        yolo detect predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'
        ```

    === "Custom"

        Predict with a custom model.
        ```bash
        yolo detect predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'
        ```

## Export

Export a YOLOv8n model to a different format like ONNX, CoreML, etc.

!!! Example "Example"

    === "Official"

        Export an official YOLOv8n model to ONNX format.
        ```bash
        yolo export model=yolov8n.pt format=onnx
        ```

    === "Custom"

        Export a custom-trained model to ONNX format.
        ```bash
        yolo export model=path/to/best.pt format=onnx
        ```

Available YOLOv8 export formats are in the table below. You can export to any format using the `format` argument, i.e. `format='onnx'` or `format='engine'`.

| Format                                                             | `format` Argument | Model                     | Metadata | Arguments                                           |
|--------------------------------------------------------------------|-------------------|---------------------------|----------|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -                 | `yolov8n.pt`              | ✅        | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript`     | `yolov8n.torchscript`     | ✅        | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`            | `yolov8n.onnx`            | ✅        | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`        | `yolov8n_openvino_model/` | ✅        | `imgsz`, `half`, `int8`                             |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`          | `yolov8n.engine`          | ✅        | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`          | `yolov8n.mlpackage`       | ✅        | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`     | `yolov8n_saved_model/`    | ✅        | `imgsz`, `keras`, `int8`                            |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`              | `yolov8n.pb`              | ❌        | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`          | `yolov8n.tflite`          | ✅        | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`         | `yolov8n_edgetpu.tflite`  | ✅        | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`            | `yolov8n_web_model/`      | ✅        | `imgsz`, `half`, `int8`                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`          | `yolov8n_paddle_model/`   | ✅        | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`            | `yolov8n_ncnn_model/`     | ✅        | `imgsz`, `half`                                     |

## Overriding default arguments

Default arguments can be overridden by simply passing them as arguments in the CLI in `arg=value` pairs.

!!! Tip ""

    === "Train"
        Train a detection model for `10 epochs` with `learning_rate` of `0.01`
        ```bash
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=10 lr0=0.01
        ```

    === "Predict"
        Predict a YouTube video using a pretrained segmentation model at image size 320:
        ```bash
        yolo segment predict model=yolov8n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320
        ```

    === "Val"
        Validate a pretrained detection model at batch-size 1 and image size 640:
        ```bash
        yolo detect val model=yolov8n.pt data=coco128.yaml batch=1 imgsz=640
        ```

## Overriding default config file

You can override the `default.yaml` config file entirely by passing a new file with the `cfg` arguments, i.e. `cfg=custom.yaml`.

To do this first create a copy of `default.yaml` in your current working dir with the `yolo copy-cfg` command.

This will create `default_copy.yaml`, which you can then pass as `cfg=default_copy.yaml` along with any additional args, like `imgsz=320` in this example:

!!! Example

    === "CLI"
        ```bash
        yolo copy-cfg
        yolo cfg=default_copy.yaml imgsz=320
        ```
=======
---
comments: true
description: Explore the YOLO command line interface (CLI) for easy execution of detection tasks without needing a Python environment.
keywords: YOLO CLI, command line interface, YOLO commands, detection tasks, Ultralytics, model training, model prediction
---

# Command Line Interface

The Ultralytics command line interface (CLI) provides a straightforward way to use Ultralytics YOLO models without needing a Python environment. The CLI supports running various tasks directly from the terminal using the `yolo` command, requiring no customization or Python code.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/GsXGnb-A4Kc?start=19"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Mastering Ultralytics YOLO: CLI
</p>

!!! example

    === "Syntax"

        Ultralytics `yolo` commands use the following syntax:
        ```bash
        yolo TASK MODE ARGS
        ```

        Where:
        - `TASK` (optional) is one of [detect, segment, classify, pose, obb]
        - `MODE` (required) is one of [train, val, predict, export, track, benchmark]
        - `ARGS` (optional) are any number of custom `arg=value` pairs like `imgsz=320` that override defaults.

        See all ARGS in the full [Configuration Guide](cfg.md) or with `yolo cfg`.

    === "Train"

        Train a detection model for 10 [epochs](https://www.ultralytics.com/glossary/epoch) with an initial [learning rate](https://www.ultralytics.com/glossary/learning-rate) of 0.01:

        ```bash
        yolo train data=coco8.yaml model=yolo11n.pt epochs=10 lr0=0.01
        ```

    === "Predict"

        Predict using a pretrained segmentation model on a YouTube video at image size 320:

        ```bash
        yolo predict model=yolo11n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320
        ```

    === "Val"

        Validate a pretrained detection model with a [batch size](https://www.ultralytics.com/glossary/batch-size) of 1 and image size 640:

        ```bash
        yolo val model=yolo11n.pt data=coco8.yaml batch=1 imgsz=640
        ```

    === "Export"

        Export a YOLO classification model to ONNX format with image size 224x128 (no TASK required):

        ```bash
        yolo export model=yolo11n-cls.pt format=onnx imgsz=224,128
        ```

    === "Special"

        Run special commands to view version, settings, run checks, and more:

        ```bash
        yolo help
        yolo checks
        yolo version
        yolo settings
        yolo copy-cfg
        yolo cfg
        ```

Where:

- `TASK` (optional) is one of `[detect, segment, classify, pose, obb]`. If not explicitly passed, YOLO will attempt to infer the `TASK` from the model type.
- `MODE` (required) is one of `[train, val, predict, export, track, benchmark]`
- `ARGS` (optional) are any number of custom `arg=value` pairs like `imgsz=320` that override defaults. For a full list of available `ARGS`, see the [Configuration](cfg.md) page and `defaults.yaml`.

!!! warning

    Arguments must be passed as `arg=val` pairs, separated by an equals `=` sign and delimited by spaces between pairs. Do not use `--` argument prefixes or commas `,` between arguments.

    - `yolo predict model=yolo11n.pt imgsz=640 conf=0.25` &nbsp; ✅
    - `yolo predict model yolo11n.pt imgsz 640 conf 0.25` &nbsp; ❌
    - `yolo predict --model yolo11n.pt --imgsz 640 --conf 0.25` &nbsp; ❌

## Train

Train YOLO on the COCO8 dataset for 100 epochs at image size 640. For a full list of available arguments, see the [Configuration](cfg.md) page.

!!! example

    === "Train"

        Start training YOLO11n on COCO8 for 100 epochs at image size 640:

        ```bash
        yolo detect train data=coco8.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

    === "Resume"

        Resume an interrupted training session:

        ```bash
        yolo detect train resume model=last.pt
        ```

## Val

Validate the [accuracy](https://www.ultralytics.com/glossary/accuracy) of the trained model on the COCO8 dataset. No arguments are needed as the `model` retains its training `data` and arguments as model attributes.

!!! example

    === "Official"

        Validate an official YOLO11n model:

        ```bash
        yolo detect val model=yolo11n.pt
        ```

    === "Custom"

        Validate a custom-trained model:

        ```bash
        yolo detect val model=path/to/best.pt
        ```

## Predict

Use a trained model to run predictions on images.

!!! example

    === "Official"

        Predict with an official YOLO11n model:

        ```bash
        yolo detect predict model=yolo11n.pt source='https://ultralytics.com/images/bus.jpg'
        ```

    === "Custom"

        Predict with a custom model:

        ```bash
        yolo detect predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'
        ```

## Export

Export a model to a different format like ONNX or CoreML.

!!! example

    === "Official"

        Export an official YOLO11n model to ONNX format:

        ```bash
        yolo export model=yolo11n.pt format=onnx
        ```

    === "Custom"

        Export a custom-trained model to ONNX format:

        ```bash
        yolo export model=path/to/best.pt format=onnx
        ```

Available Ultralytics export formats are in the table below. You can export to any format using the `format` argument, i.e., `format='onnx'` or `format='engine'`.

{% include "macros/export-table.md" %}

See full `export` details on the [Export](../modes/export.md) page.

## Overriding Default Arguments

Override default arguments by passing them in the CLI as `arg=value` pairs.

!!! tip

    === "Train"

        Train a detection model for 10 epochs with a learning rate of 0.01:

        ```bash
        yolo detect train data=coco8.yaml model=yolo11n.pt epochs=10 lr0=0.01
        ```

    === "Predict"

        Predict using a pretrained segmentation model on a YouTube video at image size 320:

        ```bash
        yolo segment predict model=yolo11n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320
        ```

    === "Val"

        Validate a pretrained detection model with a batch size of 1 and image size 640:

        ```bash
        yolo detect val model=yolo11n.pt data=coco8.yaml batch=1 imgsz=640
        ```

## Overriding Default Config File

Override the `default.yaml` configuration file entirely by passing a new file with the `cfg` argument, such as `cfg=custom.yaml`.

To do this, first create a copy of `default.yaml` in your current working directory with the `yolo copy-cfg` command, which creates a `default_copy.yaml` file.

You can then pass this file as `cfg=default_copy.yaml` along with any additional arguments, like `imgsz=320` in this example:

!!! example

    === "CLI"

        ```bash
        yolo copy-cfg
        yolo cfg=default_copy.yaml imgsz=320
        ```

## Solutions Commands

Ultralytics provides ready-to-use solutions for common computer vision applications through the CLI. These solutions simplify implementation of complex tasks like object counting, workout monitoring, and queue management.

!!! example

    === "Count"

        Count objects in a video or live stream:

        ```bash
        yolo solutions count show=True
        yolo solutions count source="path/to/video.mp4" # specify video file path
        ```

    === "Workout"

        Monitor workout exercises using a pose model:

        ```bash
        yolo solutions workout show=True
        yolo solutions workout source="path/to/video.mp4" # specify video file path

        # Use keypoints for ab-workouts
        yolo solutions workout kpts=[5, 11, 13] # left side
        yolo solutions workout kpts=[6, 12, 14] # right side
        ```

    === "Queue"

        Count objects in a designated queue or region:

        ```bash
        yolo solutions queue show=True
        yolo solutions queue source="path/to/video.mp4"                                # specify video file path
        yolo solutions queue region="[(20, 400), (1080, 400), (1080, 360), (20, 360)]" # configure queue coordinates
        ```

    === "Inference"

        Perform object detection, instance segmentation, or pose estimation in a web browser using Streamlit:

        ```bash
        yolo solutions inference
        yolo solutions inference model="path/to/model.pt" # use custom model
        ```

    === "Help"

        View available solutions and their options:

        ```bash
        yolo solutions help
        ```

For more information on Ultralytics solutions, visit the [Solutions](../solutions/index.md) page.

## FAQ

### How do I use the Ultralytics YOLO command line interface (CLI) for model training?

To train a model using the CLI, execute a single-line command in the terminal. For example, to train a detection model for 10 epochs with a [learning rate](https://www.ultralytics.com/glossary/learning-rate) of 0.01, run:

```bash
yolo train data=coco8.yaml model=yolo11n.pt epochs=10 lr0=0.01
```

This command uses the `train` mode with specific arguments. For a full list of available arguments, refer to the [Configuration Guide](cfg.md).

### What tasks can I perform with the Ultralytics YOLO CLI?

The Ultralytics YOLO CLI supports various tasks, including [detection](../tasks/detect.md), [segmentation](../tasks/segment.md), [classification](../tasks/classify.md), [pose estimation](../tasks/pose.md), and [oriented bounding box detection](../tasks/obb.md). You can also perform operations like:

- **Train a Model**: Run `yolo train data=<data.yaml> model=<model.pt> epochs=<num>`.
- **Run Predictions**: Use `yolo predict model=<model.pt> source=<data_source> imgsz=<image_size>`.
- **Export a Model**: Execute `yolo export model=<model.pt> format=<export_format>`.
- **Use Solutions**: Run `yolo solutions <solution_name>` for ready-made applications.

Customize each task with various arguments. For detailed syntax and examples, see the respective sections like [Train](#train), [Predict](#predict), and [Export](#export).

### How can I validate the accuracy of a trained YOLO model using the CLI?

To validate a model's [accuracy](https://www.ultralytics.com/glossary/accuracy), use the `val` mode. For example, to validate a pretrained detection model with a [batch size](https://www.ultralytics.com/glossary/batch-size) of 1 and an image size of 640, run:

```bash
yolo val model=yolo11n.pt data=coco8.yaml batch=1 imgsz=640
```

This command evaluates the model on the specified dataset and provides performance metrics like [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map), [precision](https://www.ultralytics.com/glossary/precision), and [recall](https://www.ultralytics.com/glossary/recall). For more details, refer to the [Val](#val) section.

### What formats can I export my YOLO models to using the CLI?

You can export YOLO models to various formats including ONNX, TensorRT, CoreML, TensorFlow, and more. For instance, to export a model to ONNX format, run:

```bash
yolo export model=yolo11n.pt format=onnx
```

The export command supports numerous options to optimize your model for specific deployment environments. For complete details on all available export formats and their specific parameters, visit the [Export](../modes/export.md) page.

### How do I use the pre-built solutions in the Ultralytics CLI?

Ultralytics provides ready-to-use solutions through the `solutions` command. For example, to count objects in a video:

```bash
yolo solutions count source="path/to/video.mp4"
```

These solutions require minimal configuration and provide immediate functionality for common computer vision tasks. To see all available solutions, run `yolo solutions help`. Each solution has specific parameters that can be customized to fit your needs.
>>>>>>> 0e451281e978a7c5f581cc5d9bb3d0f62e0bc632
