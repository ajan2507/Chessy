<<<<<<< HEAD
---
comments: true
description: Learn how to profile speed and accuracy of YOLOv8 across various export formats; get insights on mAP50-95, accuracy_top5 metrics, and more.
keywords: Ultralytics, YOLOv8, benchmarking, speed profiling, accuracy profiling, mAP50-95, accuracy_top5, ONNX, OpenVINO, TensorRT, YOLO export formats
---

# Model Benchmarking with Ultralytics YOLO

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics YOLO ecosystem and integrations">

## Introduction

Once your model is trained and validated, the next logical step is to evaluate its performance in various real-world scenarios. Benchmark mode in Ultralytics YOLOv8 serves this purpose by providing a robust framework for assessing the speed and accuracy of your model across a range of export formats.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/j8uQc0qB91s?start=105"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Ultralytics Modes Tutorial: Benchmark
</p>

## Why Is Benchmarking Crucial?

- **Informed Decisions:** Gain insights into the trade-offs between speed and accuracy.
- **Resource Allocation:** Understand how different export formats perform on different hardware.
- **Optimization:** Learn which export format offers the best performance for your specific use case.
- **Cost Efficiency:** Make more efficient use of hardware resources based on benchmark results.

### Key Metrics in Benchmark Mode

- **mAP50-95:** For object detection, segmentation, and pose estimation.
- **accuracy_top5:** For image classification.
- **Inference Time:** Time taken for each image in milliseconds.

### Supported Export Formats

- **ONNX:** For optimal CPU performance
- **TensorRT:** For maximal GPU efficiency
- **OpenVINO:** For Intel hardware optimization
- **CoreML, TensorFlow SavedModel, and More:** For diverse deployment needs.

!!! Tip "Tip"

    * Export to ONNX or OpenVINO for up to 3x CPU speedup.
    * Export to TensorRT for up to 5x GPU speedup.

## Usage Examples

Run YOLOv8n benchmarks on all supported export formats including ONNX, TensorRT etc. See Arguments section below for a full list of export arguments.

!!! Example

    === "Python"

        ```python
        from ultralytics.utils.benchmarks import benchmark

        # Benchmark on GPU
        benchmark(model='yolov8n.pt', data='coco8.yaml', imgsz=640, half=False, device=0)
        ```
    === "CLI"

        ```bash
        yolo benchmark model=yolov8n.pt data='coco8.yaml' imgsz=640 half=False device=0
        ```

## Arguments

Arguments such as `model`, `data`, `imgsz`, `half`, `device`, and `verbose` provide users with the flexibility to fine-tune the benchmarks to their specific needs and compare the performance of different export formats with ease.

| Key       | Value   | Description                                                           |
|-----------|---------|-----------------------------------------------------------------------|
| `model`   | `None`  | path to model file, i.e. yolov8n.pt, yolov8n.yaml                     |
| `data`    | `None`  | path to YAML referencing the benchmarking dataset (under `val` label) |
| `imgsz`   | `640`   | image size as scalar or (h, w) list, i.e. (640, 480)                  |
| `half`    | `False` | FP16 quantization                                                     |
| `int8`    | `False` | INT8 quantization                                                     |
| `device`  | `None`  | device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu  |
| `verbose` | `False` | do not continue on error (bool), or val floor threshold (float)       |

## Export Formats

Benchmarks will attempt to run automatically on all possible export formats below.

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

See full `export` details in the [Export](https://docs.ultralytics.com/modes/export/) page.
=======
---
comments: true
description: Learn how to evaluate your YOLO11 model's performance in real-world scenarios using benchmark mode. Optimize speed, accuracy, and resource allocation across export formats.
keywords: model benchmarking, YOLO11, Ultralytics, performance evaluation, export formats, ONNX, TensorRT, OpenVINO, CoreML, TensorFlow, optimization, mAP50-95, inference time
---

# Model Benchmarking with Ultralytics YOLO

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-yolov8-ecosystem-integrations.avif" alt="Ultralytics YOLO ecosystem and integrations">

## Benchmark Visualization

!!! tip "Refresh Browser"

    You may need to refresh the page to view the graphs correctly due to potential cookie issues.

<script async src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script defer src="../../javascript/benchmark.js"></script>

<canvas id="modelComparisonChart" width="1024" height="400"></canvas>

## Introduction

Once your model is trained and validated, the next logical step is to evaluate its performance in various real-world scenarios. Benchmark mode in Ultralytics YOLO11 serves this purpose by providing a robust framework for assessing the speed and [accuracy](https://www.ultralytics.com/glossary/accuracy) of your model across a range of export formats.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/rEQlAaevEFc"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Benchmark Ultralytics YOLO11 Models | How to Compare Model Performance on Different Hardware?
</p>

## Why Is Benchmarking Crucial?

- **Informed Decisions:** Gain insights into the trade-offs between speed and accuracy.
- **Resource Allocation:** Understand how different export formats perform on different hardware.
- **Optimization:** Learn which export format offers the best performance for your specific use case.
- **Cost Efficiency:** Make more efficient use of hardware resources based on benchmark results.

### Key Metrics in Benchmark Mode

- **mAP50-95:** For [object detection](https://www.ultralytics.com/glossary/object-detection), segmentation, and pose estimation.
- **accuracy_top5:** For [image classification](https://www.ultralytics.com/glossary/image-classification).
- **Inference Time:** Time taken for each image in milliseconds.

### Supported Export Formats

- **ONNX:** For optimal CPU performance
- **TensorRT:** For maximal GPU efficiency
- **OpenVINO:** For Intel hardware optimization
- **CoreML, TensorFlow SavedModel, and More:** For diverse deployment needs.

!!! tip

    * Export to ONNX or OpenVINO for up to 3x CPU speedup.
    * Export to TensorRT for up to 5x GPU speedup.

## Usage Examples

Run YOLO11n benchmarks on all supported export formats including ONNX, TensorRT etc. See Arguments section below for a full list of export arguments.

!!! example

    === "Python"

        ```python
        from ultralytics.utils.benchmarks import benchmark

        # Benchmark on GPU
        benchmark(model="yolo11n.pt", data="coco8.yaml", imgsz=640, half=False, device=0)

        # Benchmark specific export format
        benchmark(model="yolo11n.pt", data="coco8.yaml", imgsz=640, format="onnx")
        ```

    === "CLI"

        ```bash
        yolo benchmark model=yolo11n.pt data='coco8.yaml' imgsz=640 half=False device=0

        # Benchmark specific export format
        yolo benchmark model=yolo11n.pt data='coco8.yaml' imgsz=640 format=onnx
        ```

## Arguments

Arguments such as `model`, `data`, `imgsz`, `half`, `device`, `verbose` and `format` provide users with the flexibility to fine-tune the benchmarks to their specific needs and compare the performance of different export formats with ease.

| Key       | Default Value | Description                                                                                                                                                                                             |
| --------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`   | `None`        | Specifies the path to the model file. Accepts both `.pt` and `.yaml` formats, e.g., `"yolo11n.pt"` for pre-trained models or configuration files.                                                       |
| `data`    | `None`        | Path to a YAML file defining the dataset for benchmarking, typically including paths and settings for [validation data](https://www.ultralytics.com/glossary/validation-data). Example: `"coco8.yaml"`. |
| `imgsz`   | `640`         | The input image size for the model. Can be a single integer for square images or a tuple `(width, height)` for non-square, e.g., `(640, 480)`.                                                          |
| `half`    | `False`       | Enables FP16 (half-precision) inference, reducing memory usage and possibly increasing speed on compatible hardware. Use `half=True` to enable.                                                         |
| `int8`    | `False`       | Activates INT8 quantization for further optimized performance on supported devices, especially useful for edge devices. Set `int8=True` to use.                                                         |
| `device`  | `None`        | Defines the computation device(s) for benchmarking, such as `"cpu"` or `"cuda:0"`.                                                                                                                      |
| `verbose` | `False`       | Controls the level of detail in logging output. Set `verbose=True` for detailed logs.                                                                                                                   |
| `format`  | `''`          | Benchmark the model on a single export format. i.e `format=onnx`                                                                                                                                        |

## Export Formats

Benchmarks will attempt to run automatically on all possible export formats listed below. Alternatively, you can run benchmarks for a specific format by using the `format` argument, which accepts any of the formats mentioned below.

{% include "macros/export-table.md" %}

See full `export` details in the [Export](../modes/export.md) page.

## FAQ

### How do I benchmark my YOLO11 model's performance using Ultralytics?

Ultralytics YOLO11 offers a Benchmark mode to assess your model's performance across different export formats. This mode provides insights into key metrics such as [mean Average Precision](https://www.ultralytics.com/glossary/mean-average-precision-map) (mAP50-95), accuracy, and inference time in milliseconds. To run benchmarks, you can use either Python or CLI commands. For example, to benchmark on a GPU:

!!! example

    === "Python"

        ```python
        from ultralytics.utils.benchmarks import benchmark

        # Benchmark on GPU
        benchmark(model="yolo11n.pt", data="coco8.yaml", imgsz=640, half=False, device=0)
        ```

    === "CLI"

        ```bash
        yolo benchmark model=yolo11n.pt data='coco8.yaml' imgsz=640 half=False device=0
        ```

For more details on benchmark arguments, visit the [Arguments](#arguments) section.

### What are the benefits of exporting YOLO11 models to different formats?

Exporting YOLO11 models to different formats such as [ONNX](https://docs.ultralytics.com/integrations/onnx/), [TensorRT](https://docs.ultralytics.com/integrations/tensorrt/), and [OpenVINO](https://docs.ultralytics.com/integrations/openvino/) allows you to optimize performance based on your deployment environment. For instance:

- **ONNX:** Provides up to 3x CPU speedup.
- **TensorRT:** Offers up to 5x GPU speedup.
- **OpenVINO:** Specifically optimized for Intel hardware.

These formats enhance both the speed and accuracy of your models, making them more efficient for various real-world applications. Visit the [Export](../modes/export.md) page for complete details.

### Why is benchmarking crucial in evaluating YOLO11 models?

Benchmarking your YOLO11 models is essential for several reasons:

- **Informed Decisions:** Understand the trade-offs between speed and accuracy.
- **Resource Allocation:** Gauge the performance across different hardware options.
- **Optimization:** Determine which export format offers the best performance for specific use cases.
- **Cost Efficiency:** Optimize hardware usage based on benchmark results.

Key metrics such as mAP50-95, Top-5 accuracy, and inference time help in making these evaluations. Refer to the [Key Metrics](#key-metrics-in-benchmark-mode) section for more information.

### Which export formats are supported by YOLO11, and what are their advantages?

YOLO11 supports a variety of export formats, each tailored for specific hardware and use cases:

- **ONNX:** Best for CPU performance.
- **TensorRT:** Ideal for GPU efficiency.
- **OpenVINO:** Optimized for Intel hardware.
- **CoreML & [TensorFlow](https://www.ultralytics.com/glossary/tensorflow):** Useful for iOS and general ML applications.

For a complete list of supported formats and their respective advantages, check out the [Supported Export Formats](#supported-export-formats) section.

### What arguments can I use to fine-tune my YOLO11 benchmarks?

When running benchmarks, several arguments can be customized to suit specific needs:

- **model:** Path to the model file (e.g., "yolo11n.pt").
- **data:** Path to a YAML file defining the dataset (e.g., "coco8.yaml").
- **imgsz:** The input image size, either as a single integer or a tuple.
- **half:** Enable FP16 inference for better performance.
- **int8:** Activate INT8 quantization for edge devices.
- **device:** Specify the computation device (e.g., "cpu", "cuda:0").
- **verbose:** Control the level of logging detail.

For a full list of arguments, refer to the [Arguments](#arguments) section.
>>>>>>> 0e451281e978a7c5f581cc5d9bb3d0f62e0bc632
