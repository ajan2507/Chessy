<<<<<<< HEAD
---
comments: true
description: Step-by-step guide on exporting your YOLOv8 models to various format like ONNX, TensorRT, CoreML and more for deployment. Explore now!.
keywords: YOLO, YOLOv8, Ultralytics, Model export, ONNX, TensorRT, CoreML, TensorFlow SavedModel, OpenVINO, PyTorch, export model
---

# Model Export with Ultralytics YOLO

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics YOLO ecosystem and integrations">

## Introduction

The ultimate goal of training a model is to deploy it for real-world applications. Export mode in Ultralytics YOLOv8 offers a versatile range of options for exporting your trained model to different formats, making it deployable across various platforms and devices. This comprehensive guide aims to walk you through the nuances of model exporting, showcasing how to achieve maximum compatibility and performance.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/WbomGeoOT_k?si=aGmuyooWftA0ue9X"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How To Export Custom Trained Ultralytics YOLOv8 Model and Run Live Inference on Webcam.
</p>

## Why Choose YOLOv8's Export Mode?

- **Versatility:** Export to multiple formats including ONNX, TensorRT, CoreML, and more.
- **Performance:** Gain up to 5x GPU speedup with TensorRT and 3x CPU speedup with ONNX or OpenVINO.
- **Compatibility:** Make your model universally deployable across numerous hardware and software environments.
- **Ease of Use:** Simple CLI and Python API for quick and straightforward model exporting.

### Key Features of Export Mode

Here are some of the standout functionalities:

- **One-Click Export:** Simple commands for exporting to different formats.
- **Batch Export:** Export batched-inference capable models.
- **Optimized Inference:** Exported models are optimized for quicker inference times.
- **Tutorial Videos:** In-depth guides and tutorials for a smooth exporting experience.

!!! Tip "Tip"

    * Export to ONNX or OpenVINO for up to 3x CPU speedup.
    * Export to TensorRT for up to 5x GPU speedup.

## Usage Examples

Export a YOLOv8n model to a different format like ONNX or TensorRT. See Arguments section below for a full list of export arguments.

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n.pt')  # load an official model
        model = YOLO('path/to/best.pt')  # load a custom trained model

        # Export the model
        model.export(format='onnx')
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n.pt format=onnx  # export official model
        yolo export model=path/to/best.pt format=onnx  # export custom trained model
        ```

## Arguments

Export settings for YOLO models refer to the various configurations and options used to save or export the model for use in other environments or platforms. These settings can affect the model's performance, size, and compatibility with different systems. Some common YOLO export settings include the format of the exported model file (e.g. ONNX, TensorFlow SavedModel), the device on which the model will be run (e.g. CPU, GPU), and the presence of additional features such as masks or multiple labels per box. Other factors that may affect the export process include the specific task the model is being used for and the requirements or constraints of the target environment or platform. It is important to carefully consider and configure these settings to ensure that the exported model is optimized for the intended use case and can be used effectively in the target environment.

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

## Export Formats

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
=======
---
comments: true
description: Learn how to export your YOLO11 model to various formats like ONNX, TensorRT, and CoreML. Achieve maximum compatibility and performance.
keywords: YOLO11, Model Export, ONNX, TensorRT, CoreML, Ultralytics, AI, Machine Learning, Inference, Deployment
---

# Model Export with Ultralytics YOLO

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-yolov8-ecosystem-integrations.avif" alt="Ultralytics YOLO ecosystem and integrations">

## Introduction

The ultimate goal of training a model is to deploy it for real-world applications. Export mode in Ultralytics YOLO11 offers a versatile range of options for exporting your trained model to different formats, making it deployable across various platforms and devices. This comprehensive guide aims to walk you through the nuances of model exporting, showcasing how to achieve maximum compatibility and performance.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/WbomGeoOT_k?si=aGmuyooWftA0ue9X"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How To Export Custom Trained Ultralytics YOLO Model and Run Live Inference on Webcam.
</p>

## Why Choose YOLO11's Export Mode?

- **Versatility:** Export to multiple formats including [ONNX](../integrations/onnx.md), [TensorRT](../integrations/tensorrt.md), [CoreML](../integrations/coreml.md), and more.
- **Performance:** Gain up to 5x GPU speedup with TensorRT and 3x CPU speedup with ONNX or [OpenVINO](../integrations/openvino.md).
- **Compatibility:** Make your model universally deployable across numerous hardware and software environments.
- **Ease of Use:** Simple CLI and Python API for quick and straightforward model exporting.

### Key Features of Export Mode

Here are some of the standout functionalities:

- **One-Click Export:** Simple commands for exporting to different formats.
- **Batch Export:** Export batched-inference capable models.
- **Optimized Inference:** Exported models are optimized for quicker inference times.
- **Tutorial Videos:** In-depth guides and tutorials for a smooth exporting experience.

!!! tip

    * Export to [ONNX](../integrations/onnx.md) or [OpenVINO](../integrations/openvino.md) for up to 3x CPU speedup.
    * Export to [TensorRT](../integrations/tensorrt.md) for up to 5x GPU speedup.

## Usage Examples

Export a YOLO11n model to a different format like ONNX or TensorRT. See the Arguments section below for a full list of export arguments.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom trained model

        # Export the model
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n.pt format=onnx      # export official model
        yolo export model=path/to/best.pt format=onnx # export custom trained model
        ```

## Arguments

This table details the configurations and options available for exporting YOLO models to different formats. These settings are critical for optimizing the exported model's performance, size, and compatibility across various platforms and environments. Proper configuration ensures that the model is ready for deployment in the intended application with optimal efficiency.

{% include "macros/export-args.md" %}

Adjusting these parameters allows for customization of the export process to fit specific requirements, such as deployment environment, hardware constraints, and performance targets. Selecting the appropriate format and settings is essential for achieving the best balance between model size, speed, and [accuracy](https://www.ultralytics.com/glossary/accuracy).

## Export Formats

Available YOLO11 export formats are in the table below. You can export to any format using the `format` argument, i.e. `format='onnx'` or `format='engine'`. You can predict or validate directly on exported models, i.e. `yolo predict model=yolo11n.onnx`. Usage examples are shown for your model after export completes.

{% include "macros/export-table.md" %}

## FAQ

### How do I export a YOLO11 model to ONNX format?

Exporting a YOLO11 model to ONNX format is straightforward with Ultralytics. It provides both Python and CLI methods for exporting models.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom trained model

        # Export the model
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n.pt format=onnx      # export official model
        yolo export model=path/to/best.pt format=onnx # export custom trained model
        ```

For more details on the process, including advanced options like handling different input sizes, refer to the [ONNX integration guide](../integrations/onnx.md).

### What are the benefits of using TensorRT for model export?

Using TensorRT for model export offers significant performance improvements. YOLO11 models exported to TensorRT can achieve up to a 5x GPU speedup, making it ideal for real-time inference applications.

- **Versatility:** Optimize models for a specific hardware setup.
- **Speed:** Achieve faster inference through advanced optimizations.
- **Compatibility:** Integrate smoothly with NVIDIA hardware.

To learn more about integrating TensorRT, see the [TensorRT integration guide](../integrations/tensorrt.md).

### How do I enable INT8 quantization when exporting my YOLO11 model?

INT8 quantization is an excellent way to compress the model and speed up inference, especially on edge devices. Here's how you can enable INT8 quantization:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")  # Load a model
        model.export(format="engine", int8=True)
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n.pt format=engine int8=True # export TensorRT model with INT8 quantization
        ```

INT8 quantization can be applied to various formats, such as [TensorRT](../integrations/tensorrt.md), [OpenVINO](../integrations/openvino.md), and [CoreML](../integrations/coreml.md). For optimal quantization results, provide a representative [dataset](https://docs.ultralytics.com/datasets/) using the `data` parameter.

### Why is dynamic input size important when exporting models?

Dynamic input size allows the exported model to handle varying image dimensions, providing flexibility and optimizing processing efficiency for different use cases. When exporting to formats like [ONNX](../integrations/onnx.md) or [TensorRT](../integrations/tensorrt.md), enabling dynamic input size ensures that the model can adapt to different input shapes seamlessly.

To enable this feature, use the `dynamic=True` flag during export:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")
        model.export(format="onnx", dynamic=True)
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n.pt format=onnx dynamic=True
        ```

Dynamic input sizing is particularly useful for applications where input dimensions may vary, such as video processing or when handling images from different sources.

### What are the key export arguments to consider for optimizing model performance?

Understanding and configuring export arguments is crucial for optimizing model performance:

- **`format:`** The target format for the exported model (e.g., `onnx`, `torchscript`, `tensorflow`).
- **`imgsz:`** Desired image size for the model input (e.g., `640` or `(height, width)`).
- **`half:`** Enables FP16 quantization, reducing model size and potentially speeding up inference.
- **`optimize:`** Applies specific optimizations for mobile or constrained environments.
- **`int8:`** Enables INT8 quantization, highly beneficial for [edge AI](https://www.ultralytics.com/blog/deploying-computer-vision-applications-on-edge-ai-devices) deployments.

For deployment on specific hardware platforms, consider using specialized export formats like [TensorRT](../integrations/tensorrt.md) for NVIDIA GPUs, [CoreML](../integrations/coreml.md) for Apple devices, or [Edge TPU](../integrations/edge-tpu.md) for Google Coral devices.
>>>>>>> 0e451281e978a7c5f581cc5d9bb3d0f62e0bc632
