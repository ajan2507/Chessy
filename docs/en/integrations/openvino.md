<<<<<<< HEAD
---
comments: true
description: Discover the power of deploying your Ultralytics YOLOv8 model using OpenVINO format for up to 10x speedup vs PyTorch.
keywords: ultralytics docs, YOLOv8, export YOLOv8, YOLOv8 model deployment, exporting YOLOv8, OpenVINO, OpenVINO format
---

# Intel OpenVINO Export

<img width="1024" src="https://user-images.githubusercontent.com/26833433/252345644-0cf84257-4b34-404c-b7ce-eb73dfbcaff1.png" alt="OpenVINO Ecosystem">

In this guide, we cover exporting YOLOv8 models to the [OpenVINO](https://docs.openvino.ai/) format, which can provide up to 3x [CPU](https://docs.openvino.ai/2023.0/openvino_docs_OV_UG_supported_plugins_CPU.html) speedup as well as accelerating on other Intel hardware ([iGPU](https://docs.openvino.ai/2023.0/openvino_docs_OV_UG_supported_plugins_GPU.html), [dGPU](https://docs.openvino.ai/2023.0/openvino_docs_OV_UG_supported_plugins_GPU.html), [VPU](https://docs.openvino.ai/2022.3/openvino_docs_OV_UG_supported_plugins_VPU.html), etc.).

OpenVINO, short for Open Visual Inference & Neural Network Optimization toolkit, is a comprehensive toolkit for optimizing and deploying AI inference models. Even though the name contains Visual, OpenVINO also supports various additional tasks including language, audio, time series, etc.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/kONm9nE5_Fk?si=kzquuBrxjSbntHoU"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How To Export and Optimize an Ultralytics YOLOv8 Model for Inference with OpenVINO.
</p>

## Usage Examples

Export a YOLOv8n model to OpenVINO format and run inference with the exported model.

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLOv8n PyTorch model
        model = YOLO('yolov8n.pt')

        # Export the model
        model.export(format='openvino')  # creates 'yolov8n_openvino_model/'

        # Load the exported OpenVINO model
        ov_model = YOLO('yolov8n_openvino_model/')

        # Run inference
        results = ov_model('https://ultralytics.com/images/bus.jpg')
        ```
    === "CLI"

        ```bash
        # Export a YOLOv8n PyTorch model to OpenVINO format
        yolo export model=yolov8n.pt format=openvino  # creates 'yolov8n_openvino_model/'

        # Run inference with the exported model
        yolo predict model=yolov8n_openvino_model source='https://ultralytics.com/images/bus.jpg'
        ```

## Arguments

| Key      | Value        | Description                                          |
|----------|--------------|------------------------------------------------------|
| `format` | `'openvino'` | format to export to                                  |
| `imgsz`  | `640`        | image size as scalar or (h, w) list, i.e. (640, 480) |
| `half`   | `False`      | FP16 quantization                                    |

## Benefits of OpenVINO

1. **Performance**: OpenVINO delivers high-performance inference by utilizing the power of Intel CPUs, integrated and discrete GPUs, and FPGAs.
2. **Support for Heterogeneous Execution**: OpenVINO provides an API to write once and deploy on any supported Intel hardware (CPU, GPU, FPGA, VPU, etc.).
3. **Model Optimizer**: OpenVINO provides a Model Optimizer that imports, converts, and optimizes models from popular deep learning frameworks such as PyTorch, TensorFlow, TensorFlow Lite, Keras, ONNX, PaddlePaddle, and Caffe.
4. **Ease of Use**: The toolkit comes with more than [80 tutorial notebooks](https://github.com/openvinotoolkit/openvino_notebooks) (including [YOLOv8 optimization](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/230-yolov8-optimization)) teaching different aspects of the toolkit.

## OpenVINO Export Structure

When you export a model to OpenVINO format, it results in a directory containing the following:

1. **XML file**: Describes the network topology.
2. **BIN file**: Contains the weights and biases binary data.
3. **Mapping file**: Holds mapping of original model output tensors to OpenVINO tensor names.

You can use these files to run inference with the OpenVINO Inference Engine.

## Using OpenVINO Export in Deployment

Once you have the OpenVINO files, you can use the OpenVINO Runtime to run the model. The Runtime provides a unified API to inference across all supported Intel hardware. It also provides advanced capabilities like load balancing across Intel hardware and asynchronous execution. For more information on running the inference, refer to the [Inference with OpenVINO Runtime Guide](https://docs.openvino.ai/2023.0/openvino_docs_OV_UG_OV_Runtime_User_Guide.html).

Remember, you'll need the XML and BIN files as well as any application-specific settings like input size, scale factor for normalization, etc., to correctly set up and use the model with the Runtime.

In your deployment application, you would typically do the following steps:

1. Initialize OpenVINO by creating `core = Core()`.
2. Load the model using the `core.read_model()` method.
3. Compile the model using the `core.compile_model()` function.
4. Prepare the input (image, text, audio, etc.).
5. Run inference using `compiled_model(input_data)`.

For more detailed steps and code snippets, refer to the [OpenVINO documentation](https://docs.openvino.ai/) or [API tutorial](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/002-openvino-api/002-openvino-api.ipynb).

## OpenVINO YOLOv8 Benchmarks

YOLOv8 benchmarks below were run by the Ultralytics team on 4 different model formats measuring speed and accuracy: PyTorch, TorchScript, ONNX and OpenVINO. Benchmarks were run on Intel Flex and Arc GPUs, and on Intel Xeon CPUs at FP32 precision (with the `half=False` argument).

!!! Note

    The benchmarking results below are for reference and might vary based on the exact hardware and software configuration of a system, as well as the current workload of the system at the time the benchmarks are run.

    All benchmarks run with `openvino` Python package version [2023.0.1](https://pypi.org/project/openvino/2023.0.1/).

### Intel Flex GPU

The Intel® Data Center GPU Flex Series is a versatile and robust solution designed for the intelligent visual cloud. This GPU supports a wide array of workloads including media streaming, cloud gaming, AI visual inference, and virtual desktop Infrastructure workloads. It stands out for its open architecture and built-in support for the AV1 encode, providing a standards-based software stack for high-performance, cross-architecture applications. The Flex Series GPU is optimized for density and quality, offering high reliability, availability, and scalability.

Benchmarks below run on Intel® Data Center GPU Flex 170 at FP32 precision.

<div align="center">
<img width="800" src="https://user-images.githubusercontent.com/26833433/253741543-62659bf8-1765-4d0b-b71c-8a4f9885506a.jpg" alt="Flex GPU benchmarks">
</div>

| Model   | Format      | Status | Size (MB) | mAP50-95(B) | Inference time (ms/im) |
|---------|-------------|--------|-----------|-------------|------------------------|
| YOLOv8n | PyTorch     | ✅      | 6.2       | 0.3709      | 21.79                  |
| YOLOv8n | TorchScript | ✅      | 12.4      | 0.3704      | 23.24                  |
| YOLOv8n | ONNX        | ✅      | 12.2      | 0.3704      | 37.22                  |
| YOLOv8n | OpenVINO    | ✅      | 12.3      | 0.3703      | 3.29                   |
| YOLOv8s | PyTorch     | ✅      | 21.5      | 0.4471      | 31.89                  |
| YOLOv8s | TorchScript | ✅      | 42.9      | 0.4472      | 32.71                  |
| YOLOv8s | ONNX        | ✅      | 42.8      | 0.4472      | 43.42                  |
| YOLOv8s | OpenVINO    | ✅      | 42.9      | 0.4470      | 3.92                   |
| YOLOv8m | PyTorch     | ✅      | 49.7      | 0.5013      | 50.75                  |
| YOLOv8m | TorchScript | ✅      | 99.2      | 0.4999      | 47.90                  |
| YOLOv8m | ONNX        | ✅      | 99.0      | 0.4999      | 63.16                  |
| YOLOv8m | OpenVINO    | ✅      | 49.8      | 0.4997      | 7.11                   |
| YOLOv8l | PyTorch     | ✅      | 83.7      | 0.5293      | 77.45                  |
| YOLOv8l | TorchScript | ✅      | 167.2     | 0.5268      | 85.71                  |
| YOLOv8l | ONNX        | ✅      | 166.8     | 0.5268      | 88.94                  |
| YOLOv8l | OpenVINO    | ✅      | 167.0     | 0.5264      | 9.37                   |
| YOLOv8x | PyTorch     | ✅      | 130.5     | 0.5404      | 100.09                 |
| YOLOv8x | TorchScript | ✅      | 260.7     | 0.5371      | 114.64                 |
| YOLOv8x | ONNX        | ✅      | 260.4     | 0.5371      | 110.32                 |
| YOLOv8x | OpenVINO    | ✅      | 260.6     | 0.5367      | 15.02                  |

This table represents the benchmark results for five different models (YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x) across four different formats (PyTorch, TorchScript, ONNX, OpenVINO), giving us the status, size, mAP50-95(B) metric, and inference time for each combination.

### Intel Arc GPU

Intel® Arc™ represents Intel's foray into the dedicated GPU market. The Arc™ series, designed to compete with leading GPU manufacturers like AMD and Nvidia, caters to both the laptop and desktop markets. The series includes mobile versions for compact devices like laptops, and larger, more powerful versions for desktop computers.

The Arc™ series is divided into three categories: Arc™ 3, Arc™ 5, and Arc™ 7, with each number indicating the performance level. Each category includes several models, and the 'M' in the GPU model name signifies a mobile, integrated variant.

Early reviews have praised the Arc™ series, particularly the integrated A770M GPU, for its impressive graphics performance. The availability of the Arc™ series varies by region, and additional models are expected to be released soon. Intel® Arc™ GPUs offer high-performance solutions for a range of computing needs, from gaming to content creation.

Benchmarks below run on Intel® Arc 770 GPU at FP32 precision.

<div align="center">
<img width="800" src="https://user-images.githubusercontent.com/26833433/253741545-8530388f-8fd1-44f7-a4ae-f875d59dc282.jpg" alt="Arc GPU benchmarks">
</div>

| Model   | Format      | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
|---------|-------------|--------|-----------|---------------------|------------------------|
| YOLOv8n | PyTorch     | ✅      | 6.2       | 0.3709              | 88.79                  |
| YOLOv8n | TorchScript | ✅      | 12.4      | 0.3704              | 102.66                 |
| YOLOv8n | ONNX        | ✅      | 12.2      | 0.3704              | 57.98                  |
| YOLOv8n | OpenVINO    | ✅      | 12.3      | 0.3703              | 8.52                   |
| YOLOv8s | PyTorch     | ✅      | 21.5      | 0.4471              | 189.83                 |
| YOLOv8s | TorchScript | ✅      | 42.9      | 0.4472              | 227.58                 |
| YOLOv8s | ONNX        | ✅      | 42.7      | 0.4472              | 142.03                 |
| YOLOv8s | OpenVINO    | ✅      | 42.9      | 0.4469              | 9.19                   |
| YOLOv8m | PyTorch     | ✅      | 49.7      | 0.5013              | 411.64                 |
| YOLOv8m | TorchScript | ✅      | 99.2      | 0.4999              | 517.12                 |
| YOLOv8m | ONNX        | ✅      | 98.9      | 0.4999              | 298.68                 |
| YOLOv8m | OpenVINO    | ✅      | 99.1      | 0.4996              | 12.55                  |
| YOLOv8l | PyTorch     | ✅      | 83.7      | 0.5293              | 725.73                 |
| YOLOv8l | TorchScript | ✅      | 167.1     | 0.5268              | 892.83                 |
| YOLOv8l | ONNX        | ✅      | 166.8     | 0.5268              | 576.11                 |
| YOLOv8l | OpenVINO    | ✅      | 167.0     | 0.5262              | 17.62                  |
| YOLOv8x | PyTorch     | ✅      | 130.5     | 0.5404              | 988.92                 |
| YOLOv8x | TorchScript | ✅      | 260.7     | 0.5371              | 1186.42                |
| YOLOv8x | ONNX        | ✅      | 260.4     | 0.5371              | 768.90                 |
| YOLOv8x | OpenVINO    | ✅      | 260.6     | 0.5367              | 19                     |

### Intel Xeon CPU

The Intel® Xeon® CPU is a high-performance, server-grade processor designed for complex and demanding workloads. From high-end cloud computing and virtualization to artificial intelligence and machine learning applications, Xeon® CPUs provide the power, reliability, and flexibility required for today's data centers.

Notably, Xeon® CPUs deliver high compute density and scalability, making them ideal for both small businesses and large enterprises. By choosing Intel® Xeon® CPUs, organizations can confidently handle their most demanding computing tasks and foster innovation while maintaining cost-effectiveness and operational efficiency.

Benchmarks below run on 4th Gen Intel® Xeon® Scalable CPU at FP32 precision.

<div align="center">
<img width="800" src="https://user-images.githubusercontent.com/26833433/253741546-dcd8e52a-fc38-424f-b87e-c8365b6f28dc.jpg" alt="Xeon CPU benchmarks">
</div>

| Model   | Format      | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
|---------|-------------|--------|-----------|---------------------|------------------------|
| YOLOv8n | PyTorch     | ✅      | 6.2       | 0.3709              | 24.36                  |
| YOLOv8n | TorchScript | ✅      | 12.4      | 0.3704              | 23.93                  |
| YOLOv8n | ONNX        | ✅      | 12.2      | 0.3704              | 39.86                  |
| YOLOv8n | OpenVINO    | ✅      | 12.3      | 0.3704              | 11.34                  |
| YOLOv8s | PyTorch     | ✅      | 21.5      | 0.4471              | 33.77                  |
| YOLOv8s | TorchScript | ✅      | 42.9      | 0.4472              | 34.84                  |
| YOLOv8s | ONNX        | ✅      | 42.8      | 0.4472              | 43.23                  |
| YOLOv8s | OpenVINO    | ✅      | 42.9      | 0.4471              | 13.86                  |
| YOLOv8m | PyTorch     | ✅      | 49.7      | 0.5013              | 53.91                  |
| YOLOv8m | TorchScript | ✅      | 99.2      | 0.4999              | 53.51                  |
| YOLOv8m | ONNX        | ✅      | 99.0      | 0.4999              | 64.16                  |
| YOLOv8m | OpenVINO    | ✅      | 99.1      | 0.4996              | 28.79                  |
| YOLOv8l | PyTorch     | ✅      | 83.7      | 0.5293              | 75.78                  |
| YOLOv8l | TorchScript | ✅      | 167.2     | 0.5268              | 79.13                  |
| YOLOv8l | ONNX        | ✅      | 166.8     | 0.5268              | 88.45                  |
| YOLOv8l | OpenVINO    | ✅      | 167.0     | 0.5263              | 56.23                  |
| YOLOv8x | PyTorch     | ✅      | 130.5     | 0.5404              | 96.60                  |
| YOLOv8x | TorchScript | ✅      | 260.7     | 0.5371              | 114.28                 |
| YOLOv8x | ONNX        | ✅      | 260.4     | 0.5371              | 111.02                 |
| YOLOv8x | OpenVINO    | ✅      | 260.6     | 0.5371              | 83.28                  |

### Intel Core CPU

The Intel® Core® series is a range of high-performance processors by Intel. The lineup includes Core i3 (entry-level), Core i5 (mid-range), Core i7 (high-end), and Core i9 (extreme performance). Each series caters to different computing needs and budgets, from everyday tasks to demanding professional workloads. With each new generation, improvements are made to performance, energy efficiency, and features.

Benchmarks below run on 13th Gen Intel® Core® i7-13700H CPU at FP32 precision.

<div align="center">
<img width="800" src="https://user-images.githubusercontent.com/26833433/254559985-727bfa43-93fa-4fec-a417-800f869f3f9e.jpg" alt="Core CPU benchmarks">
</div>

| Model   | Format      | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
|---------|-------------|--------|-----------|---------------------|------------------------|
| YOLOv8n | PyTorch     | ✅      | 6.2       | 0.4478              | 104.61                 |
| YOLOv8n | TorchScript | ✅      | 12.4      | 0.4525              | 112.39                 |
| YOLOv8n | ONNX        | ✅      | 12.2      | 0.4525              | 28.02                  |
| YOLOv8n | OpenVINO    | ✅      | 12.3      | 0.4504              | 23.53                  |
| YOLOv8s | PyTorch     | ✅      | 21.5      | 0.5885              | 194.83                 |
| YOLOv8s | TorchScript | ✅      | 43.0      | 0.5962              | 202.01                 |
| YOLOv8s | ONNX        | ✅      | 42.8      | 0.5962              | 65.74                  |
| YOLOv8s | OpenVINO    | ✅      | 42.9      | 0.5966              | 38.66                  |
| YOLOv8m | PyTorch     | ✅      | 49.7      | 0.6101              | 355.23                 |
| YOLOv8m | TorchScript | ✅      | 99.2      | 0.6120              | 424.78                 |
| YOLOv8m | ONNX        | ✅      | 99.0      | 0.6120              | 173.39                 |
| YOLOv8m | OpenVINO    | ✅      | 99.1      | 0.6091              | 69.80                  |
| YOLOv8l | PyTorch     | ✅      | 83.7      | 0.6591              | 593.00                 |
| YOLOv8l | TorchScript | ✅      | 167.2     | 0.6580              | 697.54                 |
| YOLOv8l | ONNX        | ✅      | 166.8     | 0.6580              | 342.15                 |
| YOLOv8l | OpenVINO    | ✅      | 167.0     | 0.0708              | 117.69                 |
| YOLOv8x | PyTorch     | ✅      | 130.5     | 0.6651              | 804.65                 |
| YOLOv8x | TorchScript | ✅      | 260.8     | 0.6650              | 921.46                 |
| YOLOv8x | ONNX        | ✅      | 260.4     | 0.6650              | 526.66                 |
| YOLOv8x | OpenVINO    | ✅      | 260.6     | 0.6619              | 158.73                 |

## Reproduce Our Results

To reproduce the Ultralytics benchmarks above on all export [formats](../modes/export.md) run this code:

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLOv8n PyTorch model
        model = YOLO('yolov8n.pt')

        # Benchmark YOLOv8n speed and accuracy on the COCO128 dataset for all all export formats
        results= model.benchmarks(data='coco128.yaml')
        ```
    === "CLI"

        ```bash
        # Benchmark YOLOv8n speed and accuracy on the COCO128 dataset for all all export formats
        yolo benchmark model=yolov8n.pt data=coco128.yaml
        ```

    Note that benchmarking results might vary based on the exact hardware and software configuration of a system, as well as the current workload of the system at the time the benchmarks are run. For the most reliable results use a dataset with a large number of images, i.e. `data='coco128.yaml' (128 val images), or `data='coco.yaml'` (5000 val images).

## Conclusion

The benchmarking results clearly demonstrate the benefits of exporting the YOLOv8 model to the OpenVINO format. Across different models and hardware platforms, the OpenVINO format consistently outperforms other formats in terms of inference speed while maintaining comparable accuracy.

For the Intel® Data Center GPU Flex Series, the OpenVINO format was able to deliver inference speeds almost 10 times faster than the original PyTorch format. On the Xeon CPU, the OpenVINO format was twice as fast as the PyTorch format. The accuracy of the models remained nearly identical across the different formats.

The benchmarks underline the effectiveness of OpenVINO as a tool for deploying deep learning models. By converting models to the OpenVINO format, developers can achieve significant performance improvements, making it easier to deploy these models in real-world applications.

For more detailed information and instructions on using OpenVINO, refer to the [official OpenVINO documentation](https://docs.openvino.ai/).
=======
---
comments: true
description: Learn to export YOLO11 models to OpenVINO format for up to 3x CPU speedup and hardware acceleration on Intel GPU and NPU.
keywords: YOLO11, OpenVINO, model export, Intel, AI inference, CPU speedup, GPU acceleration, NPU, deep learning
---

# Intel OpenVINO Export

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/openvino-ecosystem.avif" alt="OpenVINO Ecosystem">

In this guide, we cover exporting YOLO11 models to the [OpenVINO](https://docs.openvino.ai/) format, which can provide up to 3x [CPU](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html) speedup, as well as accelerating YOLO inference on Intel [GPU](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html) and [NPU](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html) hardware.

OpenVINO, short for Open Visual Inference & [Neural Network](https://www.ultralytics.com/glossary/neural-network-nn) Optimization toolkit, is a comprehensive toolkit for optimizing and deploying AI inference models. Even though the name contains Visual, OpenVINO also supports various additional tasks including language, audio, time series, etc.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/kONm9nE5_Fk?si=kzquuBrxjSbntHoU"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How To Export and Optimize an Ultralytics YOLOv8 Model for Inference with OpenVINO.
</p>

## Usage Examples

Export a YOLO11n model to OpenVINO format and run inference with the exported model.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO11n PyTorch model
        model = YOLO("yolo11n.pt")

        # Export the model
        model.export(format="openvino")  # creates 'yolo11n_openvino_model/'

        # Load the exported OpenVINO model
        ov_model = YOLO("yolo11n_openvino_model/")

        # Run inference
        results = ov_model("https://ultralytics.com/images/bus.jpg")

        # Run inference with specified device, available devices: ["intel:gpu", "intel:npu", "intel:cpu"]
        results = ov_model("https://ultralytics.com/images/bus.jpg", device="intel:gpu")
        ```

    === "CLI"

        ```bash
        # Export a YOLO11n PyTorch model to OpenVINO format
        yolo export model=yolo11n.pt format=openvino # creates 'yolo11n_openvino_model/'

        # Run inference with the exported model
        yolo predict model=yolo11n_openvino_model source='https://ultralytics.com/images/bus.jpg'

        # Run inference with specified device, available devices: ["intel:gpu", "intel:npu", "intel:cpu"]
        yolo predict model=yolo11n_openvino_model source='https://ultralytics.com/images/bus.jpg' device="intel:gpu"
        ```

## Export Arguments

| Argument   | Type             | Default        | Description                                                                                                                                                                                                                                                      |
| ---------- | ---------------- | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`            | `'openvino'`   | Target format for the exported model, defining compatibility with various deployment environments.                                                                                                                                                               |
| `imgsz`    | `int` or `tuple` | `640`          | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions.                                                                                                                                |
| `half`     | `bool`           | `False`        | Enables FP16 (half-precision) quantization, reducing model size and potentially speeding up inference on supported hardware.                                                                                                                                     |
| `int8`     | `bool`           | `False`        | Activates INT8 quantization, further compressing the model and speeding up inference with minimal [accuracy](https://www.ultralytics.com/glossary/accuracy) loss, primarily for edge devices.                                                                    |
| `dynamic`  | `bool`           | `False`        | Allows dynamic input sizes, enhancing flexibility in handling varying image dimensions.                                                                                                                                                                          |
| `nms`      | `bool`           | `False`        | Adds Non-Maximum Suppression (NMS), essential for accurate and efficient detection post-processing.                                                                                                                                                              |
| `batch`    | `int`            | `1`            | Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode.                                                                                                                          |
| `data`     | `str`            | `'coco8.yaml'` | Path to the [dataset](https://docs.ultralytics.com/datasets/) configuration file (default: `coco8.yaml`), essential for quantization.                                                                                                                            |
| `fraction` | `float`          | `1.0`          | Specifies the fraction of the dataset to use for INT8 quantization calibration. Allows for calibrating on a subset of the full dataset, useful for experiments or when resources are limited. If not specified with INT8 enabled, the full dataset will be used. |

For more details about the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).

!!! warning

    OpenVINO™ is compatible with most Intel® processors but to ensure optimal performance:

    1. Verify OpenVINO™ support
        Check whether your Intel® chip is officially supported by OpenVINO™ using [Intel's compatibility list](https://docs.openvino.ai/2025/about-openvino/release-notes-openvino/system-requirements.html).

    2. Identify your accelerator
        Determine if your processor includes an integrated NPU (Neural Processing Unit) or GPU (integrated GPU) by consulting [Intel's hardware guide](https://www.intel.com/content/www/us/en/support/articles/000097597/processors.html).

    3. Install the latest drivers
        If your chip supports an NPU or GPU but OpenVINO™ isn't detecting it, you may need to install or update the associated drivers. Follow the [driver‑installation instructions](https://medium.com/openvino-toolkit/how-to-run-openvino-on-a-linux-ai-pc-52083ce14a98) to enable full acceleration.

    By following these three steps, you can ensure OpenVINO™ runs optimally on your Intel® hardware.

## Benefits of OpenVINO

1. **Performance**: OpenVINO delivers high-performance inference by utilizing the power of Intel CPUs, integrated and discrete GPUs, and FPGAs.
2. **Support for Heterogeneous Execution**: OpenVINO provides an API to write once and deploy on any supported Intel hardware (CPU, GPU, FPGA, VPU, etc.).
3. **Model Optimizer**: OpenVINO provides a Model Optimizer that imports, converts, and optimizes models from popular [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) frameworks such as PyTorch, [TensorFlow](https://www.ultralytics.com/glossary/tensorflow), TensorFlow Lite, Keras, ONNX, PaddlePaddle, and Caffe.
4. **Ease of Use**: The toolkit comes with more than [80 tutorial notebooks](https://github.com/openvinotoolkit/openvino_notebooks) (including [YOLOv8 optimization](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/yolov8-optimization)) teaching different aspects of the toolkit.

## OpenVINO Export Structure

When you export a model to OpenVINO format, it results in a directory containing the following:

1. **XML file**: Describes the network topology.
2. **BIN file**: Contains the weights and biases binary data.
3. **Mapping file**: Holds mapping of original model output tensors to OpenVINO tensor names.

You can use these files to run inference with the OpenVINO Inference Engine.

## Using OpenVINO Export in Deployment

Once your model is successfully exported to the OpenVINO format, you have two primary options for running inference:

1. Use the `ultralytics` package, which provides a high-level API and wraps the OpenVINO Runtime.

2. Use the native `openvino` package for more advanced or customized control over inference behavior.

### Inference with Ultralytics

The ultralytics package allows you to easily run inference using the exported OpenVINO model via the predict method. You can also specify the target device (e.g., `intel:gpu`, `intel:npu`, `intel:cpu`) using the device argument.

```python
from ultralytics import YOLO

# Load the exported OpenVINO model
ov_model = YOLO("yolo11n_openvino_model/")  # the path of your exported OpenVINO model
# Run inference with the exported model
ov_model.predict(device="intel:gpu")  # specify the device you want to run inference on
```

This approach is ideal for fast prototyping or deployment when you don't need full control over the inference pipeline.

### Inference with OpenVINO Runtime

The openvino Runtime provides a unified API to inference across all supported Intel hardware. It also provides advanced capabilities like load balancing across Intel hardware and asynchronous execution. For more information on running the inference, refer to the [YOLO11 notebooks](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/yolov11-optimization).

Remember, you'll need the XML and BIN files as well as any application-specific settings like input size, scale factor for normalization, etc., to correctly set up and use the model with the Runtime.

In your deployment application, you would typically do the following steps:

1. Initialize OpenVINO by creating `core = Core()`.
2. Load the model using the `core.read_model()` method.
3. Compile the model using the `core.compile_model()` function.
4. Prepare the input (image, text, audio, etc.).
5. Run inference using `compiled_model(input_data)`.

For more detailed steps and code snippets, refer to the [OpenVINO documentation](https://docs.openvino.ai/) or [API tutorial](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/openvino-api/openvino-api.ipynb).

## OpenVINO YOLO11 Benchmarks

The Ultralytics team benchmarked YOLO11 across various model formats and [precision](https://www.ultralytics.com/glossary/precision), evaluating speed and accuracy on different Intel devices compatible with OpenVINO.

!!! note

    The benchmarking results below are for reference and might vary based on the exact hardware and software configuration of a system, as well as the current workload of the system at the time the benchmarks are run.

    All benchmarks run with `openvino` Python package version [2025.1.0](https://pypi.org/project/openvino/2025.1.0/).

### Intel Core CPU

The Intel® Core® series is a range of high-performance processors by Intel. The lineup includes Core i3 (entry-level), Core i5 (mid-range), Core i7 (high-end), and Core i9 (extreme performance). Each series caters to different computing needs and budgets, from everyday tasks to demanding professional workloads. With each new generation, improvements are made to performance, energy efficiency, and features.

Benchmarks below run on 12th Gen Intel® Core® i9-12900KS CPU at FP32 precision.

<div align="center">
<img width="800" src="https://github.com/ultralytics/docs/releases/download/0/openvino-corei9.avif" alt="Core CPU benchmarks">
</div>

??? abstract "Detailed Benchmark Results"

    | Model   | Format      | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
    | ------- | ----------- | ------ | --------- | ------------------- | ---------------------- |
    | YOLO11n | PyTorch     | ✅     | 5.4       | 0.5071              | 21.00                  |
    | YOLO11n | TorchScript | ✅     | 10.5      | 0.5077              | 21.39                  |
    | YOLO11n | ONNX        | ✅     | 10.2      | 0.5077              | 15.55                  |
    | YOLO11n | OpenVINO    | ✅     | 10.4      | 0.5077              | 11.49                  |
    | YOLO11s | PyTorch     | ✅     | 18.4      | 0.5770              | 43.16                  |
    | YOLO11s | TorchScript | ✅     | 36.6      | 0.5781              | 50.06                  |
    | YOLO11s | ONNX        | ✅     | 36.3      | 0.5781              | 31.53                  |
    | YOLO11s | OpenVINO    | ✅     | 36.4      | 0.5781              | 30.82                  |
    | YOLO11m | PyTorch     | ✅     | 38.8      | 0.6257              | 110.60                 |
    | YOLO11m | TorchScript | ✅     | 77.3      | 0.6306              | 128.09                 |
    | YOLO11m | ONNX        | ✅     | 76.9      | 0.6306              | 76.06                  |
    | YOLO11m | OpenVINO    | ✅     | 77.1      | 0.6306              | 79.38                  |
    | YOLO11l | PyTorch     | ✅     | 49.0      | 0.6367              | 150.38                 |
    | YOLO11l | TorchScript | ✅     | 97.7      | 0.6408              | 172.57                 |
    | YOLO11l | ONNX        | ✅     | 97.0      | 0.6408              | 108.91                 |
    | YOLO11l | OpenVINO    | ✅     | 97.3      | 0.6408              | 102.30                 |
    | YOLO11x | PyTorch     | ✅     | 109.3     | 0.6989              | 272.72                 |
    | YOLO11x | TorchScript | ✅     | 218.1     | 0.6900              | 320.86                 |
    | YOLO11x | ONNX        | ✅     | 217.5     | 0.6900              | 196.20                 |
    | YOLO11x | OpenVINO    | ✅     | 217.8     | 0.6900              | 195.32                 |

### Intel® Core™ Ultra

The Intel® Core™ Ultra™ series represents a new benchmark in high-performance computing, engineered to meet the evolving demands of modern users—from gamers and creators to professionals leveraging AI. This next-generation lineup is more than a traditional CPU series; it combines powerful CPU cores, integrated high-performance GPU capabilities, and a dedicated Neural Processing Unit (NPU) within a single chip, offering a unified solution for diverse and intensive computing workloads.

At the heart of the Intel® Core Ultra™ architecture is a hybrid design that enables exceptional performance across traditional processing tasks, GPU-accelerated workloads, and AI-driven operations. The inclusion of the NPU enhances on-device AI inference, enabling faster, more efficient machine learning and data processing across a wide range of applications.

The Core Ultra™ family includes various models tailored for different performance needs, with options ranging from energy-efficient designs to high-power variants marked by the "H" designation—ideal for laptops and compact form factors that demand serious computing power. Across the lineup, users benefit from the synergy of CPU, GPU, and NPU integration, delivering remarkable efficiency, responsiveness, and multitasking capabilities.

As part of Intel's ongoing innovation, the Core Ultra™ series sets a new standard for future-ready computing. With multiple models available and more on the horizon, this series underscores Intel's commitment to delivering cutting-edge solutions for the next generation of intelligent, AI-enhanced devices.

Benchmarks below run on Intel® Core™ Ultra™ 7 258V and Intel® Core™ Ultra™ 7 265K at FP32 and INT8 precision.

#### Intel® Core™ Ultra™ 7 258V

!!! tip "Benchmarks"

    === "Integrated Intel® Arc™ GPU"

        <div align="center">
        <img width="800" src="https://github.com/ultralytics/docs/releases/download/0/openvino-ultra7-258V-gpu.avif" alt="Intel Core Ultra GPU benchmarks">
        </div>

        ??? abstract "Detailed Benchmark Results"

            | Model   | Format   | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
            | ------- | -------- | --------- | ------ | --------- | ------------------- | ---------------------- |
            | YOLO11n | PyTorch  | FP32      | ✅     | 5.4       | 0.5052              | 32.27                  |
            | YOLO11n | OpenVINO | FP32      | ✅     | 10.4      | 0.5068              | 11.84                  |
            | YOLO11n | OpenVINO | INT8      | ✅     | 3.3       | 0.4969              | 11.24                  |
            | YOLO11s | PyTorch  | FP32      | ✅     | 18.4      | 0.5776              | 92.09                  |
            | YOLO11s | OpenVINO | FP32      | ✅     | 36.4      | 0.5797              | 14.82                  |
            | YOLO11s | OpenVINO | INT8      | ✅     | 9.8       | 0.5751              | 12.88                  |
            | YOLO11m | PyTorch  | FP32      | ✅     | 38.8      | 0.6262              | 277.24                 |
            | YOLO11m | OpenVINO | FP32      | ✅     | 77.1      | 0.6306              | 22.94                  |
            | YOLO11m | OpenVINO | INT8      | ✅     | 20.2      | 0.6126              | 17.85                  |
            | YOLO11l | PyTorch  | FP32      | ✅     | 49.0      | 0.6361              | 348.97                 |
            | YOLO11l | OpenVINO | FP32      | ✅     | 97.3      | 0.6365              | 27.34                  |
            | YOLO11l | OpenVINO | INT8      | ✅     | 25.7      | 0.6242              | 20.83                  |
            | YOLO11x | PyTorch  | FP32      | ✅     | 109.3     | 0.6984              | 666.07                 |
            | YOLO11x | OpenVINO | FP32      | ✅     | 217.8     | 0.6890              | 39.09                  |
            | YOLO11x | OpenVINO | INT8      | ✅     | 55.9      | 0.6856              | 30.60                  |

    === "Intel® Lunar Lake CPU"

        <div align="center">
        <img width="800" src="https://github.com/ultralytics/docs/releases/download/0/openvino-ultra7-258V-cpu.avif" alt="Intel Core Ultra CPU benchmarks">
        </div>

        ??? abstract "Detailed Benchmark Results"

            | Model   | Format   | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
            | ------- | -------- | --------- | ------ | --------- | ------------------- | ---------------------- |
            | YOLO11n | PyTorch  | FP32      | ✅     | 5.4       | 0.5052              | 32.27                  |
            | YOLO11n | OpenVINO | FP32      | ✅     | 10.4      | 0.5077              | 32.55                  |
            | YOLO11n | OpenVINO | INT8      | ✅     | 3.3       | 0.4980              | 22.98                  |
            | YOLO11s | PyTorch  | FP32      | ✅     | 18.4      | 0.5776              | 92.09                  |
            | YOLO11s | OpenVINO | FP32      | ✅     | 36.4      | 0.5782              | 98.38                  |
            | YOLO11s | OpenVINO | INT8      | ✅     | 9.8       | 0.5745              | 52.84                  |
            | YOLO11m | PyTorch  | FP32      | ✅     | 38.8      | 0.6262              | 277.24                 |
            | YOLO11m | OpenVINO | FP32      | ✅     | 77.1      | 0.6307              | 275.74                 |
            | YOLO11m | OpenVINO | INT8      | ✅     | 20.2      | 0.6172              | 132.63                 |
            | YOLO11l | PyTorch  | FP32      | ✅     | 49.0      | 0.6361              | 348.97                 |
            | YOLO11l | OpenVINO | FP32      | ✅     | 97.3      | 0.6361              | 348.97                 |
            | YOLO11l | OpenVINO | INT8      | ✅     | 25.7      | 0.6240              | 171.36                 |
            | YOLO11x | PyTorch  | FP32      | ✅     | 109.3     | 0.6984              | 666.07                 |
            | YOLO11x | OpenVINO | FP32      | ✅     | 217.8     | 0.6900              | 783.16                 |
            | YOLO11x | OpenVINO | INT8      | ✅     | 55.9      | 0.6890              | 346.82                 |


    === "Integrated Intel® AI Boost NPU"

        <div align="center">
        <img width="800" src="https://github.com/ultralytics/docs/releases/download/0/openvino-ultra7-258V-npu.avif" alt="Intel Core Ultra NPU benchmarks">
        </div>

        ??? abstract "Detailed Benchmark Results"

            | Model   | Format   | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
            | ------- | -------- | --------- | ------ | --------- | ------------------- | ---------------------- |
            | YOLO11n | PyTorch  | FP32      | ✅     | 5.4       | 0.5052              | 32.27                  |
            | YOLO11n | OpenVINO | FP32      | ✅     | 10.4      | 0.5085              | 8.33                   |
            | YOLO11n | OpenVINO | INT8      | ✅     | 3.3       | 0.5019              | 8.91                   |
            | YOLO11s | PyTorch  | FP32      | ✅     | 18.4      | 0.5776              | 92.09                  |
            | YOLO11s | OpenVINO | FP32      | ✅     | 36.4      | 0.5788              | 9.72                   |
            | YOLO11s | OpenVINO | INT8      | ✅     | 9.8       | 0.5710              | 10.58                  |
            | YOLO11m | PyTorch  | FP32      | ✅     | 38.8      | 0.6262              | 277.24                 |
            | YOLO11m | OpenVINO | FP32      | ✅     | 77.1      | 0.6301              | 19.41                  |
            | YOLO11m | OpenVINO | INT8      | ✅     | 20.2      | 0.6124              | 18.26                  |
            | YOLO11l | PyTorch  | FP32      | ✅     | 49.0      | 0.6361              | 348.97                 |
            | YOLO11l | OpenVINO | FP32      | ✅     | 97.3      | 0.6362              | 23.70                  |
            | YOLO11l | OpenVINO | INT8      | ✅     | 25.7      | 0.6240              | 21.40                  |
            | YOLO11x | PyTorch  | FP32      | ✅     | 109.3     | 0.6984              | 666.07                 |
            | YOLO11x | OpenVINO | FP32      | ✅     | 217.8     | 0.6892              | 43.91                  |
            | YOLO11x | OpenVINO | INT8      | ✅     | 55.9      | 0.6890              | 34.04                  |

#### Intel® Core™ Ultra™ 7 265K

!!! tip "Benchmarks"

    === "Integrated Intel® Arc™ GPU"

        <div align="center">
        <img width="800" src="https://github.com/ultralytics/docs/releases/download/0/openvino-ultra7-265K-gpu.avif" alt="Intel Core Ultra GPU benchmarks">
        </div>

        ??? abstract "Detailed Benchmark Results"

            | Model   | Format   | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
            | ------- | -------- | --------- | ------ | --------- | ------------------- | ---------------------- |
            | YOLO11n | PyTorch  | FP32      | ✅     | 5.4       | 0.5072              | 16.29                  |
            | YOLO11n | OpenVINO | FP32      | ✅     | 10.4      | 0.5079              | 13.13                  |
            | YOLO11n | OpenVINO | INT8      | ✅     | 3.3       | 0.4976              | 8.86                   |
            | YOLO11s | PyTorch  | FP32      | ✅     | 18.4      | 0.5771              | 39.61                  |
            | YOLO11s | OpenVINO | FP32      | ✅     | 36.4      | 0.5808              | 18.26                  |
            | YOLO11s | OpenVINO | INT8      | ✅     | 9.8       | 0.5726              | 13.24                  |
            | YOLO11m | PyTorch  | FP32      | ✅     | 38.8      | 0.6258              | 100.65                 |
            | YOLO11m | OpenVINO | FP32      | ✅     | 77.1      | 0.6310              | 43.50                  |
            | YOLO11m | OpenVINO | INT8      | ✅     | 20.2      | 0.6137              | 20.90                  |
            | YOLO11l | PyTorch  | FP32      | ✅     | 49.0      | 0.6367              | 131.37                 |
            | YOLO11l | OpenVINO | FP32      | ✅     | 97.3      | 0.6371              | 54.52                  |
            | YOLO11l | OpenVINO | INT8      | ✅     | 25.7      | 0.6226              | 27.36                  |
            | YOLO11x | PyTorch  | FP32      | ✅     | 109.3     | 0.6990              | 212.45                 |
            | YOLO11x | OpenVINO | FP32      | ✅     | 217.8     | 0.6884              | 112.76                 |
            | YOLO11x | OpenVINO | INT8      | ✅     | 55.9      | 0.6900              | 52.06                  |


    === "Intel® Arrow Lake CPU"

        <div align="center">
        <img width="800" src="https://github.com/ultralytics/docs/releases/download/0/openvino-ultra7-265K-cpu.avif" alt="Intel Core Ultra CPU benchmarks">
        </div>

        ??? abstract "Detailed Benchmark Results"

            | Model   | Format   | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
            | ------- | -------- | --------- | ------ | --------- | ------------------- | ---------------------- |
            | YOLO11n | PyTorch  | FP32      | ✅     | 5.4       | 0.5072              | 16.29                  |
            | YOLO11n | OpenVINO | FP32      | ✅     | 10.4      | 0.5077              | 15.04                  |
            | YOLO11n | OpenVINO | INT8      | ✅     | 3.3       | 0.4980              | 11.60                  |
            | YOLO11s | PyTorch  | FP32      | ✅     | 18.4      | 0.5771              | 39.61                  |
            | YOLO11s | OpenVINO | FP32      | ✅     | 36.4      | 0.5782              | 33.45                  |
            | YOLO11s | OpenVINO | INT8      | ✅     | 9.8       | 0.5745              | 20.64                  |
            | YOLO11m | PyTorch  | FP32      | ✅     | 38.8      | 0.6258              | 100.65                 |
            | YOLO11m | OpenVINO | FP32      | ✅     | 77.1      | 0.6307              | 81.15                  |
            | YOLO11m | OpenVINO | INT8      | ✅     | 20.2      | 0.6172              | 44.63                  |
            | YOLO11l | PyTorch  | FP32      | ✅     | 49.0      | 0.6367              | 131.37                 |
            | YOLO11l | OpenVINO | FP32      | ✅     | 97.3      | 0.6409              | 103.77                 |
            | YOLO11l | OpenVINO | INT8      | ✅     | 25.7      | 0.6240              | 58.00                  |
            | YOLO11x | PyTorch  | FP32      | ✅     | 109.3     | 0.6990              | 212.45                 |
            | YOLO11x | OpenVINO | FP32      | ✅     | 217.8     | 0.6900              | 208.37                 |
            | YOLO11x | OpenVINO | INT8      | ✅     | 55.9      | 0.6897              | 113.04                 |


    === "Integrated Intel® AI Boost NPU"

        <div align="center">
        <img width="800" src="https://github.com/ultralytics/docs/releases/download/0/openvino-ultra7-265K-npu.avif" alt="Intel Core Ultra NPU benchmarks">
        </div>

        ??? abstract "Detailed Benchmark Results"

            | Model   | Format   | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
            | ------- | -------- | --------- | ------ | --------- | ------------------- | ---------------------- |
            | YOLO11n | PyTorch  | FP32      | ✅     | 5.4       | 0.5072              | 16.29                  |
            | YOLO11n | OpenVINO | FP32      | ✅     | 10.4      | 0.5075              | 8.02                   |
            | YOLO11n | OpenVINO | INT8      | ✅     | 3.3       | 0.3656              | 9.28                   |
            | YOLO11s | PyTorch  | FP32      | ✅     | 18.4      | 0.5771              | 39.61                  |
            | YOLO11s | OpenVINO | FP32      | ✅     | 36.4      | 0.5801              | 13.12                  |
            | YOLO11s | OpenVINO | INT8      | ✅     | 9.8       | 0.5686              | 13.12                  |
            | YOLO11m | PyTorch  | FP32      | ✅     | 38.8      | 0.6258              | 100.65                 |
            | YOLO11m | OpenVINO | FP32      | ✅     | 77.1      | 0.6310              | 29.88                  |
            | YOLO11m | OpenVINO | INT8      | ✅     | 20.2      | 0.6111              | 26.32                  |
            | YOLO11l | PyTorch  | FP32      | ✅     | 49.0      | 0.6367              | 131.37                 |
            | YOLO11l | OpenVINO | FP32      | ✅     | 97.3      | 0.6356              | 37.08                  |
            | YOLO11l | OpenVINO | INT8      | ✅     | 25.7      | 0.6245              | 30.81                  |
            | YOLO11x | PyTorch  | FP32      | ✅     | 109.3     | 0.6990              | 212.45                 |
            | YOLO11x | OpenVINO | FP32      | ✅     | 217.8     | 0.6894              | 68.48                  |
            | YOLO11x | OpenVINO | INT8      | ✅     | 55.9      | 0.6417              | 49.76                  |

## Intel® Arc GPU

Intel® Arc™ is Intel's line of discrete graphics cards designed for high-performance gaming, content creation, and AI workloads. The Arc series features advanced GPU architectures that support real-time ray tracing, AI-enhanced graphics, and high-resolution gaming. With a focus on performance and efficiency, Intel® Arc™ aims to compete with other leading GPU brands while providing unique features like hardware-accelerated AV1 encoding and support for the latest graphics APIs.

Benchmarks below run on Intel Arc A770 and Intel Arc B580 at FP32 and INT8 precision.

### Intel Arc A770

<div align="center">
<img width="800" src="https://github.com/ultralytics/docs/releases/download/0/openvino-arc-a770-gpu.avif" alt="Intel Core Ultra CPU benchmarks">
</div>

??? abstract "Detailed Benchmark Results"

    | Model   | Format   | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
    | ------- | -------- | --------- | ------ | --------- | ------------------- | ---------------------- |
    | YOLO11n | PyTorch  | FP32      | ✅     | 5.4       | 0.5072              | 16.29                  |
    | YOLO11n | OpenVINO | FP32      | ✅     | 10.4      | 0.5073              | 6.98                   |
    | YOLO11n | OpenVINO | INT8      | ✅     | 3.3       | 0.4978              | 7.24                   |
    | YOLO11s | PyTorch  | FP32      | ✅     | 18.4      | 0.5771              | 39.61                  |
    | YOLO11s | OpenVINO | FP32      | ✅     | 36.4      | 0.5798              | 9.41                   |
    | YOLO11s | OpenVINO | INT8      | ✅     | 9.8       | 0.5751              | 8.72                   |
    | YOLO11m | PyTorch  | FP32      | ✅     | 38.8      | 0.6258              | 100.65                 |
    | YOLO11m | OpenVINO | FP32      | ✅     | 77.1      | 0.6311              | 14.88                  |
    | YOLO11m | OpenVINO | INT8      | ✅     | 20.2      | 0.6126              | 11.97                  |
    | YOLO11l | PyTorch  | FP32      | ✅     | 49.0      | 0.6367              | 131.37                 |
    | YOLO11l | OpenVINO | FP32      | ✅     | 97.3      | 0.6364              | 19.17                  |
    | YOLO11l | OpenVINO | INT8      | ✅     | 25.7      | 0.6241              | 15.75                  |
    | YOLO11x | PyTorch  | FP32      | ✅     | 109.3     | 0.6990              | 212.45                 |
    | YOLO11x | OpenVINO | FP32      | ✅     | 217.8     | 0.6888              | 18.13                  |
    | YOLO11x | OpenVINO | INT8      | ✅     | 55.9      | 0.6930              | 18.91                  |

### Intel Arc B580

<div align="center">
<img width="800" src="https://github.com/ultralytics/docs/releases/download/0/openvino-arc-b580-gpu.avif" alt="Intel Core Ultra CPU benchmarks">
</div>

??? abstract "Detailed Benchmark Results"

    | Model   | Format   | Precision | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
    | ------- | -------- | --------- | ------ | --------- | ------------------- | ---------------------- |
    | YOLO11n | PyTorch  | FP32      | ✅     | 5.4       | 0.5072              | 16.29                  |
    | YOLO11n | OpenVINO | FP32      | ✅     | 10.4      | 0.5072              | 4.27                   |
    | YOLO11n | OpenVINO | INT8      | ✅     | 3.3       | 0.4981              | 4.33                   |
    | YOLO11s | PyTorch  | FP32      | ✅     | 18.4      | 0.5771              | 39.61                  |
    | YOLO11s | OpenVINO | FP32      | ✅     | 36.4      | 0.5789              | 5.04                   |
    | YOLO11s | OpenVINO | INT8      | ✅     | 9.8       | 0.5746              | 4.97                   |
    | YOLO11m | PyTorch  | FP32      | ✅     | 38.8      | 0.6258              | 100.65                 |
    | YOLO11m | OpenVINO | FP32      | ✅     | 77.1      | 0.6306              | 6.45                   |
    | YOLO11m | OpenVINO | INT8      | ✅     | 20.2      | 0.6125              | 6.28                   |
    | YOLO11l | PyTorch  | FP32      | ✅     | 49.0      | 0.6367              | 131.37                 |
    | YOLO11l | OpenVINO | FP32      | ✅     | 97.3      | 0.6360              | 8.23                   |
    | YOLO11l | OpenVINO | INT8      | ✅     | 25.7      | 0.6236              | 8.49                   |
    | YOLO11x | PyTorch  | FP32      | ✅     | 109.3     | 0.6990              | 212.45                 |
    | YOLO11x | OpenVINO | FP32      | ✅     | 217.8     | 0.6889              | 11.10                  |
    | YOLO11x | OpenVINO | INT8      | ✅     | 55.9      | 0.6924              | 10.30                  |

## Reproduce Our Results

To reproduce the Ultralytics benchmarks above on all export [formats](../modes/export.md) run this code:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO11n PyTorch model
        model = YOLO("yolo11n.pt")

        # Benchmark YOLO11n speed and accuracy on the COCO128 dataset for all export formats
        results = model.benchmark(data="coco128.yaml")
        ```

    === "CLI"

        ```bash
        # Benchmark YOLO11n speed and accuracy on the COCO128 dataset for all export formats
        yolo benchmark model=yolo11n.pt data=coco128.yaml
        ```

    Note that benchmarking results might vary based on the exact hardware and software configuration of a system, as well as the current workload of the system at the time the benchmarks are run. For the most reliable results use a dataset with a large number of images, i.e. `data='coco.yaml'` (5000 val images).

## Conclusion

The benchmarking results clearly demonstrate the benefits of exporting the YOLO11 model to the OpenVINO format. Across different models and hardware platforms, the OpenVINO format consistently outperforms other formats in terms of inference speed while maintaining comparable accuracy.

The benchmarks underline the effectiveness of OpenVINO as a tool for deploying deep learning models. By converting models to the OpenVINO format, developers can achieve significant performance improvements, making it easier to deploy these models in real-world applications.

For more detailed information and instructions on using OpenVINO, refer to the [official OpenVINO documentation](https://docs.openvino.ai/).

## FAQ

### How do I export YOLO11 models to OpenVINO format?

Exporting YOLO11 models to the OpenVINO format can significantly enhance CPU speed and enable GPU and NPU accelerations on Intel hardware. To export, you can use either Python or CLI as shown below:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO11n PyTorch model
        model = YOLO("yolo11n.pt")

        # Export the model
        model.export(format="openvino")  # creates 'yolo11n_openvino_model/'
        ```

    === "CLI"

        ```bash
        # Export a YOLO11n PyTorch model to OpenVINO format
        yolo export model=yolo11n.pt format=openvino # creates 'yolo11n_openvino_model/'
        ```

For more information, refer to the [export formats documentation](../modes/export.md).

### What are the benefits of using OpenVINO with YOLO11 models?

Using Intel's OpenVINO toolkit with YOLO11 models offers several benefits:

1. **Performance**: Achieve up to 3x speedup on CPU inference and leverage Intel GPUs and NPUs for acceleration.
2. **Model Optimizer**: Convert, optimize, and execute models from popular frameworks like PyTorch, TensorFlow, and ONNX.
3. **Ease of Use**: Over 80 tutorial notebooks are available to help users get started, including ones for YOLO11.
4. **Heterogeneous Execution**: Deploy models on various Intel hardware with a unified API.

For detailed performance comparisons, visit our [benchmarks section](#openvino-yolo11-benchmarks).

### How can I run inference using a YOLO11 model exported to OpenVINO?

After exporting a YOLO11n model to OpenVINO format, you can run inference using Python or CLI:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported OpenVINO model
        ov_model = YOLO("yolo11n_openvino_model/")

        # Run inference
        results = ov_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Run inference with the exported model
        yolo predict model=yolo11n_openvino_model source='https://ultralytics.com/images/bus.jpg'
        ```

Refer to our [predict mode documentation](../modes/predict.md) for more details.

### Why should I choose Ultralytics YOLO11 over other models for OpenVINO export?

Ultralytics YOLO11 is optimized for real-time object detection with high accuracy and speed. Specifically, when combined with OpenVINO, YOLO11 provides:

- Up to 3x speedup on Intel CPUs
- Seamless deployment on Intel GPUs and NPUs
- Consistent and comparable accuracy across various export formats

For in-depth performance analysis, check our detailed [YOLO11 benchmarks](#openvino-yolo11-benchmarks) on different hardware.

### Can I benchmark YOLO11 models on different formats such as PyTorch, ONNX, and OpenVINO?

Yes, you can benchmark YOLO11 models in various formats including PyTorch, TorchScript, ONNX, and OpenVINO. Use the following code snippet to run benchmarks on your chosen dataset:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO11n PyTorch model
        model = YOLO("yolo11n.pt")

        # Benchmark YOLO11n speed and [accuracy](https://www.ultralytics.com/glossary/accuracy) on the COCO8 dataset for all export formats
        results = model.benchmark(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # Benchmark YOLO11n speed and accuracy on the COCO8 dataset for all export formats
        yolo benchmark model=yolo11n.pt data=coco8.yaml
        ```

For detailed benchmark results, refer to our [benchmarks section](#openvino-yolo11-benchmarks) and [export formats](../modes/export.md) documentation.
>>>>>>> 0e451281e978a7c5f581cc5d9bb3d0f62e0bc632
