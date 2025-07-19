<<<<<<< HEAD
# YOLOv8/YOLOv5 Inference C++

This example demonstrates how to perform inference using YOLOv8 and YOLOv5 models in C++ with OpenCV's DNN API.

## Usage

```bash
git clone ultralytics
cd ultralytics
pip install .
cd examples/YOLOv8-CPP-Inference

# Add a **yolov8\_.onnx** and/or **yolov5\_.onnx** model(s) to the ultralytics folder.
# Edit the **main.cpp** to change the **projectBasePath** to match your user.

# Note that by default the CMake file will try and import the CUDA library to be used with the OpenCVs dnn (cuDNN) GPU Inference.
# If your OpenCV build does not use CUDA/cuDNN you can remove that import call and run the example on CPU.

mkdir build
cd build
cmake ..
make
./Yolov8CPPInference
```

## Exporting YOLOv8 and YOLOv5 Models

To export YOLOv8 models:

```commandline
yolo export model=yolov8s.pt imgsz=480,640 format=onnx opset=12
```

To export YOLOv5 models:

```commandline
python3 export.py --weights yolov5s.pt --img 480 640 --include onnx --opset 12
```

yolov8s.onnx:

![image](https://user-images.githubusercontent.com/40023722/217356132-a4cecf2e-2729-4acb-b80a-6559022d7707.png)

yolov5s.onnx:

![image](https://user-images.githubusercontent.com/40023722/217357005-07464492-d1da-42e3-98a7-fc753f87d5e6.png)

This repository utilizes OpenCV's DNN API to run ONNX exported models of YOLOv5 and YOLOv8. In theory, it should work for YOLOv6 and YOLOv7 as well, but they have not been tested. Note that the example networks are exported with rectangular (640x480) resolutions, but any exported resolution will work. You may want to use the letterbox approach for square images, depending on your use case.

The **main** branch version uses Qt as a GUI wrapper. The primary focus here is the **Inference** class file, which demonstrates how to transpose YOLOv8 models to work as YOLOv5 models.
=======
# YOLOv8/YOLOv5 C++ Inference with OpenCV DNN

This example demonstrates how to perform inference using Ultralytics YOLOv8 and YOLOv5 models in C++ leveraging the [OpenCV DNN module](https://docs.opencv.org/4.x/d6/d0f/group__dnn.html).

## 🛠️ Usage

Follow these steps to set up and run the C++ inference example:

```bash
# 1. Clone the Ultralytics repository
git clone https://github.com/ultralytics/ultralytics
cd ultralytics

# 2. Install Ultralytics Python package (needed for exporting models)
pip install .

# 3. Navigate to the C++ example directory
cd examples/YOLOv8-CPP-Inference

# 4. Export Models: Add yolov8*.onnx and/or yolov5*.onnx models (see export instructions below)
#    Place the exported ONNX models in the current directory (YOLOv8-CPP-Inference).

# 5. Update Source Code: Edit main.cpp and set the 'projectBasePath' variable
#    to the absolute path of the 'YOLOv8-CPP-Inference' directory on your system.
#    Example: std::string projectBasePath = "/path/to/your/ultralytics/examples/YOLOv8-CPP-Inference";

# 6. Configure OpenCV DNN Backend (Optional - CUDA):
#    - The default CMakeLists.txt attempts to use CUDA for GPU acceleration with OpenCV DNN.
#    - If your OpenCV build doesn't support CUDA/cuDNN, or you want CPU inference,
#      remove the CUDA-related lines from CMakeLists.txt.

# 7. Build the project
mkdir build
cd build
cmake ..
make

# 8. Run the inference executable
./Yolov8CPPInference
```

## ✨ Exporting YOLOv8 and YOLOv5 Models

You need to export your trained PyTorch models to the [ONNX](https://onnx.ai/) format to use them with OpenCV DNN.

**Exporting Ultralytics YOLOv8 Models:**

Use the Ultralytics CLI to export. Ensure you specify the desired `imgsz` and `opset`. For compatibility with this example, `opset=12` is recommended.

```bash
yolo export model=yolov8s.pt imgsz=640,480 format=onnx opset=12 # Example: 640x480 resolution
```

**Exporting YOLOv5 Models:**

Use the `export.py` script from the YOLOv5 repository structure (included within the cloned `ultralytics` repo).

```bash
# Assuming you are in the 'ultralytics' base directory after cloning
python export.py --weights yolov5s.pt --imgsz 640 480 --include onnx --opset 12 # Example: 640x480 resolution
```

Place the generated `.onnx` files (e.g., `yolov8s.onnx`, `yolov5s.onnx`) into the `ultralytics/examples/YOLOv8-CPP-Inference/` directory.

**Example Output:**

_yolov8s.onnx:_

![YOLOv8 ONNX Output](https://user-images.githubusercontent.com/40023722/217356132-a4cecf2e-2729-4acb-b80a-6559022d7707.png)

_yolov5s.onnx:_

![YOLOv5 ONNX Output](https://user-images.githubusercontent.com/40023722/217357005-07464492-d1da-42e3-98a7-fc753f87d5e6.png)

## 📝 Notes

- This repository utilizes the [OpenCV DNN API](https://docs.opencv.org/4.x/d6/d0f/group__dnn.html) to run [ONNX](https://onnx.ai/) exported models of YOLOv5 and Ultralytics YOLOv8.
- While not explicitly tested, it might theoretically work for other YOLO architectures like YOLOv6 and YOLOv7 if their ONNX export formats are compatible.
- The example models are exported with a rectangular resolution (640x480), but the code should handle models exported with different resolutions. Consider using techniques like [letterboxing](https://docs.ultralytics.com/modes/predict/#letterbox) if your input images have different aspect ratios than the model's training resolution, especially for square `imgsz` exports.
- The `main` branch version includes a simple GUI wrapper using [Qt](https://www.qt.io/). However, the core logic resides in the `Inference` class (`inference.h`, `inference.cpp`).
- A key part of the `Inference` class demonstrates how to handle the output differences between YOLOv5 and YOLOv8 models, effectively transposing YOLOv8's output format to match the structure expected from YOLOv5 for consistent post-processing.

## 🤝 Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please feel free to open an issue or submit a pull request. See our [Contributing Guide](https://docs.ultralytics.com/help/contributing/) for more details.
>>>>>>> 0e451281e978a7c5f581cc5d9bb3d0f62e0bc632
