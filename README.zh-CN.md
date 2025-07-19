<<<<<<< HEAD
<div align="center">
  <p>
    <a href="https://yolovision.ultralytics.com/" target="_blank">
      <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/im/banner-yolo-vision-2023.png" alt="YOLO Vision banner"></a>
  </p>

[中文](https://docs.ultralytics.com/zh/) | [한국어](https://docs.ultralytics.com/ko/) | [日本語](https://docs.ultralytics.com/ja/) | [Русский](https://docs.ultralytics.com/ru/) | [Deutsch](https://docs.ultralytics.com/de/) | [Français](https://docs.ultralytics.com/fr/) | [Español](https://docs.ultralytics.com/es/) | [Português](https://docs.ultralytics.com/pt/) | [हिन्दी](https://docs.ultralytics.com/hi/) | [العربية](https://docs.ultralytics.com/ar/) <br>

<div>
    <a href="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml"><img src="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml/badge.svg" alt="Ultralytics CI"></a>
    <a href="https://codecov.io/github/ultralytics/ultralytics"><img src="https://codecov.io/github/ultralytics/ultralytics/branch/main/graph/badge.svg?token=HHW7IIVFVY" alt="Ultralytics Code Coverage"></a>
    <a href="https://zenodo.org/badge/latestdoi/264818686"><img src="https://zenodo.org/badge/264818686.svg" alt="YOLOv8 Citation"></a>
    <a href="https://hub.docker.com/r/ultralytics/ultralytics"><img src="https://img.shields.io/docker/pulls/ultralytics/ultralytics?logo=docker" alt="Docker Pulls"></a>
    <a href="https://ultralytics.com/discord"><img alt="Discord" src="https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue"></a>
    <br>
    <a href="https://console.paperspace.com/github/ultralytics/ultralytics"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient"></a>
    <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
    <a href="https://www.kaggle.com/ultralytics/yolov8"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
</div>
<br>

[Ultralytics](https://ultralytics.com) [YOLOv8](https://github.com/ultralytics/ultralytics) 是一款前沿、最先进（SOTA）的模型，基于先前 YOLO 版本的成功，引入了新功能和改进，进一步提升性能和灵活性。YOLOv8 设计快速、准确且易于使用，使其成为各种物体检测与跟踪、实例分割、图像分类和姿态估计任务的绝佳选择。

我们希望这里的资源能帮助您充分利用 YOLOv8。请浏览 YOLOv8 <a href="https://docs.ultralytics.com/">文档</a> 了解详细信息，在 <a href="https://github.com/ultralytics/ultralytics/issues/new/choose">GitHub</a> 上提交问题以获得支持，并加入我们的 <a href="https://ultralytics.com/discord">Discord</a> 社区进行问题和讨论！

如需申请企业许可，请在 [Ultralytics Licensing](https://ultralytics.com/license) 处填写表格

<img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/yolo-comparison-plots.png" alt="YOLOv8 performance plots"></a>

<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="2%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="2%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="2%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://youtube.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="2%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="2%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://www.instagram.com/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-instagram.png" width="2%" alt="Ultralytics Instagram"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://ultralytics.com/discord"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="2%" alt="Ultralytics Discord"></a>
</div>
</div>

## <div align="center">文档</div>

请参阅下面的快速安装和使用示例，以及 [YOLOv8 文档](https://docs.ultralytics.com) 上有关训练、验证、预测和部署的完整文档。

<details open>
<summary>安装</summary>

使用Pip在一个[**Python>=3.8**](https://www.python.org/)环境中安装`ultralytics`包，此环境还需包含[**PyTorch>=1.8**](https://pytorch.org/get-started/locally/)。这也会安装所有必要的[依赖项](https://github.com/ultralytics/ultralytics/blob/main/requirements.txt)。

[![PyPI version](https://badge.fury.io/py/ultralytics.svg)](https://badge.fury.io/py/ultralytics) [![Downloads](https://static.pepy.tech/badge/ultralytics)](https://pepy.tech/project/ultralytics)

```bash
pip install ultralytics
```

如需使用包括[Conda](https://anaconda.org/conda-forge/ultralytics)、[Docker](https://hub.docker.com/r/ultralytics/ultralytics)和Git在内的其他安装方法，请参考[快速入门指南](https://docs.ultralytics.com/quickstart)。

</details>

<details open>
<summary>Usage</summary>

#### CLI

YOLOv8 可以在命令行界面（CLI）中直接使用，只需输入 `yolo` 命令：

```bash
yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'
```

`yolo` 可用于各种任务和模式，并接受其他参数，例如 `imgsz=640`。查看 YOLOv8 [CLI 文档](https://docs.ultralytics.com/usage/cli)以获取示例。

#### Python

YOLOv8 也可以在 Python 环境中直接使用，并接受与上述 CLI 示例中相同的[参数](https://docs.ultralytics.com/usage/cfg/)：

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("yolov8n.yaml")  # 从头开始构建新模型
model = YOLO("yolov8n.pt")  # 加载预训练模型（建议用于训练）

# 使用模型
model.train(data="coco128.yaml", epochs=3)  # 训练模型
metrics = model.val()  # 在验证集上评估模型性能
results = model("https://ultralytics.com/images/bus.jpg")  # 对图像进行预测
success = model.export(format="onnx")  # 将模型导出为 ONNX 格式
```

查看 YOLOv8 [Python 文档](https://docs.ultralytics.com/usage/python)以获取更多示例。

</details>

## <div align="center">模型</div>

在[COCO](https://docs.ultralytics.com/datasets/detect/coco)数据集上预训练的YOLOv8 [检测](https://docs.ultralytics.com/tasks/detect)，[分割](https://docs.ultralytics.com/tasks/segment)和[姿态](https://docs.ultralytics.com/tasks/pose)模型可以在这里找到，以及在[ImageNet](https://docs.ultralytics.com/datasets/classify/imagenet)数据集上预训练的YOLOv8 [分类](https://docs.ultralytics.com/tasks/classify)模型。所有的检测，分割和姿态模型都支持[追踪](https://docs.ultralytics.com/modes/track)模式。

<img width="1024" src="https://raw.githubusercontent.com/ultralytics/assets/main/im/banner-tasks.png" alt="Ultralytics YOLO supported tasks">

所有[模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models)在首次使用时会自动从最新的Ultralytics [发布版本](https://github.com/ultralytics/assets/releases)下载。

<details open><summary>检测 (COCO)</summary>

查看[检测文档](https://docs.ultralytics.com/tasks/detect/)以获取这些在[COCO](https://docs.ultralytics.com/datasets/detect/coco/)上训练的模型的使用示例，其中包括80个预训练类别。

| 模型                                                                                   | 尺寸<br><sup>(像素) | mAP<sup>val<br>50-95 | 速度<br><sup>CPU ONNX<br>(ms) | 速度<br><sup>A100 TensorRT<br>(ms) | 参数<br><sup>(M) | FLOPs<br><sup>(B) |
| ------------------------------------------------------------------------------------ | --------------- | -------------------- | --------------------------- | -------------------------------- | -------------- | ----------------- |
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt) | 640             | 37.3                 | 80.4                        | 0.99                             | 3.2            | 8.7               |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt) | 640             | 44.9                 | 128.4                       | 1.20                             | 11.2           | 28.6              |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) | 640             | 50.2                 | 234.7                       | 1.83                             | 25.9           | 78.9              |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt) | 640             | 52.9                 | 375.2                       | 2.39                             | 43.7           | 165.2             |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt) | 640             | 53.9                 | 479.1                       | 3.53                             | 68.2           | 257.8             |

- **mAP<sup>val</sup>** 值是基于单模型单尺度在 [COCO val2017](http://cocodataset.org) 数据集上的结果。 <br>通过 `yolo val detect data=coco.yaml device=0` 复现
- **速度** 是使用 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 实例对 COCO val 图像进行平均计算的。 <br>通过 `yolo val detect data=coco.yaml batch=1 device=0|cpu` 复现

</details>

<details><summary>检测（Open Image V7）</summary>

查看[检测文档](https://docs.ultralytics.com/tasks/detect/)以获取这些在[Open Image V7](https://docs.ultralytics.com/datasets/detect/open-images-v7/)上训练的模型的使用示例，其中包括600个预训练类别。

| 模型                                                                                        | 尺寸<br><sup>(像素) | mAP<sup>验证<br>50-95 | 速度<br><sup>CPU ONNX<br>(毫秒) | 速度<br><sup>A100 TensorRT<br>(毫秒) | 参数<br><sup>(M) | 浮点运算<br><sup>(B) |
| ----------------------------------------------------------------------------------------- | --------------- | ------------------- | --------------------------- | -------------------------------- | -------------- | ---------------- |
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-oiv7.pt) | 640             | 18.4                | 142.4                       | 1.21                             | 3.5            | 10.5             |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-oiv7.pt) | 640             | 27.7                | 183.1                       | 1.40                             | 11.4           | 29.7             |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-oiv7.pt) | 640             | 33.6                | 408.5                       | 2.26                             | 26.2           | 80.6             |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-oiv7.pt) | 640             | 34.9                | 596.9                       | 2.43                             | 44.1           | 167.4            |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-oiv7.pt) | 640             | 36.3                | 860.6                       | 3.56                             | 68.7           | 260.6            |

- **mAP<sup>验证</sup>** 值适用于在[Open Image V7](https://docs.ultralytics.com/datasets/detect/open-images-v7/)数据集上的单模型单尺度。 <br>通过 `yolo val detect data=open-images-v7.yaml device=0` 以复现。
- **速度** 在使用[Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/)实例对Open Image V7验证图像进行平均测算。 <br>通过 `yolo val detect data=open-images-v7.yaml batch=1 device=0|cpu` 以复现。

</details>

<details><summary>分割 (COCO)</summary>

查看[分割文档](https://docs.ultralytics.com/tasks/segment/)以获取这些在[COCO-Seg](https://docs.ultralytics.com/datasets/segment/coco/)上训练的模型的使用示例，其中包括80个预训练类别。

| 模型                                                                                           | 尺寸<br><sup>(像素) | mAP<sup>box<br>50-95 | mAP<sup>mask<br>50-95 | 速度<br><sup>CPU ONNX<br>(ms) | 速度<br><sup>A100 TensorRT<br>(ms) | 参数<br><sup>(M) | FLOPs<br><sup>(B) |
| -------------------------------------------------------------------------------------------- | --------------- | -------------------- | --------------------- | --------------------------- | -------------------------------- | -------------- | ----------------- |
| [YOLOv8n-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt) | 640             | 36.7                 | 30.5                  | 96.1                        | 1.21                             | 3.4            | 12.6              |
| [YOLOv8s-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt) | 640             | 44.6                 | 36.8                  | 155.7                       | 1.47                             | 11.8           | 42.6              |
| [YOLOv8m-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt) | 640             | 49.9                 | 40.8                  | 317.0                       | 2.18                             | 27.3           | 110.2             |
| [YOLOv8l-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-seg.pt) | 640             | 52.3                 | 42.6                  | 572.4                       | 2.79                             | 46.0           | 220.5             |
| [YOLOv8x-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt) | 640             | 53.4                 | 43.4                  | 712.1                       | 4.02                             | 71.8           | 344.1             |

- **mAP<sup>val</sup>** 值是基于单模型单尺度在 [COCO val2017](http://cocodataset.org) 数据集上的结果。 <br>通过 `yolo val segment data=coco-seg.yaml device=0` 复现
- **速度** 是使用 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 实例对 COCO val 图像进行平均计算的。 <br>通过 `yolo val segment data=coco-seg.yaml batch=1 device=0|cpu` 复现

</details>

<details><summary>姿态 (COCO)</summary>

查看[姿态文档](https://docs.ultralytics.com/tasks/pose/)以获取这些在[COCO-Pose](https://docs.ultralytics.com/datasets/pose/coco/)上训练的模型的使用示例，其中包括1个预训练类别，即人。

| 模型                                                                                                   | 尺寸<br><sup>(像素) | mAP<sup>pose<br>50-95 | mAP<sup>pose<br>50 | 速度<br><sup>CPU ONNX<br>(ms) | 速度<br><sup>A100 TensorRT<br>(ms) | 参数<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------------------------------------------------------------------------------------------------- | --------------- | --------------------- | ------------------ | --------------------------- | -------------------------------- | -------------- | ----------------- |
| [YOLOv8n-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt)       | 640             | 50.4                  | 80.1               | 131.8                       | 1.18                             | 3.3            | 9.2               |
| [YOLOv8s-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt)       | 640             | 60.0                  | 86.2               | 233.2                       | 1.42                             | 11.6           | 30.2              |
| [YOLOv8m-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-pose.pt)       | 640             | 65.0                  | 88.8               | 456.3                       | 2.00                             | 26.4           | 81.0              |
| [YOLOv8l-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-pose.pt)       | 640             | 67.6                  | 90.0               | 784.5                       | 2.59                             | 44.4           | 168.6             |
| [YOLOv8x-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose.pt)       | 640             | 69.2                  | 90.2               | 1607.1                      | 3.73                             | 69.4           | 263.2             |
| [YOLOv8x-pose-p6](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose-p6.pt) | 1280            | 71.6                  | 91.2               | 4088.7                      | 10.04                            | 99.1           | 1066.4            |

- **mAP<sup>val</sup>** 值是基于单模型单尺度在 [COCO Keypoints val2017](http://cocodataset.org) 数据集上的结果。 <br>通过 `yolo val pose data=coco-pose.yaml device=0` 复现
- **速度** 是使用 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 实例对 COCO val 图像进行平均计算的。 <br>通过 `yolo val pose data=coco-pose.yaml batch=1 device=0|cpu` 复现

</details>

<details><summary>分类 (ImageNet)</summary>

查看[分类文档](https://docs.ultralytics.com/tasks/classify/)以获取这些在[ImageNet](https://docs.ultralytics.com/datasets/classify/imagenet/)上训练的模型的使用示例，其中包括1000个预训练类别。

| 模型                                                                                           | 尺寸<br><sup>(像素) | acc<br><sup>top1 | acc<br><sup>top5 | 速度<br><sup>CPU ONNX<br>(ms) | 速度<br><sup>A100 TensorRT<br>(ms) | 参数<br><sup>(M) | FLOPs<br><sup>(B) at 640 |
| -------------------------------------------------------------------------------------------- | --------------- | ---------------- | ---------------- | --------------------------- | -------------------------------- | -------------- | ------------------------ |
| [YOLOv8n-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-cls.pt) | 224             | 66.6             | 87.0             | 12.9                        | 0.31                             | 2.7            | 4.3                      |
| [YOLOv8s-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-cls.pt) | 224             | 72.3             | 91.1             | 23.4                        | 0.35                             | 6.4            | 13.5                     |
| [YOLOv8m-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-cls.pt) | 224             | 76.4             | 93.2             | 85.4                        | 0.62                             | 17.0           | 42.7                     |
| [YOLOv8l-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-cls.pt) | 224             | 78.0             | 94.1             | 163.0                       | 0.87                             | 37.5           | 99.7                     |
| [YOLOv8x-cls](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-cls.pt) | 224             | 78.4             | 94.3             | 232.0                       | 1.01                             | 57.4           | 154.8                    |

- **acc** 值是模型在 [ImageNet](https://www.image-net.org/) 数据集验证集上的准确率。 <br>通过 `yolo val classify data=path/to/ImageNet device=0` 复现
- **速度** 是使用 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 实例对 ImageNet val 图像进行平均计算的。 <br>通过 `yolo val classify data=path/to/ImageNet batch=1 device=0|cpu` 复现

</details>

## <div align="center">集成</div>

我们与领先的AI平台的关键整合扩展了Ultralytics产品的功能，增强了数据集标签化、训练、可视化和模型管理等任务。探索Ultralytics如何与[Roboflow](https://roboflow.com/?ref=ultralytics)、ClearML、[Comet](https://bit.ly/yolov8-readme-comet)、Neural Magic以及[OpenVINO](https://docs.ultralytics.com/integrations/openvino)合作，优化您的AI工作流程。

<br>
<a href="https://bit.ly/ultralytics_hub" target="_blank">
<img width="100%" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics active learning integrations"></a>
<br>
<br>

<div align="center">
  <a href="https://roboflow.com/?ref=ultralytics">
    <img src="https://github.com/ultralytics/assets/raw/main/partners/logo-roboflow.png" width="10%" alt="Roboflow logo"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="15%" height="0" alt="space">
  <a href="https://cutt.ly/yolov5-readme-clearml">
    <img src="https://github.com/ultralytics/assets/raw/main/partners/logo-clearml.png" width="10%" alt="ClearML logo"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="15%" height="0" alt="space">
  <a href="https://bit.ly/yolov8-readme-comet">
    <img src="https://github.com/ultralytics/assets/raw/main/partners/logo-comet.png" width="10%" alt="Comet ML logo"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="15%" height="0" alt="space">
  <a href="https://bit.ly/yolov5-neuralmagic">
    <img src="https://github.com/ultralytics/assets/raw/main/partners/logo-neuralmagic.png" width="10%" alt="NeuralMagic logo"></a>
</div>

|                                      Roboflow                                      |                                 ClearML ⭐ NEW                                  |                                     Comet ⭐ NEW                                      |                                  Neural Magic ⭐ NEW                                   |
| :--------------------------------------------------------------------------------: | :----------------------------------------------------------------------------: | :----------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------: |
| 使用 [Roboflow](https://roboflow.com/?ref=ultralytics) 将您的自定义数据集直接标记并导出至 YOLOv8 进行训练 | 使用 [ClearML](https://cutt.ly/yolov5-readme-clearml)（开源！）自动跟踪、可视化，甚至远程训练 YOLOv8 | 免费且永久，[Comet](https://bit.ly/yolov8-readme-comet) 让您保存 YOLOv8 模型、恢复训练，并以交互式方式查看和调试预测 | 使用 [Neural Magic DeepSparse](https://bit.ly/yolov5-neuralmagic) 使 YOLOv8 推理速度提高多达 6 倍 |

## <div align="center">Ultralytics HUB</div>

体验 [Ultralytics HUB](https://bit.ly/ultralytics_hub) ⭐ 带来的无缝 AI，这是一个一体化解决方案，用于数据可视化、YOLOv5 和即将推出的 YOLOv8 🚀 模型训练和部署，无需任何编码。通过我们先进的平台和用户友好的 [Ultralytics 应用程序](https://ultralytics.com/app_install)，轻松将图像转化为可操作的见解，并实现您的 AI 愿景。现在就开始您的**免费**之旅！

<a href="https://bit.ly/ultralytics_hub" target="_blank">
<img width="100%" src="https://github.com/ultralytics/assets/raw/main/im/ultralytics-hub.png" alt="Ultralytics HUB preview image"></a>

## <div align="center">贡献</div>

我们喜欢您的参与！没有社区的帮助，YOLOv5 和 YOLOv8 将无法实现。请参阅我们的[贡献指南](https://docs.ultralytics.com/help/contributing)以开始使用，并填写我们的[调查问卷](https://ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey)向我们提供您的使用体验反馈。感谢所有贡献者的支持！🙏

<!-- SVG image from https://opencollective.com/ultralytics/contributors.svg?width=990 -->

<a href="https://github.com/ultralytics/yolov5/graphs/contributors">
<img width="100%" src="https://github.com/ultralytics/assets/raw/main/im/image-contributors.png" alt="Ultralytics open-source contributors"></a>

## <div align="center">许可证</div>

Ultralytics 提供两种许可证选项以适应各种使用场景：

- **AGPL-3.0 许可证**：这个[OSI 批准](https://opensource.org/licenses/)的开源许可证非常适合学生和爱好者，可以推动开放的协作和知识分享。请查看[LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 文件以了解更多细节。
- **企业许可证**：专为商业用途设计，该许可证允许将 Ultralytics 的软件和 AI 模型无缝集成到商业产品和服务中，从而绕过 AGPL-3.0 的开源要求。如果您的场景涉及将我们的解决方案嵌入到商业产品中，请通过 [Ultralytics Licensing](https://ultralytics.com/license)与我们联系。

## <div align="center">联系方式</div>

对于 Ultralytics 的错误报告和功能请求，请访问 [GitHub Issues](https://github.com/ultralytics/ultralytics/issues)，并加入我们的 [Discord](https://ultralytics.com/discord) 社区进行问题和讨论！

<br>
<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://youtube.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.instagram.com/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-instagram.png" width="3%" alt="Ultralytics Instagram"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/discord"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>
=======
<div align="center">
  <p>
    <a href="https://www.ultralytics.com/blog/ultralytics-yolo11-has-arrived-redefine-whats-possible-in-ai" target="_blank">
      <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png" alt="Ultralytics YOLO banner"></a>
  </p>

[中文](https://docs.ultralytics.com/zh/) | [한국어](https://docs.ultralytics.com/ko/) | [日本語](https://docs.ultralytics.com/ja/) | [Русский](https://docs.ultralytics.com/ru/) | [Deutsch](https://docs.ultralytics.com/de/) | [Français](https://docs.ultralytics.com/fr/) | [Español](https://docs.ultralytics.com/es) | [Português](https://docs.ultralytics.com/pt/) | [Türkçe](https://docs.ultralytics.com/tr/) | [Tiếng Việt](https://docs.ultralytics.com/vi/) | [العربية](https://docs.ultralytics.com/ar/) <br>

<div>
    <a href="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yml"><img src="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yml/badge.svg" alt="Ultralytics CI"></a>
    <a href="https://pepy.tech/projects/ultralytics"><img src="https://static.pepy.tech/badge/ultralytics" alt="Ultralytics Downloads"></a>
    <a href="https://zenodo.org/badge/latestdoi/264818686"><img src="https://zenodo.org/badge/264818686.svg" alt="Ultralytics YOLO Citation"></a>
    <a href="https://discord.com/invite/ultralytics"><img alt="Ultralytics Discord" src="https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue"></a>
    <a href="https://community.ultralytics.com/"><img alt="Ultralytics Forums" src="https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue"></a>
    <a href="https://www.reddit.com/r/ultralytics/"><img alt="Ultralytics Reddit" src="https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue"></a>
    <br>
    <a href="https://console.paperspace.com/github/ultralytics/ultralytics"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run Ultralytics on Gradient"></a>
    <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Ultralytics In Colab"></a>
    <a href="https://www.kaggle.com/models/ultralytics/yolo11"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open Ultralytics In Kaggle"></a>
    <a href="https://mybinder.org/v2/gh/ultralytics/ultralytics/HEAD?labpath=examples%2Ftutorial.ipynb"><img src="https://mybinder.org/badge_logo.svg" alt="Open Ultralytics In Binder"></a>
</div>
</div>
<br>

[Ultralytics](https://www.ultralytics.com/) 基于多年在计算机视觉和人工智能领域的基础研究，创造了尖端的、最先进的（SOTA）[YOLO 模型](https://www.ultralytics.com/yolo)。我们的模型不断更新以提高性能和灵活性，具有**速度快**、**精度高**和**易于使用**的特点。它们在[目标检测](https://docs.ultralytics.com/tasks/detect/)、[跟踪](https://docs.ultralytics.com/modes/track/)、[实例分割](https://docs.ultralytics.com/tasks/segment/)、[图像分类](https://docs.ultralytics.com/tasks/classify/)和[姿态估计](https://docs.ultralytics.com/tasks/pose/)任务中表现出色。

在 [Ultralytics 文档](https://docs.ultralytics.com/)中查找详细文档。通过 [GitHub Issues](https://github.com/ultralytics/ultralytics/issues/new/choose) 获取支持。加入 [Discord](https://discord.com/invite/ultralytics)、[Reddit](https://www.reddit.com/r/ultralytics/) 和 [Ultralytics 社区论坛](https://community.ultralytics.com/)参与讨论！

如需商业用途，请在 [Ultralytics 授权许可](https://www.ultralytics.com/license)申请企业许可证。

<a href="https://docs.ultralytics.com/models/yolo11/" target="_blank">
  <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/refs/heads/main/yolo/performance-comparison.png" alt="YOLO11 performance plots">
</a>

<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="2%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="2%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="2%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="2%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="2%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="2%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="2%" alt="Ultralytics Discord"></a>
</div>

## 📄 文档

请参阅下文了解快速安装和使用示例。有关训练、验证、预测和部署的全面指南，请参阅我们的完整 [Ultralytics 文档](https://docs.ultralytics.com/)。

<details open>
<summary>安装</summary>

在 [**Python>=3.8**](https://www.python.org/) 环境中安装 `ultralytics` 包，包括所有[依赖项](https://github.com/ultralytics/ultralytics/blob/main/pyproject.toml)，并确保 [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/)。

[![PyPI - Version](https://img.shields.io/pypi/v/ultralytics?logo=pypi&logoColor=white)](https://pypi.org/project/ultralytics/) [![Ultralytics Downloads](https://static.pepy.tech/badge/ultralytics)](https://www.pepy.tech/projects/ultralytics) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ultralytics?logo=python&logoColor=gold)](https://pypi.org/project/ultralytics/)

```bash
pip install ultralytics
```

有关其他安装方法，包括 [Conda](https://anaconda.org/conda-forge/ultralytics)、[Docker](https://hub.docker.com/r/ultralytics/ultralytics) 以及通过 Git 从源代码构建，请查阅[快速入门指南](https://docs.ultralytics.com/quickstart/)。

[![Conda Version](https://img.shields.io/conda/vn/conda-forge/ultralytics?logo=condaforge)](https://anaconda.org/conda-forge/ultralytics) [![Docker Image Version](https://img.shields.io/docker/v/ultralytics/ultralytics?sort=semver&logo=docker)](https://hub.docker.com/r/ultralytics/ultralytics) [![Ultralytics Docker Pulls](https://img.shields.io/docker/pulls/ultralytics/ultralytics?logo=docker)](https://hub.docker.com/r/ultralytics/ultralytics)

</details>

<details open>
<summary>使用方法</summary>

### CLI

您可以直接通过命令行界面（CLI）使用 `yolo` 命令来运行 Ultralytics YOLO：

```bash
# 使用预训练的 YOLO 模型（例如 YOLO11n）对图像进行预测
yolo predict model=yolo11n.pt source='https://ultralytics.com/images/bus.jpg'
```

`yolo` 命令支持各种任务和模式，并接受额外的参数，如 `imgsz=640`。浏览 YOLO [CLI 文档](https://docs.ultralytics.com/usage/cli/)获取更多示例。

### Python

Ultralytics YOLO 也可以直接集成到您的 Python 项目中。它接受与 CLI 相同的[配置参数](https://docs.ultralytics.com/usage/cfg/)：

```python
from ultralytics import YOLO

# 加载一个预训练的 YOLO11n 模型
model = YOLO("yolo11n.pt")

# 在 COCO8 数据集上训练模型 100 个周期
train_results = model.train(
    data="coco8.yaml",  # 数据集配置文件路径
    epochs=100,  # 训练周期数
    imgsz=640,  # 训练图像尺寸
    device="cpu",  # 运行设备（例如 'cpu', 0, [0,1,2,3]）
)

# 评估模型在验证集上的性能
metrics = model.val()

# 对图像执行目标检测
results = model("path/to/image.jpg")  # 对图像进行预测
results[0].show()  # 显示结果

# 将模型导出为 ONNX 格式以进行部署
path = model.export(format="onnx")  # 返回导出模型的路径
```

在 YOLO [Python 文档](https://docs.ultralytics.com/usage/python/)中发现更多示例。

</details>

## ✨ 模型

Ultralytics 支持广泛的 YOLO 模型，从早期的版本如 [YOLOv3](https://docs.ultralytics.com/models/yolov3/) 到最新的 [YOLO11](https://docs.ultralytics.com/models/yolo11/)。下表展示了在 [COCO](https://docs.ultralytics.com/datasets/detect/coco/) 数据集上预训练的 YOLO11 模型，用于[检测](https://docs.ultralytics.com/tasks/detect/)、[分割](https://docs.ultralytics.com/tasks/segment/)和[姿态估计](https://docs.ultralytics.com/tasks/pose/)任务。此外，还提供了在 [ImageNet](https://docs.ultralytics.com/datasets/classify/imagenet/) 数据集上预训练的[分类](https://docs.ultralytics.com/tasks/classify/)模型。[跟踪](https://docs.ultralytics.com/modes/track/)模式与所有检测、分割和姿态模型兼容。所有[模型](https://docs.ultralytics.com/models/)在首次使用时都会自动从最新的 Ultralytics [发布版本](https://github.com/ultralytics/assets/releases)下载。

<a href="https://docs.ultralytics.com/tasks/" target="_blank">
    <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-yolov8-tasks-banner.avif" alt="Ultralytics YOLO supported tasks">
</a>
<br>
<br>

<details open><summary>检测 (COCO)</summary>

浏览[检测文档](https://docs.ultralytics.com/tasks/detect/)获取使用示例。这些模型在 [COCO 数据集](https://cocodataset.org/)上训练，包含 80 个对象类别。

| 模型                                                                                 | 尺寸<br><sup>(像素) | mAP<sup>val<br>50-95 | 速度<br><sup>CPU ONNX<br>(毫秒) | 速度<br><sup>T4 TensorRT10<br>(毫秒) | 参数<br><sup>(百万) | FLOPs<br><sup>(十亿) |
| ------------------------------------------------------------------------------------ | ------------------- | -------------------- | ------------------------------- | ------------------------------------ | ------------------- | -------------------- |
| [YOLO11n](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt) | 640                 | 39.5                 | 56.1 ± 0.8                      | 1.5 ± 0.0                            | 2.6                 | 6.5                  |
| [YOLO11s](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt) | 640                 | 47.0                 | 90.0 ± 1.2                      | 2.5 ± 0.0                            | 9.4                 | 21.5                 |
| [YOLO11m](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt) | 640                 | 51.5                 | 183.2 ± 2.0                     | 4.7 ± 0.1                            | 20.1                | 68.0                 |
| [YOLO11l](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt) | 640                 | 53.4                 | 238.6 ± 1.4                     | 6.2 ± 0.1                            | 25.3                | 86.9                 |
| [YOLO11x](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt) | 640                 | 54.7                 | 462.8 ± 6.7                     | 11.3 ± 0.2                           | 56.9                | 194.9                |

- **mAP<sup>val</sup>** 值指的是在 [COCO val2017](https://cocodataset.org/) 数据集上的单模型单尺度性能。详见 [YOLO 性能指标](https://docs.ultralytics.com/guides/yolo-performance-metrics/)。<br>使用 `yolo val detect data=coco.yaml device=0` 复现结果。
- **速度** 指标是在 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 实例上对 COCO val 图像进行平均测量的。CPU 速度使用 [ONNX](https://onnx.ai/) 导出进行测量。GPU 速度使用 [TensorRT](https://developer.nvidia.com/tensorrt) 导出进行测量。<br>使用 `yolo val detect data=coco.yaml batch=1 device=0|cpu` 复现结果。

</details>

<details><summary>分割 (COCO)</summary>

请参阅[分割文档](https://docs.ultralytics.com/tasks/segment/)获取使用示例。这些模型在 [COCO-Seg](https://docs.ultralytics.com/datasets/segment/coco/) 数据集上训练，包含 80 个类别。

| 模型                                                                                         | 尺寸<br><sup>(像素) | mAP<sup>box<br>50-95 | mAP<sup>mask<br>50-95 | 速度<br><sup>CPU ONNX<br>(毫秒) | 速度<br><sup>T4 TensorRT10<br>(毫秒) | 参数<br><sup>(百万) | FLOPs<br><sup>(十亿) |
| -------------------------------------------------------------------------------------------- | ------------------- | -------------------- | --------------------- | ------------------------------- | ------------------------------------ | ------------------- | -------------------- |
| [YOLO11n-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt) | 640                 | 38.9                 | 32.0                  | 65.9 ± 1.1                      | 1.8 ± 0.0                            | 2.9                 | 10.4                 |
| [YOLO11s-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt) | 640                 | 46.6                 | 37.8                  | 117.6 ± 4.9                     | 2.9 ± 0.0                            | 10.1                | 35.5                 |
| [YOLO11m-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-seg.pt) | 640                 | 51.5                 | 41.5                  | 281.6 ± 1.2                     | 6.3 ± 0.1                            | 22.4                | 123.3                |
| [YOLO11l-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-seg.pt) | 640                 | 53.4                 | 42.9                  | 344.2 ± 3.2                     | 7.8 ± 0.2                            | 27.6                | 142.2                |
| [YOLO11x-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-seg.pt) | 640                 | 54.7                 | 43.8                  | 664.5 ± 3.2                     | 15.8 ± 0.7                           | 62.1                | 319.0                |

- **mAP<sup>val</sup>** 值指的是在 [COCO val2017](https://cocodataset.org/) 数据集上的单模型单尺度性能。详见 [YOLO 性能指标](https://docs.ultralytics.com/guides/yolo-performance-metrics/)。<br>使用 `yolo val segment data=coco.yaml device=0` 复现结果。
- **速度** 指标是在 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 实例上对 COCO val 图像进行平均测量的。CPU 速度使用 [ONNX](https://onnx.ai/) 导出进行测量。GPU 速度使用 [TensorRT](https://developer.nvidia.com/tensorrt) 导出进行测量。<br>使用 `yolo val segment data=coco.yaml batch=1 device=0|cpu` 复现结果。

</details>

<details><summary>分类 (ImageNet)</summary>

请查阅[分类文档](https://docs.ultralytics.com/tasks/classify/)获取使用示例。这些模型在 [ImageNet](https://docs.ultralytics.com/datasets/classify/imagenet/) 数据集上训练，涵盖 1000 个类别。

| 模型                                                                                         | 尺寸<br><sup>(像素) | acc<br><sup>top1 | acc<br><sup>top5 | 速度<br><sup>CPU ONNX<br>(毫秒) | 速度<br><sup>T4 TensorRT10<br>(毫秒) | 参数<br><sup>(百万) | FLOPs<br><sup>(十亿) @ 224 |
| -------------------------------------------------------------------------------------------- | ------------------- | ---------------- | ---------------- | ------------------------------- | ------------------------------------ | ------------------- | -------------------------- |
| [YOLO11n-cls](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-cls.pt) | 224                 | 70.0             | 89.4             | 5.0 ± 0.3                       | 1.1 ± 0.0                            | 1.6                 | 0.5                        |
| [YOLO11s-cls](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-cls.pt) | 224                 | 75.4             | 92.7             | 7.9 ± 0.2                       | 1.3 ± 0.0                            | 5.5                 | 1.6                        |
| [YOLO11m-cls](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-cls.pt) | 224                 | 77.3             | 93.9             | 17.2 ± 0.4                      | 2.0 ± 0.0                            | 10.4                | 5.0                        |
| [YOLO11l-cls](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-cls.pt) | 224                 | 78.3             | 94.3             | 23.2 ± 0.3                      | 2.8 ± 0.0                            | 12.9                | 6.2                        |
| [YOLO11x-cls](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-cls.pt) | 224                 | 79.5             | 94.9             | 41.4 ± 0.9                      | 3.8 ± 0.0                            | 28.4                | 13.7                       |

- **acc** 值表示模型在 [ImageNet](https://www.image-net.org/) 数据集验证集上的准确率。<br>使用 `yolo val classify data=path/to/ImageNet device=0` 复现结果。
- **速度** 指标是在 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 实例上对 ImageNet val 图像进行平均测量的。CPU 速度使用 [ONNX](https://onnx.ai/) 导出进行测量。GPU 速度使用 [TensorRT](https://developer.nvidia.com/tensorrt) 导出进行测量。<br>使用 `yolo val classify data=path/to/ImageNet batch=1 device=0|cpu` 复现结果。

</details>

<details><summary>姿态估计 (COCO)</summary>

请参阅[姿态估计文档](https://docs.ultralytics.com/tasks/pose/)获取使用示例。这些模型在 [COCO-Pose](https://docs.ultralytics.com/datasets/pose/coco/) 数据集上训练，专注于 'person' 类别。

| 模型                                                                                           | 尺寸<br><sup>(像素) | mAP<sup>pose<br>50-95 | mAP<sup>pose<br>50 | 速度<br><sup>CPU ONNX<br>(毫秒) | 速度<br><sup>T4 TensorRT10<br>(毫秒) | 参数<br><sup>(百万) | FLOPs<br><sup>(十亿) |
| ---------------------------------------------------------------------------------------------- | ------------------- | --------------------- | ------------------ | ------------------------------- | ------------------------------------ | ------------------- | -------------------- |
| [YOLO11n-pose](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt) | 640                 | 50.0                  | 81.0               | 52.4 ± 0.5                      | 1.7 ± 0.0                            | 2.9                 | 7.6                  |
| [YOLO11s-pose](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-pose.pt) | 640                 | 58.9                  | 86.3               | 90.5 ± 0.6                      | 2.6 ± 0.0                            | 9.9                 | 23.2                 |
| [YOLO11m-pose](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-pose.pt) | 640                 | 64.9                  | 89.4               | 187.3 ± 0.8                     | 4.9 ± 0.1                            | 20.9                | 71.7                 |
| [YOLO11l-pose](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-pose.pt) | 640                 | 66.1                  | 89.9               | 247.7 ± 1.1                     | 6.4 ± 0.1                            | 26.2                | 90.7                 |
| [YOLO11x-pose](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt) | 640                 | 69.5                  | 91.1               | 488.0 ± 13.9                    | 12.1 ± 0.2                           | 58.8                | 203.3                |

- **mAP<sup>val</sup>** 值指的是在 [COCO Keypoints val2017](https://docs.ultralytics.com/datasets/pose/coco/) 数据集上的单模型单尺度性能。详见 [YOLO 性能指标](https://docs.ultralytics.com/guides/yolo-performance-metrics/)。<br>使用 `yolo val pose data=coco-pose.yaml device=0` 复现结果。
- **速度** 指标是在 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 实例上对 COCO val 图像进行平均测量的。CPU 速度使用 [ONNX](https://onnx.ai/) 导出进行测量。GPU 速度使用 [TensorRT](https://developer.nvidia.com/tensorrt) 导出进行测量。<br>使用 `yolo val pose data=coco-pose.yaml batch=1 device=0|cpu` 复现结果。

</details>

<details><summary>定向边界框 (DOTAv1)</summary>

请查阅 [OBB 文档](https://docs.ultralytics.com/tasks/obb/)获取使用示例。这些模型在 [DOTAv1](https://docs.ultralytics.com/datasets/obb/dota-v2/#dota-v10) 数据集上训练，包含 15 个类别。

| 模型                                                                                         | 尺寸<br><sup>(像素) | mAP<sup>test<br>50 | 速度<br><sup>CPU ONNX<br>(毫秒) | 速度<br><sup>T4 TensorRT10<br>(毫秒) | 参数<br><sup>(百万) | FLOPs<br><sup>(十亿) |
| -------------------------------------------------------------------------------------------- | ------------------- | ------------------ | ------------------------------- | ------------------------------------ | ------------------- | -------------------- |
| [YOLO11n-obb](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-obb.pt) | 1024                | 78.4               | 117.6 ± 0.8                     | 4.4 ± 0.0                            | 2.7                 | 17.2                 |
| [YOLO11s-obb](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-obb.pt) | 1024                | 79.5               | 219.4 ± 4.0                     | 5.1 ± 0.0                            | 9.7                 | 57.5                 |
| [YOLO11m-obb](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-obb.pt) | 1024                | 80.9               | 562.8 ± 2.9                     | 10.1 ± 0.4                           | 20.9                | 183.5                |
| [YOLO11l-obb](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-obb.pt) | 1024                | 81.0               | 712.5 ± 5.0                     | 13.5 ± 0.6                           | 26.2                | 232.0                |
| [YOLO11x-obb](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-obb.pt) | 1024                | 81.3               | 1408.6 ± 7.7                    | 28.6 ± 1.0                           | 58.8                | 520.2                |

- **mAP<sup>test</sup>** 值指的是在 [DOTAv1 测试集](https://captain-whu.github.io/DOTA/dataset.html)上的单模型多尺度性能。<br>通过 `yolo val obb data=DOTAv1.yaml device=0 split=test` 复现结果，并将合并后的结果提交到 [DOTA 评估服务器](https://captain-whu.github.io/DOTA/evaluation.html)。
- **速度** 指标是在 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 实例上对 [DOTAv1 val 图像](https://docs.ultralytics.com/datasets/obb/dota-v2/#dota-v10)进行平均测量的。CPU 速度使用 [ONNX](https://onnx.ai/) 导出进行测量。GPU 速度使用 [TensorRT](https://developer.nvidia.com/tensorrt) 导出进行测量。<br>通过 `yolo val obb data=DOTAv1.yaml batch=1 device=0|cpu` 复现结果。

</details>

## 🧩 集成

我们与领先 AI 平台的关键集成扩展了 Ultralytics 产品的功能，增强了数据集标注、训练、可视化和模型管理等任务。了解 Ultralytics 如何与 [Weights & Biases](https://docs.ultralytics.com/integrations/weights-biases/)、[Comet ML](https://docs.ultralytics.com/integrations/comet/)、[Roboflow](https://docs.ultralytics.com/integrations/roboflow/) 和 [Intel OpenVINO](https://docs.ultralytics.com/integrations/openvino/) 等合作伙伴协作，优化您的 AI 工作流程。在 [Ultralytics 集成](https://docs.ultralytics.com/integrations/)了解更多信息。

<a href="https://docs.ultralytics.com/integrations/" target="_blank">
    <img width="100%" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics active learning integrations">
</a>
<br>
<br>

<div align="center">
  <a href="https://www.ultralytics.com/hub">
    <img src="https://github.com/ultralytics/assets/raw/main/partners/logo-ultralytics-hub.png" width="10%" alt="Ultralytics HUB logo"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="15%" height="0" alt="space">
  <a href="https://docs.ultralytics.com/integrations/weights-biases/">
    <img src="https://github.com/ultralytics/assets/raw/main/partners/logo-wb.png" width="10%" alt="Weights & Biases logo"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="15%" height="0" alt="space">
  <a href="https://docs.ultralytics.com/integrations/comet/">
    <img src="https://github.com/ultralytics/assets/raw/main/partners/logo-comet.png" width="10%" alt="Comet ML logo"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="15%" height="0" alt="space">
  <a href="https://docs.ultralytics.com/integrations/neural-magic/">
    <img src="https://github.com/ultralytics/assets/raw/main/partners/logo-neuralmagic.png" width="10%" alt="Neural Magic logo"></a>
</div>

|                                              Ultralytics HUB 🌟                                               |                                              Weights & Biases                                               |                                                               Comet                                                                |                                                       Neural Magic                                                       |
| :-----------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------: |
| 简化 YOLO 工作流程：使用 [Ultralytics HUB](https://hub.ultralytics.com/) 轻松进行标注、训练和部署。立即试用！ | 使用 [Weights & Biases](https://docs.ultralytics.com/integrations/weights-biases/) 跟踪实验、超参数和结果。 | 永久免费的 [Comet ML](https://docs.ultralytics.com/integrations/comet/) 让您能够保存 YOLO 模型、恢复训练并交互式地可视化预测结果。 | 使用 [Neural Magic DeepSparse](https://docs.ultralytics.com/integrations/neural-magic/)，将 YOLO 推理速度提高多达 6 倍。 |

## 🌟 Ultralytics HUB

通过 [Ultralytics HUB](https://hub.ultralytics.com/) 体验无缝 AI，这是一个集数据可视化、训练 YOLO 模型和部署于一体的平台——无需编码。使用我们尖端的平台和用户友好的 [Ultralytics App](https://www.ultralytics.com/app-install)，轻松将图像转化为可操作的见解，并将您的 AI 愿景变为现实。立即**免费**开始您的旅程！

<a href="https://www.ultralytics.com/hub" target="_blank">
<img width="100%" src="https://github.com/ultralytics/assets/raw/main/im/ultralytics-hub.png" alt="Ultralytics HUB preview image"></a>

## 🤝 贡献

我们依靠社区协作蓬勃发展！没有像您这样的开发者的贡献，Ultralytics YOLO 就不会成为如今最先进的框架。请参阅我们的[贡献指南](https://docs.ultralytics.com/help/contributing/)开始贡献。我们也欢迎您的反馈——通过完成我们的[调查问卷](https://www.ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey)分享您的体验。非常**感谢** 🙏 每一位贡献者！

<!-- SVG image from https://opencollective.com/ultralytics/contributors.svg?width=1280 -->

[![Ultralytics open-source contributors](https://raw.githubusercontent.com/ultralytics/assets/main/im/image-contributors.png)](https://github.com/ultralytics/ultralytics/graphs/contributors)

我们期待您的贡献，帮助 Ultralytics 生态系统变得更好！

## 📜 许可证

Ultralytics 提供两种许可选项以满足不同需求：

- **AGPL-3.0 许可证**：这种经 [OSI 批准](https://opensource.org/license)的开源许可证非常适合学生、研究人员和爱好者。它鼓励开放协作和知识共享。有关完整详细信息，请参阅 [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 文件。
- **Ultralytics 企业许可证**：专为商业用途设计，此许可证允许将 Ultralytics 软件和 AI 模型无缝集成到商业产品和服务中，绕过 AGPL-3.0 的开源要求。如果您的使用场景涉及商业部署，请通过 [Ultralytics 授权许可](https://www.ultralytics.com/license)与我们联系。

## 📞 联系方式

有关 Ultralytics 软件的错误报告和功能请求，请访问 [GitHub Issues](https://github.com/ultralytics/ultralytics/issues)。如有疑问、讨论和社区支持，请加入我们在 [Discord](https://discord.com/invite/ultralytics)、[Reddit](https://www.reddit.com/r/ultralytics/?rdt=44154) 和 [Ultralytics 社区论坛](https://community.ultralytics.com/)上的活跃社区。我们随时为您提供有关 Ultralytics 的所有帮助！

<br>
<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>
>>>>>>> 0e451281e978a7c5f581cc5d9bb3d0f62e0bc632
