<<<<<<< HEAD
---
comments: true
description: Learn to freeze YOLOv5 layers for efficient transfer learning. Optimize your model retraining with less resources and faster training times.
keywords: YOLOv5, freeze layers, transfer learning, model retraining, Ultralytics
---

📚 This guide explains how to **freeze** YOLOv5 🚀 layers when **transfer learning**. Transfer learning is a useful way to quickly retrain a model on new data without having to retrain the entire network. Instead, part of the initial weights are frozen in place, and the rest of the weights are used to compute loss and are updated by the optimizer. This requires less resources than normal training and allows for faster training times, though it may also result in reductions to final trained accuracy.

## Before You Start

Clone repo and install [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a [**Python>=3.8.0**](https://www.python.org/) environment, including [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/). [Models](https://github.com/ultralytics/yolov5/tree/master/models) and [datasets](https://github.com/ultralytics/yolov5/tree/master/data) download automatically from the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases).

```bash
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

## Freeze Backbone

All layers that match the train.py `freeze` list in train.py will be frozen by setting their gradients to zero before training starts.

```python
# Freeze
freeze = [f'model.{x}.' for x in range(freeze)]  # layers to freeze
for k, v in model.named_parameters():
    v.requires_grad = True  # train all layers
    if any(x in k for x in freeze):
        print(f'freezing {k}')
        v.requires_grad = False
```

To see a list of module names:

```python
for k, v in model.named_parameters():
    print(k)

"""Output:
model.0.conv.conv.weight
model.0.conv.bn.weight
model.0.conv.bn.bias
model.1.conv.weight
model.1.bn.weight
model.1.bn.bias
model.2.cv1.conv.weight
model.2.cv1.bn.weight
...
model.23.m.0.cv2.bn.weight
model.23.m.0.cv2.bn.bias
model.24.m.0.weight
model.24.m.0.bias
model.24.m.1.weight
model.24.m.1.bias
model.24.m.2.weight
model.24.m.2.bias
"""
```

Looking at the model architecture we can see that the model backbone is layers 0-9:

```yaml
# YOLOv5 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Focus, [64, 3]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, BottleneckCSP, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 9, BottleneckCSP, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, BottleneckCSP, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 1, SPP, [1024, [5, 9, 13]]],
   [-1, 3, BottleneckCSP, [1024, False]],  # 9
  ]

# YOLOv5 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, BottleneckCSP, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, BottleneckCSP, [256, False]],  # 17 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, BottleneckCSP, [512, False]],  # 20 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, BottleneckCSP, [1024, False]],  # 23 (P5/32-large)

   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```

so we can define the freeze list to contain all modules with 'model.0.' - 'model.9.' in their names:

```bash
python train.py --freeze 10
```

## Freeze All Layers

To freeze the full model except for the final output convolution layers in Detect(), we set freeze list to contain all modules with 'model.0.' - 'model.23.' in their names:

```bash
python train.py --freeze 24
```

## Results

We train YOLOv5m on VOC on both of the above scenarios, along with a default model (no freezing), starting from the official COCO pretrained `--weights yolov5m.pt`:

```bash
train.py --batch 48 --weights yolov5m.pt --data voc.yaml --epochs 50 --cache --img 512 --hyp hyp.finetune.yaml
```

### Accuracy Comparison

The results show that freezing speeds up training, but reduces final accuracy slightly.

![Freezing training mAP50 results](https://user-images.githubusercontent.com/26833433/98394454-11579f80-205b-11eb-8e57-d8318e1cc2f8.png)

![Freezing training mAP50-95 results](https://user-images.githubusercontent.com/26833433/98394459-13216300-205b-11eb-871b-49e20691a423.png)

<img width="922" alt="Table results" src="https://user-images.githubusercontent.com/26833433/98394485-22081580-205b-11eb-9e37-1f9869fe91d8.png">

### GPU Utilization Comparison

Interestingly, the more modules are frozen the less GPU memory is required to train, and the lower GPU utilization. This indicates that larger models, or models trained at larger --image-size may benefit from freezing in order to train faster.

![Training GPU memory allocated percent](https://user-images.githubusercontent.com/26833433/98394920-c2f6d080-205b-11eb-9611-fd68522b4e0e.png)

![Training GPU memory utilization percent](https://user-images.githubusercontent.com/26833433/98394918-bf634980-205b-11eb-948d-311036ef9325.png)

## Supported Environments

Ultralytics provides a range of ready-to-use environments, each pre-installed with essential dependencies such as [CUDA](https://developer.nvidia.com/cuda), [CUDNN](https://developer.nvidia.com/cudnn), [Python](https://www.python.org/), and [PyTorch](https://pytorch.org/), to kickstart your projects.

- **Free GPU Notebooks**: <a href="https://bit.ly/yolov5-paperspace-notebook"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient"></a> <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://www.kaggle.com/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
- **Google Cloud**: [GCP Quickstart Guide](../environments/google_cloud_quickstart_tutorial.md)
- **Amazon**: [AWS Quickstart Guide](../environments/aws_quickstart_tutorial.md)
- **Azure**: [AzureML Quickstart Guide](../environments/azureml_quickstart_tutorial.md)
- **Docker**: [Docker Quickstart Guide](../environments/docker_image_quickstart_tutorial.md) <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>

## Project Status

<a href="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml"><img src="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml/badge.svg" alt="YOLOv5 CI"></a>

This badge indicates that all [YOLOv5 GitHub Actions](https://github.com/ultralytics/yolov5/actions) Continuous Integration (CI) tests are successfully passing. These CI tests rigorously check the functionality and performance of YOLOv5 across various key aspects: [training](https://github.com/ultralytics/yolov5/blob/master/train.py), [validation](https://github.com/ultralytics/yolov5/blob/master/val.py), [inference](https://github.com/ultralytics/yolov5/blob/master/detect.py), [export](https://github.com/ultralytics/yolov5/blob/master/export.py), and [benchmarks](https://github.com/ultralytics/yolov5/blob/master/benchmarks.py). They ensure consistent and reliable operation on macOS, Windows, and Ubuntu, with tests conducted every 24 hours and upon each new commit.
=======
---
comments: true
description: Learn to freeze YOLOv5 layers for efficient transfer learning, reducing resources and speeding up training while maintaining accuracy.
keywords: YOLOv5, transfer learning, freeze layers, machine learning, deep learning, model training, PyTorch, Ultralytics
---

# Transfer Learning with Frozen Layers in YOLOv5

📚 This guide explains how to **freeze** [YOLOv5](https://github.com/ultralytics/yolov5) 🚀 layers when implementing [transfer learning](https://www.ultralytics.com/glossary/transfer-learning). Transfer learning is a powerful [machine learning (ML)](https://www.ultralytics.com/glossary/machine-learning-ml) technique that allows you to quickly retrain a model on new data without retraining the entire network from scratch. By freezing the weights of initial layers and only updating the parameters of later layers, you can significantly reduce computational resource requirements and training time. However, this approach might slightly impact the final model [accuracy](https://www.ultralytics.com/glossary/accuracy).

## Before You Start

First, clone the YOLOv5 repository and install the necessary dependencies listed in [`requirements.txt`](https://github.com/ultralytics/yolov5/blob/master/requirements.txt). Ensure you have a [**Python>=3.8.0**](https://www.python.org/) environment with [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/) installed. Pre-trained [models](https://github.com/ultralytics/yolov5/tree/master/models) and required [datasets](https://github.com/ultralytics/yolov5/tree/master/data) will be downloaded automatically from the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases).

```bash
git clone https://github.com/ultralytics/yolov5 # clone repository
cd yolov5
pip install -r requirements.txt # install dependencies
```

## How Layer Freezing Works

When you freeze layers in a [neural network](https://www.ultralytics.com/glossary/neural-network-nn), you prevent their parameters (weights and biases) from being updated during the training process. In PyTorch, this is achieved by setting the `requires_grad` attribute of the layer's tensors to `False`. Consequently, gradients are not computed for these layers during [backpropagation](https://www.ultralytics.com/glossary/backpropagation), saving computation and memory.

Here's how YOLOv5 implements layer freezing in its [training script](https://github.com/ultralytics/yolov5/blob/master/train.py):

```python
# Freeze specified layers
freeze = [f"model.{x}." for x in range(freeze)]  # Define layers to freeze based on module index
for k, v in model.named_parameters():
    v.requires_grad = True  # Ensure all parameters are initially trainable
    if any(x in k for x in freeze):
        print(f"Freezing layer: {k}")
        v.requires_grad = False  # Disable gradient calculation for frozen layers
```

## Exploring Model Architecture

Understanding the structure of the YOLOv5 model is crucial for deciding which layers to freeze. You can inspect the names of all modules and their parameters using the following Python snippet:

```python
# Assuming 'model' is your loaded YOLOv5 model instance
for name, param in model.named_parameters():
    print(name)

"""
Example Output:
model.0.conv.conv.weight
model.0.conv.bn.weight
model.0.conv.bn.bias
model.1.conv.weight
model.1.bn.weight
model.1.bn.bias
model.2.cv1.conv.weight
model.2.cv1.bn.weight
...
"""
```

The YOLOv5 architecture typically consists of a [backbone](https://www.ultralytics.com/glossary/backbone) (layers 0-9 in standard configurations like YOLOv5s/m/l/x) responsible for [feature extraction](https://www.ultralytics.com/glossary/feature-extraction), and a head (the remaining layers) which performs [object detection](https://www.ultralytics.com/glossary/object-detection).

```yaml
# Example YOLOv5 v6.0 backbone structure
backbone:
    # [from, number, module, args]
    - [-1, 1, Conv, [64, 6, 2, 2]]  # Layer 0: Initial convolution (P1/2 stride)
    - [-1, 1, Conv, [128, 3, 2]] # Layer 1: Downsampling convolution (P2/4 stride)
    - [-1, 3, C3, [128]]          # Layer 2: C3 module
    - [-1, 1, Conv, [256, 3, 2]] # Layer 3: Downsampling convolution (P3/8 stride)
    - [-1, 6, C3, [256]]          # Layer 4: C3 module
    - [-1, 1, Conv, [512, 3, 2]] # Layer 5: Downsampling convolution (P4/16 stride)
    - [-1, 9, C3, [512]]          # Layer 6: C3 module
    - [-1, 1, Conv, [1024, 3, 2]]# Layer 7: Downsampling convolution (P5/32 stride)
    - [-1, 3, C3, [1024]]         # Layer 8: C3 module
    - [-1, 1, SPPF, [1024, 5]]    # Layer 9: Spatial Pyramid Pooling Fast

# Example YOLOv5 v6.0 head structure
head:
    - [-1, 1, Conv, [512, 1, 1]] # Layer 10
    - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # Layer 11
    - [[-1, 6], 1, Concat, [1]] # Layer 12: Concatenate with backbone P4 (from layer 6)
    - [-1, 3, C3, [512, False]] # Layer 13: C3 module
    # ... subsequent head layers for feature fusion and detection
```

## Freezing Options

You can control which layers are frozen using the `--freeze` argument in the training command. This argument specifies the index of the first _unfrozen_ module; all modules before this index will have their weights frozen.

### Freeze Backbone Only

To freeze the entire backbone (layers 0 through 9), which is common when adapting the model to new object classes while retaining general feature extraction capabilities learned from a large dataset like [COCO](https://docs.ultralytics.com/datasets/detect/coco/):

```bash
python train.py --weights yolov5m.pt --data your_dataset.yaml --freeze 10
```

This strategy is effective when your target dataset shares similar low-level visual features (edges, textures) with the original training data (e.g., COCO) but contains different object categories.

### Freeze All Except Final Detection Layers

To freeze almost the entire network, leaving only the final output convolution layers (part of the `Detect` module, typically the last module, e.g., module 24 in YOLOv5s) trainable:

```bash
python train.py --weights yolov5m.pt --data your_dataset.yaml --freeze 24
```

This approach is useful when you primarily need to adjust the model for a different number of output classes while keeping the vast majority of learned features intact. It requires the least computational resources for [fine-tuning](https://www.ultralytics.com/glossary/fine-tuning).

## Performance Comparison

To illustrate the effects of freezing layers, we trained YOLOv5m on the [Pascal VOC dataset](https://docs.ultralytics.com/datasets/detect/voc/) for 50 [epochs](https://www.ultralytics.com/glossary/epoch), starting from the official COCO pre-trained [weights](https://www.ultralytics.com/glossary/model-weights) (`yolov5m.pt`). We compared three scenarios: training all layers (`--freeze 0`), freezing the backbone (`--freeze 10`), and freezing all but the final detection layers (`--freeze 24`).

```bash
# Example command for training with backbone frozen
python train.py --batch 48 --weights yolov5m.pt --data voc.yaml --epochs 50 --cache --img 512 --hyp hyp.finetune.yaml --freeze 10
```

### Accuracy Results

The results show that freezing layers can accelerate training significantly but may lead to a slight reduction in final [mAP (mean Average Precision)](https://www.ultralytics.com/glossary/mean-average-precision-map). Training all layers generally yields the best accuracy, while freezing more layers offers faster training at the cost of potentially lower performance.

![Training mAP50 results comparing different freezing strategies](https://github.com/ultralytics/docs/releases/download/0/freezing-training-map50-results.avif)
_mAP50 comparison during training_

![Training mAP50-95 results comparing different freezing strategies](https://github.com/ultralytics/docs/releases/download/0/freezing-training-map50-95-results.avif)
_mAP50-95 comparison during training_

<img width="922" alt="Table summarizing performance results" src="https://github.com/ultralytics/docs/releases/download/0/table-results.avif">
*Summary table of performance metrics*

### Resource Utilization

Freezing more layers substantially reduces [GPU](https://www.ultralytics.com/glossary/gpu-graphics-processing-unit) memory requirements and overall utilization. This makes transfer learning with frozen layers an attractive option when working with limited hardware resources, allowing for training larger models or using larger image sizes than might otherwise be possible.

![GPU memory allocated percentage during training](https://github.com/ultralytics/docs/releases/download/0/training-gpu-memory-allocated-percent.avif)
_GPU Memory Allocated (%)_

![GPU memory utilization percentage during training](https://github.com/ultralytics/docs/releases/download/0/training-gpu-memory-utilization-percent.avif)
_GPU Utilization (%)_

## When to Use Layer Freezing

Layer freezing during transfer learning is particularly advantageous in several situations:

1.  **Limited Computational Resources**: If you have constraints on GPU memory or processing power.
2.  **Small Datasets**: When your target dataset is significantly smaller than the original pre-training dataset, freezing helps prevent [overfitting](https://www.ultralytics.com/glossary/overfitting).
3.  **Rapid Prototyping**: When you need to quickly adapt an existing model to a new task or domain for initial evaluation.
4.  **Similar Feature Domains**: If the low-level features in your new dataset are very similar to those in the dataset the model was pre-trained on.

Explore more about the nuances of transfer learning in our [glossary entry](https://www.ultralytics.com/glossary/transfer-learning) and consider techniques like [hyperparameter tuning](https://docs.ultralytics.com/guides/hyperparameter-tuning/) for optimizing performance.

## Supported Environments

Ultralytics offers various ready-to-use environments with essential dependencies like [CUDA](https://developer.nvidia.com/cuda-zone), [CuDNN](https://developer.nvidia.com/cudnn), [Python](https://www.python.org/), and [PyTorch](https://pytorch.org/) pre-installed.

- **Free GPU Notebooks**: <a href="https://bit.ly/yolov5-paperspace-notebook"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient"></a> <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://github.com/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
- **Google Cloud**: [GCP Quickstart Guide](../environments/google_cloud_quickstart_tutorial.md)
- **Amazon**: [AWS Quickstart Guide](../environments/aws_quickstart_tutorial.md)
- **Azure**: [AzureML Quickstart Guide](../environments/azureml_quickstart_tutorial.md)
- **Docker**: [Docker Quickstart Guide](../environments/docker_image_quickstart_tutorial.md) <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>

## Project Status

<a href="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml"><img src="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml/badge.svg" alt="YOLOv5 Continuous Integration Status"></a>

This badge confirms that all [YOLOv5 GitHub Actions](https://github.com/ultralytics/yolov5/actions) Continuous Integration (CI) tests are passing successfully. These CI tests rigorously evaluate the functionality and performance of YOLOv5 across key operations: [training](https://github.com/ultralytics/yolov5/blob/master/train.py), [validation](https://github.com/ultralytics/yolov5/blob/master/val.py), [inference](https://github.com/ultralytics/yolov5/blob/master/detect.py), [export](https://github.com/ultralytics/yolov5/blob/master/export.py), and [benchmarks](https://github.com/ultralytics/yolov5/blob/master/benchmarks.py). They ensure consistent and reliable operation on macOS, Windows, and Ubuntu, running automatically every 24 hours and on each new code commit.
>>>>>>> 0e451281e978a7c5f581cc5d9bb3d0f62e0bc632
