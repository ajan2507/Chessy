<<<<<<< HEAD
---
comments: true
description: Explore the cutting-edge Segment Anything Model (SAM) from Ultralytics that allows real-time image segmentation. Learn about its promptable segmentation, zero-shot performance, and how to use it.
keywords: Ultralytics, image segmentation, Segment Anything Model, SAM, SA-1B dataset, real-time performance, zero-shot transfer, object detection, image analysis, machine learning
---

# Segment Anything Model (SAM)

Welcome to the frontier of image segmentation with the Segment Anything Model, or SAM. This revolutionary model has changed the game by introducing promptable image segmentation with real-time performance, setting new standards in the field.

## Introduction to SAM: The Segment Anything Model

The Segment Anything Model, or SAM, is a cutting-edge image segmentation model that allows for promptable segmentation, providing unparalleled versatility in image analysis tasks. SAM forms the heart of the Segment Anything initiative, a groundbreaking project that introduces a novel model, task, and dataset for image segmentation.

SAM's advanced design allows it to adapt to new image distributions and tasks without prior knowledge, a feature known as zero-shot transfer. Trained on the expansive [SA-1B dataset](https://ai.facebook.com/datasets/segment-anything/), which contains more than 1 billion masks spread over 11 million carefully curated images, SAM has displayed impressive zero-shot performance, surpassing previous fully supervised results in many cases.

![Dataset sample image](https://user-images.githubusercontent.com/26833433/238056229-0e8ffbeb-f81a-477e-a490-aff3d82fd8ce.jpg)
Example images with overlaid masks from our newly introduced dataset, SA-1B. SA-1B contains 11M diverse, high-resolution, licensed, and privacy protecting images and 1.1B high-quality segmentation masks. These masks were annotated fully automatically by SAM, and as verified by human ratings and numerous experiments, are of high quality and diversity. Images are grouped by number of masks per image for visualization (there are ∼100 masks per image on average).

## Key Features of the Segment Anything Model (SAM)

- **Promptable Segmentation Task:** SAM was designed with a promptable segmentation task in mind, allowing it to generate valid segmentation masks from any given prompt, such as spatial or text clues identifying an object.
- **Advanced Architecture:** The Segment Anything Model employs a powerful image encoder, a prompt encoder, and a lightweight mask decoder. This unique architecture enables flexible prompting, real-time mask computation, and ambiguity awareness in segmentation tasks.
- **The SA-1B Dataset:** Introduced by the Segment Anything project, the SA-1B dataset features over 1 billion masks on 11 million images. As the largest segmentation dataset to date, it provides SAM with a diverse and large-scale training data source.
- **Zero-Shot Performance:** SAM displays outstanding zero-shot performance across various segmentation tasks, making it a ready-to-use tool for diverse applications with minimal need for prompt engineering.

For an in-depth look at the Segment Anything Model and the SA-1B dataset, please visit the [Segment Anything website](https://segment-anything.com) and check out the research paper [Segment Anything](https://arxiv.org/abs/2304.02643).

## Available Models, Supported Tasks, and Operating Modes

This table presents the available models with their specific pre-trained weights, the tasks they support, and their compatibility with different operating modes like [Inference](../modes/predict.md), [Validation](../modes/val.md), [Training](../modes/train.md), and [Export](../modes/export.md), indicated by ✅ emojis for supported modes and ❌ emojis for unsupported modes.

| Model Type | Pre-trained Weights | Tasks Supported                              | Inference | Validation | Training | Export |
|------------|---------------------|----------------------------------------------|-----------|------------|----------|--------|
| SAM base   | `sam_b.pt`          | [Instance Segmentation](../tasks/segment.md) | ✅         | ❌          | ❌        | ❌      |
| SAM large  | `sam_l.pt`          | [Instance Segmentation](../tasks/segment.md) | ✅         | ❌          | ❌        | ❌      |

## How to Use SAM: Versatility and Power in Image Segmentation

The Segment Anything Model can be employed for a multitude of downstream tasks that go beyond its training data. This includes edge detection, object proposal generation, instance segmentation, and preliminary text-to-mask prediction. With prompt engineering, SAM can swiftly adapt to new tasks and data distributions in a zero-shot manner, establishing it as a versatile and potent tool for all your image segmentation needs.

### SAM prediction example

!!! Example "Segment with prompts"

    Segment image with given prompts.

    === "Python"

        ```python
        from ultralytics import SAM

        # Load a model
        model = SAM('sam_b.pt')

        # Display model information (optional)
        model.info()

        # Run inference with bboxes prompt
        model('ultralytics/assets/zidane.jpg', bboxes=[439, 437, 524, 709])

        # Run inference with points prompt
        model('ultralytics/assets/zidane.jpg', points=[900, 370], labels=[1])
        ```

!!! Example "Segment everything"

    Segment the whole image.

    === "Python"

        ```python
        from ultralytics import SAM

        # Load a model
        model = SAM('sam_b.pt')

        # Display model information (optional)
        model.info()

        # Run inference
        model('path/to/image.jpg')
        ```

    === "CLI"

        ```bash
        # Run inference with a SAM model
        yolo predict model=sam_b.pt source=path/to/image.jpg
        ```

- The logic here is to segment the whole image if you don't pass any prompts(bboxes/points/masks).

!!! Example "SAMPredictor example"

    This way you can set image once and run prompts inference multiple times without running image encoder multiple times.

    === "Prompt inference"

        ```python
        from ultralytics.models.sam import Predictor as SAMPredictor

        # Create SAMPredictor
        overrides = dict(conf=0.25, task='segment', mode='predict', imgsz=1024, model="mobile_sam.pt")
        predictor = SAMPredictor(overrides=overrides)

        # Set image
        predictor.set_image("ultralytics/assets/zidane.jpg")  # set with image file
        predictor.set_image(cv2.imread("ultralytics/assets/zidane.jpg"))  # set with np.ndarray
        results = predictor(bboxes=[439, 437, 524, 709])
        results = predictor(points=[900, 370], labels=[1])

        # Reset image
        predictor.reset_image()
        ```

    Segment everything with additional args.

    === "Segment everything"

        ```python
        from ultralytics.models.sam import Predictor as SAMPredictor

        # Create SAMPredictor
        overrides = dict(conf=0.25, task='segment', mode='predict', imgsz=1024, model="mobile_sam.pt")
        predictor = SAMPredictor(overrides=overrides)

        # Segment with additional args
        results = predictor(source="ultralytics/assets/zidane.jpg", crop_n_layers=1, points_stride=64)
        ```

- More additional args for `Segment everything` see [`Predictor/generate` Reference](../reference/models/sam/predict.md).

## SAM comparison vs YOLOv8

Here we compare Meta's smallest SAM model, SAM-b, with Ultralytics smallest segmentation model, [YOLOv8n-seg](../tasks/segment.md):

| Model                                          | Size                       | Parameters             | Speed (CPU)                |
|------------------------------------------------|----------------------------|------------------------|----------------------------|
| Meta's SAM-b                                   | 358 MB                     | 94.7 M                 | 51096 ms/im                |
| [MobileSAM](mobile-sam.md)                     | 40.7 MB                    | 10.1 M                 | 46122 ms/im                |
| [FastSAM-s](fast-sam.md) with YOLOv8 backbone  | 23.7 MB                    | 11.8 M                 | 115 ms/im                  |
| Ultralytics [YOLOv8n-seg](../tasks/segment.md) | **6.7 MB** (53.4x smaller) | **3.4 M** (27.9x less) | **59 ms/im** (866x faster) |

This comparison shows the order-of-magnitude differences in the model sizes and speeds between models. Whereas SAM presents unique capabilities for automatic segmenting, it is not a direct competitor to YOLOv8 segment models, which are smaller, faster and more efficient.

Tests run on a 2023 Apple M2 Macbook with 16GB of RAM. To reproduce this test:

!!! Example

    === "Python"
        ```python
        from ultralytics import FastSAM, SAM, YOLO

        # Profile SAM-b
        model = SAM('sam_b.pt')
        model.info()
        model('ultralytics/assets')

        # Profile MobileSAM
        model = SAM('mobile_sam.pt')
        model.info()
        model('ultralytics/assets')

        # Profile FastSAM-s
        model = FastSAM('FastSAM-s.pt')
        model.info()
        model('ultralytics/assets')

        # Profile YOLOv8n-seg
        model = YOLO('yolov8n-seg.pt')
        model.info()
        model('ultralytics/assets')
        ```

## Auto-Annotation: A Quick Path to Segmentation Datasets

Auto-annotation is a key feature of SAM, allowing users to generate a [segmentation dataset](https://docs.ultralytics.com/datasets/segment) using a pre-trained detection model. This feature enables rapid and accurate annotation of a large number of images, bypassing the need for time-consuming manual labeling.

### Generate Your Segmentation Dataset Using a Detection Model

To auto-annotate your dataset with the Ultralytics framework, use the `auto_annotate` function as shown below:

!!! Example

    === "Python"
        ```python
        from ultralytics.data.annotator import auto_annotate

        auto_annotate(data="path/to/images", det_model="yolov8x.pt", sam_model='sam_b.pt')
        ```

| Argument   | Type                | Description                                                                                             | Default      |
|------------|---------------------|---------------------------------------------------------------------------------------------------------|--------------|
| data       | str                 | Path to a folder containing images to be annotated.                                                     |              |
| det_model  | str, optional       | Pre-trained YOLO detection model. Defaults to 'yolov8x.pt'.                                             | 'yolov8x.pt' |
| sam_model  | str, optional       | Pre-trained SAM segmentation model. Defaults to 'sam_b.pt'.                                             | 'sam_b.pt'   |
| device     | str, optional       | Device to run the models on. Defaults to an empty string (CPU or GPU, if available).                    |              |
| output_dir | str, None, optional | Directory to save the annotated results. Defaults to a 'labels' folder in the same directory as 'data'. | None         |

The `auto_annotate` function takes the path to your images, with optional arguments for specifying the pre-trained detection and SAM segmentation models, the device to run the models on, and the output directory for saving the annotated results.

Auto-annotation with pre-trained models can dramatically cut down the time and effort required for creating high-quality segmentation datasets. This feature is especially beneficial for researchers and developers dealing with large image collections, as it allows them to focus on model development and evaluation rather than manual annotation.

## Citations and Acknowledgements

If you find SAM useful in your research or development work, please consider citing our paper:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @misc{kirillov2023segment,
              title={Segment Anything},
              author={Alexander Kirillov and Eric Mintun and Nikhila Ravi and Hanzi Mao and Chloe Rolland and Laura Gustafson and Tete Xiao and Spencer Whitehead and Alexander C. Berg and Wan-Yen Lo and Piotr Dollár and Ross Girshick},
              year={2023},
              eprint={2304.02643},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

We would like to express our gratitude to Meta AI for creating and maintaining this valuable resource for the computer vision community.

*keywords: Segment Anything, Segment Anything Model, SAM, Meta SAM, image segmentation, promptable segmentation, zero-shot performance, SA-1B dataset, advanced architecture, auto-annotation, Ultralytics, pre-trained models, SAM base, SAM large, instance segmentation, computer vision, AI, artificial intelligence, machine learning, data annotation, segmentation masks, detection model, YOLO detection model, bibtex, Meta AI.*
=======
---
comments: true
description: Explore the revolutionary Segment Anything Model (SAM) for promptable image segmentation with zero-shot performance. Discover key features, datasets, and usage tips.
keywords: Segment Anything, SAM, image segmentation, promptable segmentation, zero-shot performance, SA-1B dataset, advanced architecture, auto-annotation, Ultralytics, pre-trained models, instance segmentation, computer vision, AI, machine learning
---

# Segment Anything Model (SAM)

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/inference-with-meta-sam-and-sam2-using-ultralytics-python-package.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="How to use Segment Anything In Colab"></a>

Welcome to the frontier of [image segmentation](https://www.ultralytics.com/glossary/image-segmentation) with the Segment Anything Model, or SAM. This revolutionary model has changed the game by introducing promptable image segmentation with real-time performance, setting new standards in the field.

## Introduction to SAM: The Segment Anything Model

The Segment Anything Model, or SAM, is a cutting-edge image segmentation model that allows for promptable segmentation, providing unparalleled versatility in image analysis tasks. SAM forms the heart of the Segment Anything initiative, a groundbreaking project that introduces a novel model, task, and dataset for image segmentation.

SAM's advanced design allows it to adapt to new image distributions and tasks without prior knowledge, a feature known as zero-shot transfer. Trained on the expansive [SA-1B dataset](https://ai.meta.com/datasets/segment-anything/), which contains more than 1 billion masks spread over 11 million carefully curated images, SAM has displayed impressive zero-shot performance, surpassing previous fully supervised results in many cases.

![Dataset sample image](https://github.com/ultralytics/docs/releases/download/0/sa-1b-dataset-sample.avif) **SA-1B Example images.** Dataset images overlaid masks from the newly introduced SA-1B dataset. SA-1B contains 11M diverse, high-resolution, licensed, and privacy protecting images and 1.1B high-quality segmentation masks. These masks were annotated fully automatically by SAM, and as verified by human ratings and numerous experiments, are of high quality and diversity. Images are grouped by number of masks per image for visualization (there are ∼100 masks per image on average).

## Key Features of the Segment Anything Model (SAM)

- **Promptable Segmentation Task:** SAM was designed with a promptable segmentation task in mind, allowing it to generate valid segmentation masks from any given prompt, such as spatial or text clues identifying an object.
- **Advanced Architecture:** The Segment Anything Model employs a powerful image encoder, a prompt encoder, and a lightweight mask decoder. This unique architecture enables flexible prompting, real-time mask computation, and ambiguity awareness in segmentation tasks.
- **The SA-1B Dataset:** Introduced by the Segment Anything project, the SA-1B dataset features over 1 billion masks on 11 million images. As the largest segmentation dataset to date, it provides SAM with a diverse and large-scale training data source.
- **Zero-Shot Performance:** SAM displays outstanding zero-shot performance across various segmentation tasks, making it a ready-to-use tool for diverse applications with minimal need for [prompt engineering](https://www.ultralytics.com/glossary/prompt-engineering).

For an in-depth look at the Segment Anything Model and the SA-1B dataset, please visit the [Segment Anything website](https://segment-anything.com/) and check out the research paper [Segment Anything](https://arxiv.org/abs/2304.02643).

## Available Models, Supported Tasks, and Operating Modes

This table presents the available models with their specific pre-trained weights, the tasks they support, and their compatibility with different operating modes like [Inference](../modes/predict.md), [Validation](../modes/val.md), [Training](../modes/train.md), and [Export](../modes/export.md), indicated by ✅ emojis for supported modes and ❌ emojis for unsupported modes.

| Model Type | Pre-trained Weights                                                                 | Tasks Supported                              | Inference | Validation | Training | Export |
| ---------- | ----------------------------------------------------------------------------------- | -------------------------------------------- | --------- | ---------- | -------- | ------ |
| SAM base   | [sam_b.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/sam_b.pt) | [Instance Segmentation](../tasks/segment.md) | ✅        | ❌         | ❌       | ❌     |
| SAM large  | [sam_l.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/sam_l.pt) | [Instance Segmentation](../tasks/segment.md) | ✅        | ❌         | ❌       | ❌     |

## How to Use SAM: Versatility and Power in Image Segmentation

The Segment Anything Model can be employed for a multitude of downstream tasks that go beyond its training data. This includes edge detection, object proposal generation, [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation), and preliminary text-to-mask prediction. With prompt engineering, SAM can swiftly adapt to new tasks and data distributions in a zero-shot manner, establishing it as a versatile and potent tool for all your image segmentation needs.

### SAM prediction example

!!! example "Segment with prompts"

    Segment image with given prompts.

    === "Python"

        ```python
        from ultralytics import SAM

        # Load a model
        model = SAM("sam_b.pt")

        # Display model information (optional)
        model.info()

        # Run inference with bboxes prompt
        results = model("ultralytics/assets/zidane.jpg", bboxes=[439, 437, 524, 709])

        # Run inference with single point
        results = model(points=[900, 370], labels=[1])

        # Run inference with multiple points
        results = model(points=[[400, 370], [900, 370]], labels=[1, 1])

        # Run inference with multiple points prompt per object
        results = model(points=[[[400, 370], [900, 370]]], labels=[[1, 1]])

        # Run inference with negative points prompt
        results = model(points=[[[400, 370], [900, 370]]], labels=[[1, 0]])
        ```

!!! example "Segment everything"

    Segment the whole image.

    === "Python"

        ```python
        from ultralytics import SAM

        # Load a model
        model = SAM("sam_b.pt")

        # Display model information (optional)
        model.info()

        # Run inference
        model("path/to/image.jpg")
        ```

    === "CLI"

        ```bash
        # Run inference with a SAM model
        yolo predict model=sam_b.pt source=path/to/image.jpg
        ```

- The logic here is to segment the whole image if you don't pass any prompts(bboxes/points/masks).

!!! example "SAMPredictor example"

    This way you can set image once and run prompts inference multiple times without running image encoder multiple times.

    === "Prompt inference"

        ```python
        from ultralytics.models.sam import Predictor as SAMPredictor

        # Create SAMPredictor
        overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model="mobile_sam.pt")
        predictor = SAMPredictor(overrides=overrides)

        # Set image
        predictor.set_image("ultralytics/assets/zidane.jpg")  # set with image file
        predictor.set_image(cv2.imread("ultralytics/assets/zidane.jpg"))  # set with np.ndarray
        results = predictor(bboxes=[439, 437, 524, 709])

        # Run inference with single point prompt
        results = predictor(points=[900, 370], labels=[1])

        # Run inference with multiple points prompt
        results = predictor(points=[[400, 370], [900, 370]], labels=[[1, 1]])

        # Run inference with negative points prompt
        results = predictor(points=[[[400, 370], [900, 370]]], labels=[[1, 0]])

        # Reset image
        predictor.reset_image()
        ```

    Segment everything with additional args.

    === "Segment everything"

        ```python
        from ultralytics.models.sam import Predictor as SAMPredictor

        # Create SAMPredictor
        overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model="mobile_sam.pt")
        predictor = SAMPredictor(overrides=overrides)

        # Segment with additional args
        results = predictor(source="ultralytics/assets/zidane.jpg", crop_n_layers=1, points_stride=64)
        ```

!!! note

    All the returned `results` in above examples are [Results](../modes/predict.md#working-with-results) object which allows access predicted masks and source image easily.

- More additional args for `Segment everything` see [`Predictor/generate` Reference](../reference/models/sam/predict.md).

## SAM Comparison vs YOLO

Here we compare Meta's SAM-b model with Ultralytics smallest segmentation model, [YOLO11n-seg](../tasks/segment.md):

| Model                                                                                          | Size<br><sup>(MB)</sup> | Parameters<br><sup>(M)</sup> | Speed (CPU)<br><sup>(ms/im)</sup> |
| ---------------------------------------------------------------------------------------------- | ----------------------- | ---------------------------- | --------------------------------- |
| [Meta SAM-b](sam.md)                                                                           | 375                     | 93.7                         | 49401                             |
| [MobileSAM](mobile-sam.md)                                                                     | 40.7                    | 10.1                         | 25381                             |
| [FastSAM-s](fast-sam.md) with YOLOv8 [backbone](https://www.ultralytics.com/glossary/backbone) | 23.7                    | 11.8                         | 55.9                              |
| Ultralytics [YOLOv8n-seg](yolov8.md)                                                           | **6.7** (11.7x smaller) | **3.4** (11.4x less)         | **24.5** (1061x faster)           |
| Ultralytics [YOLO11n-seg](yolo11.md)                                                           | **5.9** (13.2x smaller) | **2.9** (13.4x less)         | **30.1** (864x faster)            |

This comparison demonstrates the substantial differences in model sizes and speeds between SAM variants and YOLO segmentation models. While SAM provides unique automatic segmentation capabilities, YOLO models, particularly YOLOv8n-seg and YOLO11n-seg, are significantly smaller, faster, and more computationally efficient.

Tests run on a 2025 Apple M4 Pro with 24GB of RAM using `torch==2.6.0` and `ultralytics==8.3.90`. To reproduce this test:

!!! example

    === "Python"

        ```python
        from ultralytics import ASSETS, SAM, YOLO, FastSAM

        # Profile SAM2-t, SAM2-b, SAM-b, MobileSAM
        for file in ["sam_b.pt", "sam2_b.pt", "sam2_t.pt", "mobile_sam.pt"]:
            model = SAM(file)
            model.info()
            model(ASSETS)

        # Profile FastSAM-s
        model = FastSAM("FastSAM-s.pt")
        model.info()
        model(ASSETS)

        # Profile YOLO models
        for file_name in ["yolov8n-seg.pt", "yolo11n-seg.pt"]:
            model = YOLO(file_name)
            model.info()
            model(ASSETS)
        ```

## Auto-Annotation: A Quick Path to Segmentation Datasets

Auto-annotation is a key feature of SAM, allowing users to generate a [segmentation dataset](../datasets/segment/index.md) using a pre-trained detection model. This feature enables rapid and accurate annotation of a large number of images, bypassing the need for time-consuming manual labeling.

### Generate Your Segmentation Dataset Using a Detection Model

To auto-annotate your dataset with the Ultralytics framework, use the `auto_annotate` function as shown below:

!!! example

    === "Python"

        ```python
        from ultralytics.data.annotator import auto_annotate

        auto_annotate(data="path/to/images", det_model="yolo11x.pt", sam_model="sam_b.pt")
        ```

{% include "macros/sam-auto-annotate.md" %}

The `auto_annotate` function takes the path to your images, with optional arguments for specifying the pre-trained detection and SAM segmentation models, the device to run the models on, and the output directory for saving the annotated results.

Auto-annotation with pre-trained models can dramatically cut down the time and effort required for creating high-quality segmentation datasets. This feature is especially beneficial for researchers and developers dealing with large image collections, as it allows them to focus on model development and evaluation rather than manual annotation.

## Citations and Acknowledgements

If you find SAM useful in your research or development work, please consider citing our paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{kirillov2023segment,
              title={Segment Anything},
              author={Alexander Kirillov and Eric Mintun and Nikhila Ravi and Hanzi Mao and Chloe Rolland and Laura Gustafson and Tete Xiao and Spencer Whitehead and Alexander C. Berg and Wan-Yen Lo and Piotr Dollár and Ross Girshick},
              year={2023},
              eprint={2304.02643},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

We would like to express our gratitude to Meta AI for creating and maintaining this valuable resource for the [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) community.

## FAQ

### What is the Segment Anything Model (SAM) by Ultralytics?

The Segment Anything Model (SAM) by Ultralytics is a revolutionary image segmentation model designed for promptable segmentation tasks. It leverages advanced architecture, including image and prompt encoders combined with a lightweight mask decoder, to generate high-quality segmentation masks from various prompts such as spatial or text cues. Trained on the expansive [SA-1B dataset](https://ai.meta.com/datasets/segment-anything/), SAM excels in zero-shot performance, adapting to new image distributions and tasks without prior knowledge.

### How can I use the Segment Anything Model (SAM) for image segmentation?

You can use the Segment Anything Model (SAM) for image segmentation by running inference with various prompts such as bounding boxes or points. Here's an example using Python:

```python
from ultralytics import SAM

# Load a model
model = SAM("sam_b.pt")

# Segment with bounding box prompt
model("ultralytics/assets/zidane.jpg", bboxes=[439, 437, 524, 709])

# Segment with points prompt
model("ultralytics/assets/zidane.jpg", points=[900, 370], labels=[1])

# Segment with multiple points prompt
model("ultralytics/assets/zidane.jpg", points=[[400, 370], [900, 370]], labels=[[1, 1]])

# Segment with multiple points prompt per object
model("ultralytics/assets/zidane.jpg", points=[[[400, 370], [900, 370]]], labels=[[1, 1]])

# Segment with negative points prompt.
model("ultralytics/assets/zidane.jpg", points=[[[400, 370], [900, 370]]], labels=[[1, 0]])
```

Alternatively, you can run inference with SAM in the command line interface (CLI):

```bash
yolo predict model=sam_b.pt source=path/to/image.jpg
```

For more detailed usage instructions, visit the [Segmentation section](#sam-prediction-example).

### How do SAM and YOLO models compare in terms of performance?

Compared to YOLO models, SAM variants like SAM-b, SAM2-t, MobileSAM, and FastSAM-s are typically larger and slower but offer unique zero-shot segmentation capabilities. For example, Ultralytics [YOLOv8n-seg](../tasks/segment.md) is **11.7x smaller** and **1069x faster** than Meta's original SAM-b model, highlighting YOLO's significant advantage in speed and efficiency. Similarly, the newer [YOLO11n-seg](../tasks/segment.md) provides even smaller size and maintains impressive inference speed. This makes YOLO models ideal for applications requiring rapid, lightweight, and computationally efficient segmentation, while SAM models excel in flexible, promptable, and zero-shot segmentation tasks.

### How can I auto-annotate my dataset using SAM?

Ultralytics' SAM offers an auto-annotation feature that allows generating segmentation datasets using a pre-trained detection model. Here's an example in Python:

```python
from ultralytics.data.annotator import auto_annotate

auto_annotate(data="path/to/images", det_model="yolov8x.pt", sam_model="sam_b.pt")
```

This function takes the path to your images and optional arguments for pre-trained detection and SAM segmentation models, along with device and output directory specifications. For a complete guide, see [Auto-Annotation](#auto-annotation-a-quick-path-to-segmentation-datasets).

### What datasets are used to train the Segment Anything Model (SAM)?

SAM is trained on the extensive [SA-1B dataset](https://ai.meta.com/datasets/segment-anything/) which comprises over 1 billion masks across 11 million images. SA-1B is the largest segmentation dataset to date, providing high-quality and diverse [training data](https://www.ultralytics.com/glossary/training-data), ensuring impressive zero-shot performance in varied segmentation tasks. For more details, visit the [Dataset section](#key-features-of-the-segment-anything-model-sam).
>>>>>>> 0e451281e978a7c5f581cc5d9bb3d0f62e0bc632
