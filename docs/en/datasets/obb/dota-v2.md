<<<<<<< HEAD
---
comments: true
description: Delve into DOTA v2, an Oriented Bounding Box (OBB) aerial imagery dataset with 1.7 million instances and 11,268 images.
keywords: DOTA v2, object detection, aerial images, computer vision, deep learning, annotations, oriented bounding boxes, OBB
---

# DOTA v2 Dataset with OBB

[DOTA v2](https://captain-whu.github.io/DOTA/index.html) stands as a specialized dataset, emphasizing object detection in aerial images. Originating from the DOTA series of datasets, it offers annotated images capturing a diverse array of aerial scenes with Oriented Bounding Boxes (OBB).

![DOTA v2 classes visual](https://user-images.githubusercontent.com/26833433/259461765-72fdd0d8-266b-44a9-8199-199329bf5ca9.jpg)

## Key Features

- Collection from various sensors and platforms, with image sizes ranging from 800 × 800 to 20,000 × 20,000 pixels.
- Features more than 1.7M Oriented Bounding Boxes across 18 categories.
- Encompasses multiscale object detection.
- Instances are annotated by experts using arbitrary (8 d.o.f.) quadrilateral, capturing objects of different scales, orientations, and shapes.

## Dataset Versions

### DOTA-v1.0

- Contains 15 common categories.
- Comprises 2,806 images with 188,282 instances.
- Split ratios: 1/2 for training, 1/6 for validation, and 1/3 for testing.

### DOTA-v1.5

- Incorporates the same images as DOTA-v1.0.
- Very small instances (less than 10 pixels) are also annotated.
- Addition of a new category: "container crane".
- A total of 403,318 instances.
- Released for the DOAI Challenge 2019 on Object Detection in Aerial Images.

### DOTA-v2.0

- Collections from Google Earth, GF-2 Satellite, and other aerial images.
- Contains 18 common categories.
- Comprises 11,268 images with a whopping 1,793,658 instances.
- New categories introduced: "airport" and "helipad".
- Image splits:
    - Training: 1,830 images with 268,627 instances.
    - Validation: 593 images with 81,048 instances.
    - Test-dev: 2,792 images with 353,346 instances.
    - Test-challenge: 6,053 images with 1,090,637 instances.

## Dataset Structure

DOTA v2 exhibits a structured layout tailored for OBB object detection challenges:

- **Images**: A vast collection of high-resolution aerial images capturing diverse terrains and structures.
- **Oriented Bounding Boxes**: Annotations in the form of rotated rectangles encapsulating objects irrespective of their orientation, ideal for capturing objects like airplanes, ships, and buildings.

## Applications

DOTA v2 serves as a benchmark for training and evaluating models specifically tailored for aerial image analysis. With the inclusion of OBB annotations, it provides a unique challenge, enabling the development of specialized object detection models that cater to aerial imagery's nuances.

## Dataset YAML

Typically, datasets incorporate a YAML (Yet Another Markup Language) file detailing the dataset's configuration. For DOTA v2, a hypothetical `DOTAv2.yaml` could be used. For accurate paths and configurations, it's vital to consult the dataset's official repository or documentation.

!!! Example "DOTAv2.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/DOTAv2.yaml"
    ```

## Usage

To train a model on the DOTA v2 dataset, you can utilize the following code snippets. Always refer to your model's documentation for a thorough list of available arguments.

!!! Warning

    Please note that all images and associated annotations in the DOTAv2 dataset can be used for academic purposes, but commercial use is prohibited. Your understanding and respect for the dataset creators' wishes are greatly appreciated!

!!! Example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Create a new YOLOv8n-OBB model from scratch
        model = YOLO('yolov8n-obb.yaml')

        # Train the model on the DOTAv2 dataset
        results = model.train(data='DOTAv2.yaml', epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Train a new YOLOv8n-OBB model on the DOTAv2 dataset
        yolo detect train data=DOTAv2.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

## Sample Data and Annotations

Having a glance at the dataset illustrates its depth:

![Dataset sample image](https://captain-whu.github.io/DOTA/images/instances-DOTA.jpg)

- **DOTA v2**: This snapshot underlines the complexity of aerial scenes and the significance of Oriented Bounding Box annotations, capturing objects in their natural orientation.

The dataset's richness offers invaluable insights into object detection challenges exclusive to aerial imagery.

## Citations and Acknowledgments

For those leveraging DOTA v2 in their endeavors, it's pertinent to cite the relevant research papers:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @article{9560031,
          author={Ding, Jian and Xue, Nan and Xia, Gui-Song and Bai, Xiang and Yang, Wen and Yang, Michael and Belongie, Serge and Luo, Jiebo and Datcu, Mihai and Pelillo, Marcello and Zhang, Liangpei},
          journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
          title={Object Detection in Aerial Images: A Large-Scale Benchmark and Challenges},
          year={2021},
          volume={},
          number={},
          pages={1-1},
          doi={10.1109/TPAMI.2021.3117983}
        }
        ```

A special note of gratitude to the team behind DOTA v2 for their commendable effort in curating this dataset. For an exhaustive understanding of the dataset and its nuances, please visit the [official DOTA v2 website](https://captain-whu.github.io/DOTA/index.html).
=======
---
comments: true
description: Explore the DOTA dataset for object detection in aerial images, featuring 1.7M Oriented Bounding Boxes across 18 categories. Ideal for aerial image analysis.
keywords: DOTA dataset, object detection, aerial images, oriented bounding boxes, OBB, DOTA v1.0, DOTA v1.5, DOTA v2.0, multiscale detection, Ultralytics
---

# DOTA Dataset with OBB

[DOTA](https://captain-whu.github.io/DOTA/index.html) stands as a specialized dataset, emphasizing [object detection](https://www.ultralytics.com/glossary/object-detection) in aerial images. Originating from the DOTA series of datasets, it offers annotated images capturing a diverse array of aerial scenes with [Oriented Bounding Boxes (OBB)](https://docs.ultralytics.com/datasets/obb/).

![DOTA classes visual](https://github.com/ultralytics/docs/releases/download/0/dota-classes-visual.avif)

## Key Features

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/JjQ-URE0LJE"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Train Ultralytics YOLO11 on the DOTA Dataset for Oriented Bounding Boxes in Google Colab
</p>

- Collection from various sensors and platforms, with image sizes ranging from 800 × 800 to 20,000 × 20,000 pixels.
- Features more than 1.7M Oriented Bounding Boxes across 18 categories.
- Encompasses multiscale object detection.
- Instances are annotated by experts using arbitrary (8 d.o.f.) quadrilateral, capturing objects of different scales, orientations, and shapes.

## Dataset Versions

### DOTA-v1.0

- Contains 15 common categories.
- Comprises 2,806 images with 188,282 instances.
- Split ratios: 1/2 for training, 1/6 for validation, and 1/3 for testing.

### DOTA-v1.5

- Incorporates the same images as DOTA-v1.0.
- Very small instances (less than 10 pixels) are also annotated.
- Addition of a new category: "container crane".
- A total of 403,318 instances.
- Released for the [DOAI Challenge 2019 on Object Detection in Aerial Images](https://captain-whu.github.io/DOAI2019/challenge.html).

### DOTA-v2.0

- Collections from Google Earth, GF-2 Satellite, and other aerial images.
- Contains 18 common categories.
- Comprises 11,268 images with a whopping 1,793,658 instances.
- New categories introduced: "airport" and "helipad".
- Image splits:
    - Training: 1,830 images with 268,627 instances.
    - Validation: 593 images with 81,048 instances.
    - Test-dev: 2,792 images with 353,346 instances.
    - Test-challenge: 6,053 images with 1,090,637 instances.

## Dataset Structure

DOTA exhibits a structured layout tailored for OBB object detection challenges:

- **Images**: A vast collection of high-resolution aerial images capturing diverse terrains and structures.
- **Oriented Bounding Boxes**: Annotations in the form of rotated rectangles encapsulating objects irrespective of their orientation, ideal for capturing objects like airplanes, ships, and buildings.

## Applications

DOTA serves as a benchmark for training and evaluating models specifically tailored for aerial image analysis. With the inclusion of OBB annotations, it provides a unique challenge, enabling the development of specialized [object detection](https://docs.ultralytics.com/tasks/detect/) models that cater to aerial imagery's nuances. The dataset is particularly valuable for applications in remote sensing, surveillance, and environmental monitoring.

## Dataset YAML

Typically, datasets incorporate a YAML (Yet Another Markup Language) file detailing the dataset's configuration. For DOTA v1 and DOTA v1.5, Ultralytics provides `DOTAv1.yaml` and `DOTAv1.5.yaml` files. For additional details on these as well as DOTA v2 please consult DOTA's official repository and documentation.

!!! example "DOTAv1.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/DOTAv1.yaml"
    ```

## Split DOTA images

To train DOTA dataset, we split original DOTA images with high-resolution into images with 1024x1024 resolution in multiscale way. This preprocessing step is crucial for efficient training as the original images can be extremely large.

!!! example "Split images"

    === "Python"

        ```python
        from ultralytics.data.split_dota import split_test, split_trainval

        # split train and val set, with labels.
        split_trainval(
            data_root="path/to/DOTAv1.0/",
            save_dir="path/to/DOTAv1.0-split/",
            rates=[0.5, 1.0, 1.5],  # multiscale
            gap=500,
        )
        # split test set, without labels.
        split_test(
            data_root="path/to/DOTAv1.0/",
            save_dir="path/to/DOTAv1.0-split/",
            rates=[0.5, 1.0, 1.5],  # multiscale
            gap=500,
        )
        ```

## Usage

To train a model on the DOTA v1 dataset, you can utilize the following code snippets. Always refer to your model's documentation for a thorough list of available arguments. For those looking to experiment with a smaller subset first, consider using the [DOTA8 dataset](https://docs.ultralytics.com/datasets/obb/dota8/), which contains just 8 images for quick testing.

!!! warning

    Please note that all images and associated annotations in the DOTAv1 dataset can be used for academic purposes, but commercial use is prohibited. Your understanding and respect for the dataset creators' wishes are greatly appreciated!

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Create a new YOLO11n-OBB model from scratch
        model = YOLO("yolo11n-obb.yaml")

        # Train the model on the DOTAv1 dataset
        results = model.train(data="DOTAv1.yaml", epochs=100, imgsz=1024)
        ```

    === "CLI"

        ```bash
        # Train a new YOLO11n-OBB model on the DOTAv1 dataset
        yolo obb train data=DOTAv1.yaml model=yolo11n-obb.pt epochs=100 imgsz=1024
        ```

## Sample Data and Annotations

Having a glance at the dataset illustrates its depth:

![Dataset sample image](https://github.com/ultralytics/docs/releases/download/0/instances-DOTA.avif)

- **DOTA examples**: This snapshot underlines the complexity of aerial scenes and the significance of Oriented [Bounding Box](https://www.ultralytics.com/glossary/bounding-box) annotations, capturing objects in their natural orientation.

The dataset's richness offers invaluable insights into object detection challenges exclusive to aerial imagery. The [DOTA-v2.0 dataset](https://www.ultralytics.com/blog/exploring-the-best-computer-vision-datasets-in-2025) has become particularly popular for remote sensing and aerial surveillance projects due to its comprehensive annotations and diverse object categories.

## Citations and Acknowledgments

For those leveraging DOTA in their endeavors, it's pertinent to cite the relevant research papers:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{9560031,
          author={Ding, Jian and Xue, Nan and Xia, Gui-Song and Bai, Xiang and Yang, Wen and Yang, Michael and Belongie, Serge and Luo, Jiebo and Datcu, Mihai and Pelillo, Marcello and Zhang, Liangpei},
          journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
          title={Object Detection in Aerial Images: A Large-Scale Benchmark and Challenges},
          year={2021},
          volume={},
          number={},
          pages={1-1},
          doi={10.1109/TPAMI.2021.3117983}
        }
        ```

A special note of gratitude to the team behind the DOTA datasets for their commendable effort in curating this dataset. For an exhaustive understanding of the dataset and its nuances, please visit the [official DOTA website](https://captain-whu.github.io/DOTA/index.html).

## FAQ

### What is the DOTA dataset and why is it important for object detection in aerial images?

The [DOTA dataset](https://captain-whu.github.io/DOTA/index.html) is a specialized dataset focused on object detection in aerial images. It features Oriented Bounding Boxes (OBB), providing annotated images from diverse aerial scenes. DOTA's diversity in object orientation, scale, and shape across its 1.7M annotations and 18 categories makes it ideal for developing and evaluating models tailored for aerial imagery analysis, such as those used in surveillance, environmental monitoring, and disaster management.

### How does the DOTA dataset handle different scales and orientations in images?

DOTA utilizes Oriented Bounding Boxes (OBB) for annotation, which are represented by rotated rectangles encapsulating objects regardless of their orientation. This method ensures that objects, whether small or at different angles, are accurately captured. The dataset's multiscale images, ranging from 800 × 800 to 20,000 × 20,000 pixels, further allow for the detection of both small and large objects effectively. This approach is particularly valuable for aerial imagery where objects appear at various angles and scales.

### How can I train a model using the DOTA dataset?

To train a model on the DOTA dataset, you can use the following example with [Ultralytics YOLO](https://docs.ultralytics.com/tasks/obb/):

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Create a new YOLO11n-OBB model from scratch
        model = YOLO("yolo11n-obb.yaml")

        # Train the model on the DOTAv1 dataset
        results = model.train(data="DOTAv1.yaml", epochs=100, imgsz=1024)
        ```

    === "CLI"

        ```bash
        # Train a new YOLO11n-OBB model on the DOTAv1 dataset
        yolo obb train data=DOTAv1.yaml model=yolo11n-obb.pt epochs=100 imgsz=1024
        ```

For more details on how to split and preprocess the DOTA images, refer to the [split DOTA images section](#split-dota-images).

### What are the differences between DOTA-v1.0, DOTA-v1.5, and DOTA-v2.0?

- **DOTA-v1.0**: Includes 15 common categories across 2,806 images with 188,282 instances. The dataset is split into training, validation, and testing sets.
- **DOTA-v1.5**: Builds upon DOTA-v1.0 by annotating very small instances (less than 10 pixels) and adding a new category, "container crane," totaling 403,318 instances.
- **DOTA-v2.0**: Expands further with annotations from Google Earth and GF-2 Satellite, featuring 11,268 images and 1,793,658 instances. It includes new categories like "airport" and "helipad."

For a detailed comparison and additional specifics, check the [dataset versions section](#dataset-versions).

### How can I prepare high-resolution DOTA images for training?

DOTA images, which can be very large, are split into smaller resolutions for manageable training. Here's a Python snippet to split images:

!!! example

    === "Python"

        ```python
        from ultralytics.data.split_dota import split_test, split_trainval

        # split train and val set, with labels.
        split_trainval(
            data_root="path/to/DOTAv1.0/",
            save_dir="path/to/DOTAv1.0-split/",
            rates=[0.5, 1.0, 1.5],  # multiscale
            gap=500,
        )
        # split test set, without labels.
        split_test(
            data_root="path/to/DOTAv1.0/",
            save_dir="path/to/DOTAv1.0-split/",
            rates=[0.5, 1.0, 1.5],  # multiscale
            gap=500,
        )
        ```

This process facilitates better training efficiency and model performance. For detailed instructions, visit the [split DOTA images section](#split-dota-images).
>>>>>>> 0e451281e978a7c5f581cc5d9bb3d0f62e0bc632
