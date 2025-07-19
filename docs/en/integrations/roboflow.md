<<<<<<< HEAD
---
comments: true
description: Learn how to use Roboflow with Ultralytics for labeling and managing images for use in training, and for evaluating model performance.
keywords: Ultralytics, YOLOv8, Roboflow, vector analysis, confusion matrix, data management, image labeling
---

# Roboflow

[Roboflow](https://roboflow.com/?ref=ultralytics) has everything you need to build and deploy computer vision models. Connect Roboflow at any step in your pipeline with APIs and SDKs, or use the end-to-end interface to automate the entire process from image to inference. Whether you’re in need of [data labeling](https://roboflow.com/annotate?ref=ultralytics), [model training](https://roboflow.com/train?ref=ultralytics), or [model deployment](https://roboflow.com/deploy?ref=ultralytics), Roboflow gives you building blocks to bring custom computer vision solutions to your project.

!!! Warning

    Roboflow users can use Ultralytics under the [AGPL license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) or procure an [Enterprise license](https://ultralytics.com/license) directly from Ultralytics. Be aware that Roboflow does **not** provide Ultralytics licenses, and it is the responsibility of the user to ensure appropriate licensing.

In this guide, we are going to showcase how to find, label, and organize data for use in training a custom Ultralytics YOLOv8 model. Use the table of contents below to jump directly to a specific section:

- Gather data for training a custom YOLOv8 model
- Upload, convert and label data for YOLOv8 format
- Pre-process and augment data for model robustness
- Dataset management for [YOLOv8](https://docs.ultralytics.com/models/yolov8/)
- Export data in 40+ formats for model training
- Upload custom YOLOv8 model weights for testing and deployment
- Gather Data for Training a Custom YOLOv8 Model

Roboflow provides two services that can help you collect data for YOLOv8 models: [Universe](https://universe.roboflow.com/?ref=ultralytics) and [Collect](https://roboflow.com/collect?ref=ultralytics).

Universe is an online repository with over 250,000 vision datasets totalling over 100 million images.

<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_universe.png" alt="Roboflow Universe" width="800">
</p>

With a [free Roboflow account](https://app.roboflow.com/?ref=ultralytics), you can export any dataset available on Universe. To export a dataset, click the "Download this Dataset" button on any dataset.


<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_dataset.png" alt="Roboflow Universe dataset export" width="800">
</p>

For YOLOv8, select "YOLOv8" as the export format:

<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_data_format.png" alt="Roboflow Universe dataset export" width="800">
</p>

Universe also has a page that aggregates all [public fine-tuned YOLOv8 models uploaded to Roboflow](https://universe.roboflow.com/search?q=model:yolov8). You can use this page to explore pre-trained models you can use for testing or [for automated data labeling](https://docs.roboflow.com/annotate/use-roboflow-annotate/model-assisted-labeling) or to prototype with [Roboflow inference](https://roboflow.com/inference?ref=ultralytics).

If you want to gather images yourself, try [Collect](https://github.com/roboflow/roboflow-collect), an open source project that allows you to automatically gather images using a webcam on the edge. You can use text or image prompts with Collect to instruct what data should be collected, allowing you to capture only the useful data you need to build your vision model.

## Upload, Convert and Label Data for YOLOv8 Format

[Roboflow Annotate](https://docs.roboflow.com/annotate/use-roboflow-annotate) is an online annotation tool for use in labeling images for object detection, classification, and segmentation.

To label data for a YOLOv8 object detection, instance segmentation, or classification model, first create a project in Roboflow.

<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_create_project.png" alt="Create a Roboflow project" width="400">
</p>

Next, upload your images, and any pre-existing annotations you have from other tools ([using one of the 40+ supported import formats](https://roboflow.com/formats?ref=ultralytics)), into Roboflow.

<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_upload_data.png" alt="Upload images to Roboflow" width="800">
</p>

Select the batch of images you have uploaded on the Annotate page to which you are taken after uploading images. Then, click "Start Annotating" to label images.

To label with bounding boxes, press the `B` key on your keyboard or click the box icon in the sidebar. Click on a point where you want to start your bounding box, then drag to create the box:

<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_annotate.png" alt="Annotating an image in Roboflow" width="800">
</p>

A pop-up will appear asking you to select a class for your annotation once you have created an annotation.

To label with polygons, press the `P` key on your keyboard, or the polygon icon in the sidebar. With the polygon annotation tool enabled, click on individual points in the image to draw a polygon.

Roboflow offers a SAM-based label assistant with which you can label images faster than ever. SAM (Segment Anything Model) is a state-of-the-art computer vision model that can precisely label images. With SAM, you can significantly speed up the image labeling process. Annotating images with polygons becomes as simple as a few clicks, rather than the tedious process of precisely clicking points around an object.

To use the label assistant, click the cursor icon in the sidebar, SAM will be loaded for use in your project.

<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_annotate_interactive.png" alt="Annotating an image in Roboflow with SAM-powered label assist" width="800">
</p>

Hover over any object in the image and SAM will recommend an annotation. You can hover to find the right place to annotate, then click to create your annotation. To amend your annotation to be more or less specific, you can click inside or outside the annotation SAM has created on the document.

You can also add tags to images from the Tags panel in the sidebar. You can apply tags to data from a particular area, taken from a specific camera, and more. You can then use these tags to search through data for images matching a tag and generate versions of a dataset with images that contain a particular tag or set of tags.

<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_tags.png" alt="Adding tags to an image in Roboflow" width="300">
</p>

Models hosted on Roboflow can be used with Label Assist, an automated annotation tool that uses your YOLOv8 model to recommend annotations. To use Label Assist, first upload a YOLOv8 model to Roboflow (see instructions later in the guide). Then, click the magic wand icon in the left sidebar and select your model for use in Label Assist.

Choose a model, then click "Continue" to enable Label Assist:

<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_label_assist.png" alt="Enabling Label Assist" width="800">
</p>

When you open new images for annotation, Label Assist will trigger and recommend annotations.

<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_label_assist.png" alt="ALabel Assist recommending an annotation" width="800">
</p>

## Dataset Management for YOLOv8

Roboflow provides a suite of tools for understanding computer vision datasets.

First, you can use dataset search to find images that meet a semantic text description (i.e. find all images that contain people), or that meet a specified label (i.e. the image is associated with a specific tag). To use dataset search, click "Dataset" in the sidebar. Then, input a search query using the search bar and associated filters at the top of the page.

For example, the following text query finds images that contain people in a dataset:

<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_dataset_management.png" alt="Searching for an image" width="800">
</p>

You can narrow your search to images with a particular tag using the "Tags" selector:

<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_filter_by_tag.png" alt="Filter images by tag" width="350">
</p>

Before you start training a model with your dataset, we recommend using Roboflow [Health Check](https://docs.roboflow.com/datasets/dataset-health-check), a web tool that provides an insight into your dataset and how you can improve the dataset prior to training a vision model.

To use Health Check, click the "Health Check" sidebar link. A list of statistics will appear that show the average size of images in your dataset, class balance, a heatmap of where annotations are in your images, and more.

<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_dataset_health_check.png" alt="Roboflow Health Check analysis" width="800">
</p>

Health Check may recommend changes to help enhance dataset performance. For example, the class balance feature may show that there is an imbalance in labels that, if solved, may boost performance or your model.

## Export Data in 40+ Formats for Model Training

To export your data, you will need a dataset version. A version is a state of your dataset frozen-in-time. To create a version, first click "Versions" in the sidebar. Then, click the "Create New Version" button. On this page, you will be able to choose augmentations and preprocessing steps to apply to your dataset:

<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_generate_dataset.png" alt="Creating a dataset version on Roboflow" width="800">
</p>

For each augmentation you select, a pop-up will appear allowing you to tune the augmentation to your needs. Here is an example of tuning a brightness augmentation within specified parameters:

<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_augmentations.png" alt="Applying augmentations to a dataset" width="800">
</p>

When your dataset version has been generated, you can export your data into a range of formats. Click the "Export Dataset" button on your dataset version page to export your data:

<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_export_data.png" alt="Exporting a dataset" width="800">
</p>

You are now ready to train YOLOv8 on a custom dataset. Follow this [written guide](https://blog.roboflow.com/how-to-train-yolov8-on-a-custom-dataset/) and [YouTube video](https://www.youtube.com/watch?v=wuZtUMEiKWY) for step-by-step instructions or refer to the [Ultralytics documentation](https://docs.ultralytics.com/modes/train/).

## Upload Custom YOLOv8 Model Weights for Testing and Deployment

Roboflow offers an infinitely scalable API for deployed models and SDKs for use with NVIDIA Jetsons, Luxonis OAKs, Raspberry Pis, GPU-based devices, and more.

You can deploy YOLOv8 models by uploading YOLOv8 weights to Roboflow. You can do this in a few lines of Python code. Create a new Python file and add the following code:

```python
import roboflow  # install with 'pip install roboflow'

roboflow.login()

rf = roboflow.Roboflow()

project = rf.workspace(WORKSPACE_ID).project("football-players-detection-3zvbc")
dataset = project.version(VERSION).download("yolov8")

project.version(dataset.version).deploy(model_type="yolov8", model_path=f"{HOME}/runs/detect/train/")
```

In this code, replace the project ID and version ID with the values for your account and project. [Learn how to retrieve your Roboflow API key](https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key).

When you run the code above, you will be asked to authenticate. Then, your model will be uploaded and an API will be created for your project. This process can take up to 30 minutes to complete.

To test your model and find deployment instructions for supported SDKs, go to the "Deploy" tab in the Roboflow sidebar. At the top of this page, a widget will appear with which you can test your model. You can use your webcam for live testing or upload images or videos.

<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_test_project.png" alt="Running inference on an example image" width="800">
</p>

You can also use your uploaded model as a [labeling assistant](https://docs.roboflow.com/annotate/use-roboflow-annotate/model-assisted-labeling). This feature uses your trained model to recommend annotations on images uploaded to Roboflow.

## How to Evaluate YOLOv8 Models

Roboflow provides a range of features for use in evaluating models.

Once you have uploaded a model to Roboflow, you can access our model evaluation tool, which provides a confusion matrix showing the performance of your model as well as an interactive vector analysis plot. These features can help you find opportunities to improve your model.

To access a confusion matrix, go to your model page on the Roboflow dashboard, then click "View Detailed Evaluation":

<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_model_eval.png" alt="Start a Roboflow model evaluation" width="800">
</p>

A pop-up will appear showing a confusion matrix:

<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_confusion_matrix.png" alt="A confusion matrix" width="800">
</p>

Hover over a box on the confusion matrix to see the value associated with the box. Click on a box to see images in the respective category. Click on an image to view the model predictions and ground truth data associated with that image.

For more insights, click Vector Analysis. This will show a scatter plot of the images in your dataset, calculated using CLIP. The closer images are in the plot, the more similar they are, semantically. Each image is represented as a dot with a color between white and red. The more red the dot, the worse the model performed.

<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_vector_analysis.png" alt="A vector analysis plot" width="800">
</p>

You can use Vector Analysis to:

- Find clusters of images;
- Identify clusters where the model performs poorly, and;
- Visualize commonalities between images on which the model performs poorly.

## Learning Resources

Want to learn more about using Roboflow for creating YOLOv8 models? The following resources may be helpful in your work.

- [Train YOLOv8 on a Custom Dataset](https://github.com/roboflow/notebooks/blob/main/notebooks/train-yolov8-object-detection-on-custom-dataset.ipynb): Follow our interactive notebook that shows you how to train a YOLOv8 model on a custom dataset.
- [Autodistill](https://autodistill.github.io/autodistill/): Use large foundation vision models to label data for specific models. You can label images for use in training YOLOv8 classification, detection, and segmentation models with Autodistill.
- [Supervision](https://roboflow.github.io/supervision/): A Python package with helpful utilities for use in working with computer vision models. You can use supervision to filter detections, compute confusion matrices, and more, all in a few lines of Python code.
- [Roboflow Blog](https://blog.roboflow.com/): The Roboflow Blog features over 500 articles on computer vision, covering topics from how to train a YOLOv8 model to annotation best practices.
- [Roboflow YouTube channel](https://www.youtube.com/@Roboflow): Browse dozens of in-depth computer vision guides on our YouTube channel, covering topics from training YOLOv8 models to automated image labeling.

## Project Showcase

Below are a few of the many pieces of feedback we have received for using YOLOv8 and Roboflow together to create computer vision models.

<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_showcase_1.png" alt="Showcase image" width="500">
<img src="https://media.roboflow.com/ultralytics/rf_showcase_2.png" alt="Showcase image" width="500">
<img src="https://media.roboflow.com/ultralytics/rf_showcase_3.png" alt="Showcase image" width="500">
</p>
=======
---
comments: true
description: Learn how to gather, label, and deploy data for custom Ultralytics YOLO models using Roboflow's powerful tools. Optimize your computer vision pipeline effortlessly.
keywords: Roboflow, Ultralytics YOLO, data labeling, computer vision, model training, model deployment, dataset management, automated image annotation, AI tools
---

# Roboflow Integration

[Roboflow](https://roboflow.com/?ref=ultralytics) provides a suite of tools designed for building and deploying [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) models. You can integrate Roboflow at various stages of your development pipeline using their APIs and SDKs, or utilize its end-to-end interface to manage the process from image collection to inference. Roboflow offers functionalities for [data labeling](https://www.ultralytics.com/glossary/data-labeling), [model training](https://docs.ultralytics.com/modes/train/), and [model deployment](https://docs.ultralytics.com/guides/model-deployment-options/), providing components for developing custom computer vision solutions alongside Ultralytics tools.

!!! question "Licensing"

    Ultralytics offers two licensing options to accommodate different use cases:

    - **AGPL-3.0 License**: This [OSI-approved open-source license](https://www.ultralytics.com/legal/agpl-3-0-software-license) is ideal for students and enthusiasts, promoting open collaboration and knowledge sharing. See the [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) file for more details.
    - **Enterprise License**: Designed for commercial use, this license allows for the seamless integration of Ultralytics software and AI models into commercial products and services. If your scenario involves commercial applications, please reach out via [Ultralytics Licensing](https://www.ultralytics.com/license).

    For more details see the [Ultralytics Licensing page](https://www.ultralytics.com/license).

This guide demonstrates how to find, label, and organize data for training a custom [Ultralytics YOLO11](../models/yolo11.md) model using Roboflow.

- [Gather Data for Training a Custom YOLO11 Model](#gather-data-for-training-a-custom-yolo11-model)
- [Upload, Convert and Label Data for YOLO11 Format](#upload-convert-and-label-data-for-yolo11-format)
- [Pre-process and Augment Data for Model Robustness](#pre-process-and-augment-data-for-model-robustness)
- [Dataset Management for YOLO11](#dataset-management-for-yolo11)
- [Export Data in 40+ Formats for Model Training](#export-data-in-40-formats-for-model-training)
- [Upload Custom YOLO11 Model Weights for Testing and Deployment](#upload-custom-yolo11-model-weights-for-testing-and-deployment)
- [How to Evaluate YOLO11 Models](#how-to-evaluate-yolo11-models)
- [Learning Resources](#learning-resources)
- [Project Showcase](#project-showcase)
- [FAQ](#faq)

## Gather Data for Training a Custom YOLO11 Model

Roboflow offers two primary services to assist in data collection for Ultralytics [YOLO models](../models/index.md): Universe and Collect. For more general information on data collection strategies, refer to our [Data Collection and Annotation Guide](../guides/data-collection-and-annotation.md).

### Roboflow Universe

Roboflow Universe is an online repository featuring a large number of vision [datasets](../datasets/index.md).

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/roboflow-universe.avif" alt="Roboflow Universe" width="800">
</p>

With a Roboflow account, you can export datasets available on Universe. To export a dataset, use the "Download this Dataset" button on the relevant dataset page.

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/roboflow-universe-dataset-export.avif" alt="Roboflow Universe dataset export" width="800">
</p>

For compatibility with Ultralytics [YOLO11](../models/yolo11.md), select "YOLO11" as the export format:

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/roboflow-universe-dataset-export-1.avif" alt="Roboflow Universe dataset export format selection" width="800">
</p>

Universe also features a page aggregating public fine-tuned YOLO models uploaded to Roboflow. This can be useful for exploring pre-trained models for testing or automated data labeling.

### Roboflow Collect

If you prefer to gather images yourself, Roboflow Collect is an open-source project enabling automatic image collection via a webcam on edge devices. You can use text or image prompts to specify the data to be collected, helping capture only the necessary images for your vision model.

## Upload, Convert and Label Data for YOLO11 Format

Roboflow Annotate is an online tool for labeling images for various computer vision tasks, including [object detection](../tasks/detect.md), [classification](../tasks/classify.md), and [segmentation](../tasks/segment.md).

To label data for an Ultralytics [YOLO](../models/index.md) model (which supports detection, instance segmentation, classification, pose estimation, and OBB), begin by creating a project in Roboflow.

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/create-roboflow-project.avif" alt="Create a Roboflow project" width="400">
</p>

Next, upload your images and any existing annotations from other tools into Roboflow.

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/upload-images-to-roboflow.avif" alt="Upload images to Roboflow" width="800">
</p>

After uploading, you'll be directed to the Annotate page. Select the batch of uploaded images and click "Start Annotating" to begin labeling.

### Annotation Tools

- **Bounding Box Annotation**: Press `B` or click the box icon. Click and drag to create the [bounding box](https://www.ultralytics.com/glossary/bounding-box). A pop-up will prompt you to select a class for the annotation.

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/annotating-an-image-in-roboflow.avif" alt="Annotating an image in Roboflow with bounding boxes" width="800">
</p>

- **Polygon Annotation**: Used for [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation). Press `P` or click the polygon icon. Click points around the object to draw the polygon.

### Label Assistant (SAM Integration)

Roboflow integrates a [Segment Anything Model (SAM)](../models/sam.md)-based label assistant to potentially speed up annotation.

To use the label assistant, click the cursor icon in the sidebar. SAM will be enabled for your project.

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/annotating-image-roboflow-sam-powered-label-assist.avif" alt="Annotating an image in Roboflow with SAM-powered label assist" width="800">
</p>

Hover over an object, and SAM may suggest an annotation. Click to accept the annotation. You can refine the annotation's specificity by clicking inside or outside the suggested area.

### Tagging

You can add tags to images using the Tags panel in the sidebar. Tags can represent attributes like location, camera source, etc. These tags allow you to search for specific images and generate dataset versions containing images with particular tags.

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/adding-tags-to-image.avif" alt="Adding tags to an image in Roboflow" width="300">
</p>

### Label Assist (Model-Based)

Models hosted on Roboflow can be used with Label Assist, an automated annotation tool that leverages your trained [YOLO11](../models/yolo11.md) model to suggest annotations. First, upload your YOLO11 model weights to Roboflow (see instructions below). Then, activate Label Assist by clicking the magic wand icon in the left sidebar and selecting your model.

Choose your model and click "Continue" to enable Label Assist:

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/rf-label-assist.avif" alt="Enabling Label Assist in Roboflow" width="800">
</p>

When you open new images for annotation, Label Assist may automatically suggest annotations based on your model's predictions.

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/rf-label-assist.avif" alt="Label Assist recommending an annotation based on a trained model" width="800">
</p>

## Dataset Management for YOLO11

Roboflow provides several tools for understanding and managing your computer vision [datasets](../datasets/index.md).

### Dataset Search

Use dataset search to find images based on semantic text descriptions (e.g., "find all images containing people") or specific labels/tags. Access this feature by clicking "Dataset" in the sidebar and using the search bar and filters.

For example, searching for images containing people:

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/searching-for-an-image.avif" alt="Searching for an image in a Roboflow dataset" width="800">
</p>

You can refine searches using tags via the "Tags" selector:

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/filter-images-by-tag.avif" alt="Filtering images by tag in Roboflow" width="350">
</p>

### Health Check

Before training, use Roboflow Health Check to gain insights into your dataset and identify potential improvements. Access it via the "Health Check" sidebar link. It provides statistics on image sizes, class balance, annotation heatmaps, and more.

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/rf-dataset-health-check.avif" alt="Roboflow Health Check analysis dashboard" width="800">
</p>

Health Check might suggest changes to enhance performance, such as addressing class imbalances identified in the class balance feature. Understanding dataset health is crucial for effective [model training](../modes/train.md).

## Pre-process and Augment Data for Model Robustness

To export your data, you need to create a dataset version, which is a snapshot of your dataset at a specific point in time. Click "Versions" in the sidebar, then "Create New Version." Here, you can apply preprocessing steps and [data augmentations](https://www.ultralytics.com/glossary/data-augmentation) to potentially enhance model robustness.

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/creating-dataset-version-on-roboflow.avif" alt="Creating a dataset version on Roboflow with preprocessing and augmentation options" width="800">
</p>

For each selected augmentation, a pop-up allows you to fine-tune its parameters such as brightness. Proper augmentation can significantly improve model generalization, a key concept discussed in our [model training tips guide](../guides/model-training-tips.md).

## Export Data in 40+ Formats for Model Training

Once your dataset version is generated, you can export it in various formats suitable for model training. Click the "Export Dataset" button on the version page.

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/exporting-dataset.avif" alt="Exporting a dataset from Roboflow" width="800">
</p>

Select the "YOLO11" format for compatibility with Ultralytics training pipelines. You are now ready to train your custom [YOLO11](../models/yolo11.md) model. Refer to the [Ultralytics Train mode documentation](../modes/train.md) for detailed instructions on initiating training with your exported dataset.

## Upload Custom YOLO11 Model Weights for Testing and Deployment

Roboflow offers a scalable API for deployed models and SDKs compatible with devices like [NVIDIA Jetson](https://developer.nvidia.com/embedded-computing), [Luxonis OAK](https://www.luxonis.com/), [Raspberry Pi](../guides/raspberry-pi.md), and GPU-based systems. Explore various [model deployment options](../guides/model-deployment-options.md) in our guides.

You can deploy YOLO11 models by uploading their weights to Roboflow using a simple [Python](https://www.python.org/) script.

Create a new Python file and add the following code:

```python
import roboflow  # install with 'pip install roboflow'

# Log in to Roboflow (requires API key)
roboflow.login()

# Initialize Roboflow client
rf = roboflow.Roboflow()

# Define your workspace and project details
WORKSPACE_ID = "your-workspace-id"  # Replace with your actual Workspace ID
PROJECT_ID = "your-project-id"  # Replace with your actual Project ID
VERSION = 1  # Replace with your desired dataset version number
MODEL_PATH = "path/to/your/runs/detect/train/"  # Replace with the path to your YOLO11 training results directory

# Get project and version
project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)
dataset = project.version(VERSION)

# Upload model weights for deployment
# Ensure model_path points to the directory containing 'best.pt'
project.version(dataset.version).deploy(
    model_type="yolov8", model_path=MODEL_PATH
)  # Note: Use "yolov8" as model_type for YOLO11 compatibility in Roboflow deployment

print(f"Model from {MODEL_PATH} uploaded to Roboflow project {PROJECT_ID}, version {VERSION}.")
print("Deployment may take up to 30 minutes.")
```

In this code, replace `your-workspace-id`, `your-project-id`, the `VERSION` number, and the `MODEL_PATH` with the values specific to your Roboflow account, project, and local training results directory. Ensure the `MODEL_PATH` correctly points to the directory containing your trained `best.pt` weights file.

When you run the code above, you will be asked to authenticate (usually via an API key). Then, your model will be uploaded, and an API endpoint will be created for your project. This process can take up to 30 minutes to complete.

To test your model and find deployment instructions for supported SDKs, go to the "Deploy" tab in the Roboflow sidebar. At the top of this page, a widget will appear allowing you to test your model using your webcam or by uploading images or videos.

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/running-inference-example-image.avif" alt="Running inference on an example image using the Roboflow deployment widget" width="800">
</p>

Your uploaded model can also be used as a labeling assistant, suggesting annotations on new images based on its training.

## How to Evaluate YOLO11 Models

Roboflow provides features for evaluating model performance. Understanding [performance metrics](../guides/yolo-performance-metrics.md) is crucial for model iteration.

After uploading a model, access the model evaluation tool via your model page on the Roboflow dashboard. Click "View Detailed Evaluation."

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/roboflow-model-evaluation.avif" alt="Initiating a Roboflow model evaluation" width="800">
</p>

This tool displays a [confusion matrix](https://www.ultralytics.com/glossary/confusion-matrix) illustrating model performance and an interactive vector analysis plot using [CLIP](https://openai.com/research/clip) embeddings. These features help identify areas for model improvement.

The confusion matrix pop-up:

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/confusion-matrix.avif" alt="A confusion matrix displayed in Roboflow" width="800">
</p>

Hover over cells to see values, and click cells to view corresponding images with model predictions and ground truth data.

Click "Vector Analysis" for a scatter plot visualizing image similarity based on CLIP embeddings. Images closer together are semantically similar. Dots represent images, colored from white (good performance) to red (poor performance).

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/vector-analysis-plot.avif" alt="A vector analysis plot in Roboflow using CLIP embeddings" width="800">
</p>

Vector Analysis helps:

- Identify image clusters.
- Pinpoint clusters where the model performs poorly.
- Understand commonalities among images causing poor performance.

## Learning Resources

Explore these resources to learn more about using Roboflow with Ultralytics YOLO11:

- **[Train YOLO11 on a Custom Dataset (Colab)](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb)**: An interactive [Google Colab](../integrations/google-colab.md) notebook guiding you through training YOLO11 on your data.
- **[YOLO11 Documentation](../models/yolo11.md)**: Learn about training, exporting, and deploying YOLO11 models within the Ultralytics framework.
- **[Ultralytics Blog](https://www.ultralytics.com/blog/ultralytics-yolo11-has-arrived-redefine-whats-possible-in-ai)**: Features articles on computer vision, including [YOLO11 training](../modes/train.md) and annotation best practices.
- **[Ultralytics YouTube Channel](https://www.youtube.com/@Ultralytics)**: Offers in-depth video guides on computer vision topics, from model training to automated labeling and [deployment](../guides/model-deployment-options.md).

## Project Showcase

Feedback from users combining Ultralytics YOLO11 and Roboflow:

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/rf-showcase-1.avif" alt="Showcase image 1" width="500">
<img src="https://github.com/ultralytics/docs/releases/download/0/rf-showcase-2.avif" alt="Showcase image 2" width="500">
<img src="https://github.com/ultralytics/docs/releases/download/0/rf-showcase-3.avif" alt="Showcase image 3" width="500">
</p>

## FAQ

## Frequently Asked Questions

### How do I label data for YOLO11 models using Roboflow?

Use Roboflow Annotate. Create a project, upload images, and use the annotation tools (`B` for [bounding boxes](https://www.ultralytics.com/glossary/bounding-box), `P` for polygons) or the SAM-based label assistant for faster labeling. Detailed steps are available in the [Upload, Convert and Label Data section](#upload-convert-and-label-data-for-yolo11-format).

### What services does Roboflow offer for collecting YOLO11 training data?

Roboflow provides Universe (access to numerous [datasets](../datasets/index.md)) and Collect (automated image gathering via webcam). These can help acquire the necessary [training data](https://www.ultralytics.com/glossary/training-data) for your YOLO11 model, complementing strategies outlined in our [Data Collection Guide](../guides/data-collection-and-annotation.md).

### How can I manage and analyze my YOLO11 dataset using Roboflow?

Utilize Roboflow's dataset search, tagging, and Health Check features. Search finds images by text or tags, while Health Check analyzes dataset quality (class balance, image sizes, etc.) to guide improvements before training. See the [Dataset Management section](#dataset-management-for-yolo11) for details.

### How do I export my YOLO11 dataset from Roboflow?

Create a dataset version in Roboflow, apply desired preprocessing and [augmentations](https://www.ultralytics.com/glossary/data-augmentation), then click "Export Dataset" and select the YOLO11 format. The process is outlined in the [Export Data section](#export-data-in-40-formats-for-model-training). This prepares your data for use with Ultralytics [training pipelines](../modes/train.md).

### How can I integrate and deploy YOLO11 models with Roboflow?

Upload your trained YOLO11 weights to Roboflow using the provided Python script. This creates a deployable API endpoint. Refer to the [Upload Custom Weights section](#upload-custom-yolo11-model-weights-for-testing-and-deployment) for the script and instructions. Explore further [deployment options](../guides/model-deployment-options.md) in our documentation.
>>>>>>> 0e451281e978a7c5f581cc5d9bb3d0f62e0bc632
