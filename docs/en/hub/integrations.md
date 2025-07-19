<<<<<<< HEAD
---
comments: true
description: Explore integration options for Ultralytics HUB. Currently featuring Roboflow for dataset integration and multiple export formats for your trained models.
keywords: Ultralytics HUB, Integrations, Roboflow, Dataset, Export, YOLOv5, YOLOv8, ONNX, CoreML, TensorRT, TensorFlow
---

# HUB Integrations

🚧 **Under Construction** 🚧

Welcome to the Integrations guide for [Ultralytics HUB](https://hub.ultralytics.com/)! We are in the process of expanding this section to provide you with comprehensive guidance on integrating your YOLOv5 and YOLOv8 models with various platforms and formats. Currently, Roboflow is our available dataset integration, with a wide range of export integrations for your trained models.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/lveF9iCMIzc?si=_Q4WB5kMB5qNe7q6"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Train Your Custom YOLO Models In A Few Clicks with Ultralytics HUB.
</p>

## Available Integrations

### Dataset Integrations

- **Roboflow**: Seamlessly import your datasets for training.

### Export Integrations

| Format                                                             | `format` Argument | Model                     | Metadata | Arguments                                           |
|--------------------------------------------------------------------|-------------------|---------------------------|----------|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -                 | `yolov8n.pt`              | ✅        | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript`     | `yolov8n.torchscript`     | ✅        | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`            | `yolov8n.onnx`            | ✅        | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](../integrations/openvino.md)                            | `openvino`        | `yolov8n_openvino_model/` | ✅        | `imgsz`, `half`, `int8`                             |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`          | `yolov8n.engine`          | ✅        | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`          | `yolov8n.mlpackage`       | ✅        | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`     | `yolov8n_saved_model/`    | ✅        | `imgsz`, `keras`, `int8`                            |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`              | `yolov8n.pb`              | ❌        | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`          | `yolov8n.tflite`          | ✅        | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`         | `yolov8n_edgetpu.tflite`  | ✅        | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`            | `yolov8n_web_model/`      | ✅        | `imgsz`, `half`, `int8`                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`          | `yolov8n_paddle_model/`   | ✅        | `imgsz`                                             |
| [NCNN](https://github.com/Tencent/ncnn)                            | `ncnn`            | `yolov8n_ncnn_model/`     | ✅        | `imgsz`, `half`                                     |

## Coming Soon

- Additional Dataset Integrations
- Detailed Export Integration Guides
- Step-by-Step Tutorials for Each Integration

## Need Immediate Assistance?

While we're in the process of creating detailed guides:

- Browse through other [HUB Docs](https://docs.ultralytics.com/hub/) for detailed guides and tutorials.
- Raise an issue on our [GitHub](https://github.com/ultralytics/hub/) for technical support.
- Join our [Discord Community](https://ultralytics.com/discord/) for live discussions and community support.

We appreciate your patience as we work to make this section comprehensive and user-friendly. Stay tuned for updates!
=======
---
comments: true
description: Explore seamless integrations between Ultralytics HUB and platforms like Roboflow. Learn how to import datasets, train models, and enhance your AI workflow.
keywords: Ultralytics HUB, Roboflow integration, dataset import, model training, AI, machine learning, model export, ONNX, OpenVINO
---

# Ultralytics HUB Integrations

Learn about [Ultralytics HUB](https://www.ultralytics.com/hub) integrations with various platforms and formats to streamline your [AI](https://www.ultralytics.com/glossary/artificial-intelligence-ai) workflows.

## Datasets

Seamlessly import your datasets into Ultralytics HUB for efficient [model training](../modes/train.md).

Once a dataset is imported, you can [train a model](./models.md#train-model) on it just as you would with native Ultralytics HUB datasets.

### Roboflow

You can easily filter Roboflow datasets on the Ultralytics HUB **Datasets** page.

![Ultralytics HUB screenshot of the Datasets page with Roboflow provider filter](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-datasets-page-roboflow-filter.avif)

Ultralytics HUB supports two types of integrations with Roboflow: **Universe** and **Workspace**.

#### Universe

The Roboflow Universe integration allows you to import one [dataset](https://www.ultralytics.com/glossary/benchmark-dataset) at a time into Ultralytics HUB from Roboflow.

##### Import

When exporting a Roboflow dataset, select the Ultralytics HUB format. This action redirects you to Ultralytics HUB and opens the **Dataset Import** dialog.

Import your Roboflow dataset by clicking the **Import** button.

![Ultralytics HUB screenshot of the Dataset Import dialog with an arrow pointing to the Import button](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-dataset-import-dialog.avif)

Next, you can train a model on your newly imported dataset.

![Ultralytics HUB screenshot of the Dataset page of a Roboflow Universe dataset with an arrow pointing to the Train Model button](https://github.com/ultralytics/docs/releases/download/0/hub-roboflow-universe-import-2.avif)

##### Remove

Navigate to the Dataset page of the Roboflow dataset you wish to remove. Open the dataset actions dropdown menu and click the **Remove** option.

![Ultralytics HUB screenshot of the Dataset page of a Roboflow Universe dataset with an arrow pointing to the Remove option](https://github.com/ultralytics/docs/releases/download/0/hub-roboflow-universe-remove.avif)

??? tip

    You can also remove an imported Roboflow dataset directly from the main **Datasets** page.

    ![Ultralytics HUB screenshot of the Datasets page with an arrow pointing to the Remove option of one of the Roboflow Universe datasets](https://github.com/ultralytics/docs/releases/download/0/hub-roboflow-remove-option.avif)

#### Workspace

The Roboflow Workspace integration allows you to import an entire Roboflow Workspace at once into Ultralytics HUB.

##### Import

Navigate to the **Integrations** page by clicking the **Integrations** button in the sidebar.

Enter your Roboflow Workspace private [API key](https://en.wikipedia.org/wiki/API_key) and click the **Add** button.

??? tip

    Clicking the **Get my API key** button will redirect you to your Roboflow Workspace settings, where you can find your private API key.

![Ultralytics HUB screenshot of the Integrations page with an arrow pointing to the Integrations button in the sidebar and one to the Add button](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-integrations-page.avif)

This connects your Ultralytics HUB account with your Roboflow Workspace, making your Roboflow datasets available within Ultralytics HUB.

![Ultralytics HUB screenshot of the Integrations page with an arrow pointing to one of the connected workspaces](https://github.com/ultralytics/docs/releases/download/0/hub-roboflow-workspace-import-2.avif)

Next, you can train a model using any of the datasets from the connected workspace.

![Ultralytics HUB screenshot of the Dataset page of a Roboflow Workspace dataset with an arrow pointing to the Train Model button](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-dataset-train-model.avif)

##### Remove

Navigate to the **Integrations** page via the sidebar. Click the **Unlink** button for the Roboflow Workspace you want to disconnect.

![Ultralytics HUB screenshot of the Integrations page  with an arrow pointing to the Integrations button in the sidebar and one to the Unlink button of one of the connected workspaces](https://github.com/ultralytics/docs/releases/download/0/hub-roboflow-workspace-remove-1.avif)

??? tip

    You can also unlink a connected Roboflow Workspace directly from the Dataset page of any dataset belonging to that workspace.

    ![Ultralytics HUB screenshot of the Dataset page of a Roboflow Workspace dataset with an arrow pointing to the remove option](https://github.com/ultralytics/docs/releases/download/0/hub-roboflow-workspace-remove-2.avif)

??? tip

    Alternatively, remove a connected Roboflow Workspace directly from the main **Datasets** page using the remove option associated with any dataset from that workspace.

    ![Ultralytics HUB screenshot of the Datasets page with an arrow pointing to the Remove option of one of the Roboflow Workspace datasets](https://github.com/ultralytics/docs/releases/download/0/hub-roboflow-remove-option.avif)

## Models

### Exports

After you train a model, you can [export it](./models.md#deploy-model) to 13 different formats using the [Export mode](../modes/export.md), including popular ones like [ONNX](https://www.ultralytics.com/glossary/onnx-open-neural-network-exchange), [OpenVINO](../integrations/openvino.md), [CoreML](../integrations/coreml.md), [TensorFlow](https://www.ultralytics.com/glossary/tensorflow), and [PaddlePaddle](../integrations/paddlepaddle.md).

![Ultralytics HUB screenshot of the Deploy tab inside the Model page with an arrow pointing to the Export card and all formats exported](https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub-deploy-export-formats.avif)

The available export formats are detailed in the table below.

{% include "macros/export-table.md" %}

## Exciting New Features on the Way 🎉

We are continuously working to expand Ultralytics HUB's integration capabilities. Upcoming features include:

- Additional [Dataset Integrations](../datasets/index.md)
- Detailed Export Integration Guides
- Step-by-Step [Tutorials](../guides/index.md) for Each Integration

## Stay Updated 🚧

This page is your go-to resource for the latest integration updates and feature rollouts. Stay connected through:

- **Newsletter:** Subscribe to [our Ultralytics newsletter](https://www.ultralytics.com/#newsletter) for announcements, releases, and early access updates.
- **Social Media:** Follow [Ultralytics on LinkedIn](https://www.linkedin.com/company/ultralytics) for behind-the-scenes content, product news, and community highlights.
- **Blog:** Dive into the [Ultralytics AI blog](https://www.ultralytics.com/blog) for in-depth articles, tutorials, and use-case spotlights.

## We Value Your Input 🗣️

Help shape the future of Ultralytics HUB by sharing your ideas, feedback, and integration requests through our [official contact form](https://www.ultralytics.com/contact).

## Thank You, Community! 🌍

Your [contributions](../help/contributing.md) and ongoing support fuel our commitment to pushing the boundaries of [AI innovation](https://github.com/ultralytics/ultralytics). Stay tuned—exciting things are just around the corner!

---

Excited for what's coming? Bookmark this page and check out our [Quickstart Guide](https://docs.ultralytics.com/quickstart/) to get started with our current tools while you wait. Get ready for a transformative AI and ML journey with Ultralytics! 🛠️🤖
>>>>>>> 0e451281e978a7c5f581cc5d9bb3d0f62e0bc632
