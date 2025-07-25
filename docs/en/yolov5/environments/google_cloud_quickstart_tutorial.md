<<<<<<< HEAD
---
comments: true
description: Discover how to deploy YOLOv5 on a GCP Deep Learning VM for seamless object detection. Ideal for ML beginners and cloud learners. Get started with our easy-to-follow tutorial!
keywords: YOLOv5, Google Cloud Platform, GCP, Deep Learning VM, ML model training, object detection, AI tutorial, cloud-based AI, machine learning setup
---

# Mastering YOLOv5 🚀 Deployment on Google Cloud Platform (GCP) Deep Learning Virtual Machine (VM) ⭐

Embarking on the journey of artificial intelligence and machine learning can be exhilarating, especially when you leverage the power and flexibility of a cloud platform. Google Cloud Platform (GCP) offers robust tools tailored for machine learning enthusiasts and professionals alike. One such tool is the Deep Learning VM that is preconfigured for data science and ML tasks. In this tutorial, we will navigate through the process of setting up YOLOv5 on a GCP Deep Learning VM. Whether you’re taking your first steps in ML or you’re a seasoned practitioner, this guide is designed to provide you with a clear pathway to implementing object detection models powered by YOLOv5.

🆓 Plus, if you're a fresh GCP user, you’re in luck with a [$300 free credit offer](https://cloud.google.com/free/docs/gcp-free-tier#free-trial) to kickstart your projects.

In addition to GCP, explore other accessible quickstart options for YOLOv5, like our [Colab Notebook](https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb) <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"> for a browser-based experience, or the scalability of [Amazon AWS](https://docs.ultralytics.com/yolov5/environments/aws_quickstart_tutorial). Furthermore, container aficionados can utilize our official Docker image at [Docker Hub](https://hub.docker.com/r/ultralytics/yolov5) <img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"> for an encapsulated environment.

## Step 1: Create and Configure Your Deep Learning VM

Let’s begin by creating a virtual machine that’s tuned for deep learning:

1. Head over to the [GCP marketplace](https://console.cloud.google.com/marketplace/details/click-to-deploy-images/deeplearning) and select the **Deep Learning VM**.
2. Opt for a **n1-standard-8** instance; it offers a balance of 8 vCPUs and 30 GB of memory, ideally suited for our needs.
3. Next, select a GPU. This depends on your workload; even a basic one like the Tesla T4 will markedly accelerate your model training.
4. Tick the box for 'Install NVIDIA GPU driver automatically on first startup?' for hassle-free setup.
5. Allocate a 300 GB SSD Persistent Disk to ensure you don't bottleneck on I/O operations.
6. Hit 'Deploy' and let GCP do its magic in provisioning your custom Deep Learning VM.

This VM comes loaded with a treasure trove of preinstalled tools and frameworks, including the [Anaconda](https://docs.anaconda.com/anaconda/packages/pkg-docs/) Python distribution, which conveniently bundles all the necessary dependencies for YOLOv5.

![GCP Marketplace illustration of setting up a Deep Learning VM](https://user-images.githubusercontent.com/26833433/105811495-95863880-5f61-11eb-841d-c2f2a5aa0ffe.png)

## Step 2: Ready the VM for YOLOv5

Following the environment setup, let's get YOLOv5 up and running:

```bash
# Clone the YOLOv5 repository
git clone https://github.com/ultralytics/yolov5

# Change the directory to the cloned repository
cd yolov5

# Install the necessary Python packages from requirements.txt
pip install -r requirements.txt
```

This setup process ensures you're working with a Python environment version 3.8.0 or newer and PyTorch 1.8 or above. Our scripts smoothly download [models](https://github.com/ultralytics/yolov5/tree/master/models) and [datasets](https://github.com/ultralytics/yolov5/tree/master/data) rending from the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases), making it hassle-free to start model training.

## Step 3: Train and Deploy Your YOLOv5 Models 🌐

With the setup complete, you're ready to delve into training and inference with YOLOv5 on your GCP VM:

```bash
# Train a model on your data
python train.py

# Validate the trained model for Precision, Recall, and mAP
python val.py --weights yolov5s.pt

# Run inference using the trained model on your images or videos
python detect.py --weights yolov5s.pt --source path/to/images

# Export the trained model to other formats for deployment
python export.py --weights yolov5s.pt --include onnx coreml tflite
```

With just a few commands, YOLOv5 allows you to train custom object detection models tailored to your specific needs or utilize pre-trained weights for quick results on a variety of tasks.

![Terminal command image illustrating model training on a GCP Deep Learning VM](https://user-images.githubusercontent.com/26833433/142223900-275e5c9e-e2b5-43f7-a21c-35c4ca7de87c.png)

## Allocate Swap Space (optional)

For those dealing with hefty datasets, consider amplifying your GCP instance with an additional 64GB of swap memory:

```bash
sudo fallocate -l 64G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
free -h  # confirm the memory increment
```

### Concluding Thoughts

Congratulations! You are now empowered to harness the capabilities of YOLOv5 with the computational prowess of Google Cloud Platform. This combination provides scalability, efficiency, and versatility for your object detection tasks. Whether for personal projects, academic research, or industrial applications, you have taken a pivotal step into the world of AI and machine learning on the cloud.

Do remember to document your journey, share insights with the Ultralytics community, and leverage the collaborative arenas such as [GitHub discussions](https://github.com/ultralytics/yolov5/discussions) to grow further. Now, go forth and innovate with YOLOv5 and GCP! 🌟

Want to keep improving your ML skills and knowledge? Dive into our [documentation and tutorials](https://docs.ultralytics.com/) for more resources. Let your AI adventure continue!
=======
---
comments: true
description: Master Ultralytics YOLOv5 deployment on Google Cloud Platform Deep Learning VM. Perfect for AI beginners and experts to achieve high-performance object detection.
keywords: YOLOv5, Google Cloud Platform, GCP, Deep Learning VM, object detection, AI, machine learning, tutorial, cloud computing, GPU acceleration, Ultralytics
---

# Mastering YOLOv5 Deployment on Google Cloud Platform (GCP) Deep Learning VM

Embarking on the journey of [artificial intelligence (AI)](https://www.ultralytics.com/glossary/artificial-intelligence-ai) and [machine learning (ML)](https://www.ultralytics.com/glossary/machine-learning-ml) can be exhilarating, especially when you leverage the power and flexibility of a [cloud computing](https://www.ultralytics.com/glossary/cloud-computing) platform. Google Cloud Platform (GCP) offers robust tools tailored for ML enthusiasts and professionals alike. One such tool is the Deep Learning VM, preconfigured for data science and ML tasks. In this tutorial, we will navigate the process of setting up [Ultralytics YOLOv5](../../models/yolov5.md) on a [GCP Deep Learning VM](https://cloud.google.com/deep-learning-vm/docs). Whether you're taking your first steps in ML or you're a seasoned practitioner, this guide provides a clear pathway to implementing [object detection](https://www.ultralytics.com/glossary/object-detection) models powered by YOLOv5.

🆓 Plus, if you're a new GCP user, you're in luck with a [$300 free credit offer](https://cloud.google.com/free/docs/free-cloud-features#free-trial) to kickstart your projects.

In addition to GCP, explore other accessible quickstart options for YOLOv5, like our [Google Colab Notebook](https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb) <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"> for a browser-based experience, or the scalability of [Amazon AWS](./aws_quickstart_tutorial.md). Furthermore, container aficionados can utilize our official Docker image available on [Docker Hub](https://hub.docker.com/r/ultralytics/yolov5) <img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"> for an encapsulated environment, following our [Docker Quickstart Guide](../../guides/docker-quickstart.md).

## Step 1: Create and Configure Your Deep Learning VM

Let's begin by creating a virtual machine optimized for [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl):

1.  Navigate to the [GCP marketplace](https://cloud.google.com/marketplace) and select the **Deep Learning VM**.
2.  Choose an **n1-standard-8** instance; it offers a balance of 8 vCPUs and 30 GB of memory, suitable for many ML tasks.
3.  Select a [GPU](https://www.ultralytics.com/glossary/gpu-graphics-processing-unit). The choice depends on your workload; even a basic T4 GPU will significantly accelerate model training.
4.  Check the box for 'Install NVIDIA GPU driver automatically on first startup?' for a seamless setup.
5.  Allocate a 300 GB SSD Persistent Disk to prevent I/O bottlenecks.
6.  Click 'Deploy' and allow GCP to provision your custom Deep Learning VM.

This VM comes pre-loaded with essential tools and frameworks, including the [Anaconda](https://www.anaconda.com/) Python distribution, which conveniently bundles many necessary dependencies for YOLOv5.

![GCP Marketplace illustration of setting up a Deep Learning VM](https://github.com/ultralytics/docs/releases/download/0/gcp-deep-learning-vm-setup.avif)

## Step 2: Prepare the VM for YOLOv5

After setting up the environment, let's get YOLOv5 installed and ready:

```bash
# Clone the YOLOv5 repository from GitHub
git clone https://github.com/ultralytics/yolov5

# Navigate into the cloned repository directory
cd yolov5

# Install the required Python packages listed in requirements.txt
pip install -r requirements.txt
```

This setup process ensures you have a Python environment version 3.8.0 or newer and [PyTorch](https://www.ultralytics.com/glossary/pytorch) 1.8 or later. Our scripts automatically download [models](https://github.com/ultralytics/yolov5/tree/master/models) and [datasets](https://github.com/ultralytics/yolov5/tree/master/data) from the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases), simplifying the process of starting model training.

## Step 3: Train and Deploy Your YOLOv5 Models

With the setup complete, you are ready to [train](../../modes/train.md), [validate](../../modes/val.md), [predict](../../modes/predict.md), and [export](../../modes/export.md) with YOLOv5 on your GCP VM:

```bash
# Train a YOLOv5 model on your dataset (e.g., yolov5s)
python train.py --data coco128.yaml --weights yolov5s.pt --img 640

# Validate the trained model to check Precision, Recall, and mAP
python val.py --weights yolov5s.pt --data coco128.yaml

# Run inference using the trained model on images or videos
python detect.py --weights yolov5s.pt --source path/to/your/images_or_videos

# Export the trained model to various formats like ONNX, CoreML, TFLite for deployment
python export.py --weights yolov5s.pt --include onnx coreml tflite
```

Using just a few commands, YOLOv5 enables you to train custom [object detection](https://docs.ultralytics.com/tasks/detect/) models tailored to your specific needs or utilize pre-trained weights for rapid results across various tasks. Explore different [model deployment options](../../guides/model-deployment-options.md) after exporting.

![Terminal command image illustrating model training on a GCP Deep Learning VM](https://github.com/ultralytics/docs/releases/download/0/terminal-command-model-training.avif)

## Allocate Swap Space (Optional)

If you are working with particularly large datasets that might exceed your VM's RAM, consider adding swap space to prevent memory errors:

```bash
# Allocate a 64GB swap file
sudo fallocate -l 64G /swapfile

# Set the correct permissions for the swap file
sudo chmod 600 /swapfile

# Set up the Linux swap area
sudo mkswap /swapfile

# Enable the swap file
sudo swapon /swapfile

# Verify the swap space allocation (should show increased swap memory)
free -h
```

## Training Custom Datasets

To train YOLOv5 on your custom dataset within GCP, follow these general steps:

1.  Prepare your dataset according to the YOLOv5 format (images and corresponding label files). See our [datasets overview](../../datasets/index.md) for guidance.
2.  Upload your dataset to your GCP VM using `gcloud compute scp` or the web console's SSH feature.
3.  Create a dataset configuration YAML file (`custom_dataset.yaml`) that specifies the paths to your training and validation data, the number of classes, and class names.
4.  Begin the [training process](../../modes/train.md) using your custom dataset YAML and potentially starting from pre-trained weights:

    ```bash
    # Example: Train YOLOv5s on a custom dataset for 100 epochs
    python train.py --img 640 --batch 16 --epochs 100 --data custom_dataset.yaml --weights yolov5s.pt
    ```

For comprehensive instructions on preparing data and training with custom datasets, consult the [Ultralytics YOLOv5 Train documentation](../../modes/train.md).

## Leveraging Cloud Storage

For efficient data management, especially with large datasets or numerous experiments, integrate your YOLOv5 workflow with [Google Cloud Storage](https://cloud.google.com/storage):

```bash
# Ensure Google Cloud SDK is installed and initialized
# If not installed: curl https://sdk.cloud.google.com/ | bash
# Then initialize: gcloud init

# Example: Copy your dataset from a GCS bucket to your VM
gsutil cp -r gs://your-data-bucket/my_dataset ./datasets/

# Example: Copy trained model weights from your VM to a GCS bucket
gsutil cp -r ./runs/train/exp/weights gs://your-models-bucket/yolov5_custom_weights/
```

This approach allows you to store large datasets and trained models securely and cost-effectively in the cloud, minimizing the storage requirements on your VM instance.

## Concluding Thoughts

Congratulations! You are now equipped to harness the capabilities of Ultralytics YOLOv5 combined with the computational power of Google Cloud Platform. This setup provides scalability, efficiency, and versatility for your object detection projects. Whether for personal exploration, academic research, or building industrial [solutions](../../solutions/index.md), you have taken a significant step into the world of AI and ML on the cloud.

Consider using [Ultralytics HUB](../../hub/index.md) for a streamlined, no-code experience to train and manage your models.

Remember to document your progress, share insights with the vibrant Ultralytics community, and utilize resources like [GitHub discussions](https://github.com/ultralytics/yolov5/discussions) for collaboration and support. Now, go forth and innovate with YOLOv5 and GCP!

Want to continue enhancing your ML skills? Dive into our [documentation](../../quickstart.md) and explore the [Ultralytics Blog](https://www.ultralytics.com/blog) for more tutorials and insights. Let your AI adventure continue!
>>>>>>> 0e451281e978a7c5f581cc5d9bb3d0f62e0bc632
