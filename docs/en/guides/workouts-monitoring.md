<<<<<<< HEAD
---
comments: true
description: Workouts Monitoring Using Ultralytics YOLOv8
keywords: Ultralytics, YOLOv8, Object Detection, Pose Estimation, PushUps, PullUps, Ab workouts, Notebook, IPython Kernel, CLI, Python SDK
---

# Workouts Monitoring using Ultralytics YOLOv8 🚀

Monitoring workouts through pose estimation with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics/) enhances exercise assessment by accurately tracking key body landmarks and joints in real-time. This technology provides instant feedback on exercise form, tracks workout routines, and measures performance metrics, optimizing training sessions for users and trainers alike.

## Advantages of Workouts Monitoring?

- **Optimized Performance:** Tailoring workouts based on monitoring data for better results.
- **Goal Achievement:** Track and adjust fitness goals for measurable progress.
- **Personalization:** Customized workout plans based on individual data for effectiveness.
- **Health Awareness:** Early detection of patterns indicating health issues or overtraining.
- **Informed Decisions:** Data-driven decisions for adjusting routines and setting realistic goals.

## Real World Applications

|                                                  Workouts Monitoring                                                   |                                                  Workouts Monitoring                                                   |
|:----------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------:|
| ![PushUps Counting](https://github.com/RizwanMunawar/ultralytics/assets/62513924/cf016a41-589f-420f-8a8c-2cc8174a16de) | ![PullUps Counting](https://github.com/RizwanMunawar/ultralytics/assets/62513924/cb20f316-fac2-4330-8445-dcf5ffebe329) |
|                                                    PushUps Counting                                                    |                                                    PullUps Counting                                                    |

!!! Example "Workouts Monitoring Example"

    === "Workouts Monitoring"
        ```python
        from ultralytics import YOLO
        from ultralytics.solutions import ai_gym
        import cv2

        model = YOLO("yolov8n-pose.pt")
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"

        gym_object = ai_gym.AIGym()  # init AI GYM module
        gym_object.set_args(line_thickness=2,
                            view_img=True,
                            pose_type="pushup",
                            kpts_to_check=[6, 8, 10])

        frame_count = 0
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                exit(0)
            frame_count += 1
            results = model.predict(im0, verbose=False)
            im0 = gym_object.start_counting(im0, results, frame_count)
        ```

    === "Workouts Monitoring with Save Output"
        ```python
        from ultralytics import YOLO
        from ultralytics.solutions import ai_gym
        import cv2

        model = YOLO("yolov8n-pose.pt")
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"

        video_writer = cv2.VideoWriter("workouts.avi",
                                       cv2.VideoWriter_fourcc(*'mp4v'),
                                       int(cap.get(5)),
                                       (int(cap.get(3)), int(cap.get(4))))

        gym_object = ai_gym.AIGym()  # init AI GYM module
        gym_object.set_args(line_thickness=2,
                            view_img=True,
                            pose_type="pushup",
                            kpts_to_check=[6, 8, 10])

        frame_count = 0
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                exit(0)
            frame_count += 1
            results = model.predict(im0, verbose=False)
            im0 = gym_object.start_counting(im0, results, frame_count)
            video_writer.write(im0)

        video_writer.release()
        ```

???+ tip "Support"

    "pushup", "pullup" and "abworkout" supported

### KeyPoints Map

![keyPoints Order Ultralytics YOLOv8 Pose](https://github.com/ultralytics/ultralytics/assets/62513924/f45d8315-b59f-47b7-b9c8-c61af1ce865b)

### Arguments `set_args`

| Name            | Type   | Default  | Description                                                                            |
|-----------------|--------|----------|----------------------------------------------------------------------------------------|
| kpts_to_check   | `list` | `None`   | List of three keypoints index, for counting specific workout, followed by keypoint Map |
| view_img        | `bool` | `False`  | Display the frame with counts                                                          |
| line_thickness  | `int`  | `2`      | Increase the thickness of count value                                                  |
| pose_type       | `str`  | `pushup` | Pose that need to be monitored, "pullup" and "abworkout" also supported                |
| pose_up_angle   | `int`  | `145`    | Pose Up Angle value                                                                    |
| pose_down_angle | `int`  | `90`     | Pose Down Angle value                                                                  |

### Arguments `model.predict`

| Name            | Type           | Default                | Description                                                                |
|-----------------|----------------|------------------------|----------------------------------------------------------------------------|
| `source`        | `str`          | `'ultralytics/assets'` | source directory for images or videos                                      |
| `conf`          | `float`        | `0.25`                 | object confidence threshold for detection                                  |
| `iou`           | `float`        | `0.7`                  | intersection over union (IoU) threshold for NMS                            |
| `imgsz`         | `int or tuple` | `640`                  | image size as scalar or (h, w) list, i.e. (640, 480)                       |
| `half`          | `bool`         | `False`                | use half precision (FP16)                                                  |
| `device`        | `None or str`  | `None`                 | device to run on, i.e. cuda device=0/1/2/3 or device=cpu                   |
| `max_det`       | `int`          | `300`                  | maximum number of detections per image                                     |
| `vid_stride`    | `bool`         | `False`                | video frame-rate stride                                                    |
| `stream_buffer` | `bool`         | `False`                | buffer all streaming frames (True) or return the most recent frame (False) |
| `visualize`     | `bool`         | `False`                | visualize model features                                                   |
| `augment`       | `bool`         | `False`                | apply image augmentation to prediction sources                             |
| `agnostic_nms`  | `bool`         | `False`                | class-agnostic NMS                                                         |
| `retina_masks`  | `bool`         | `False`                | use high-resolution segmentation masks                                     |
| `classes`       | `None or list` | `None`                 | filter results by class, i.e. classes=0, or classes=[0,2,3]                |
=======
---
comments: true
description: Optimize your fitness routine with real-time workouts monitoring using Ultralytics YOLO11. Track and improve your exercise form and performance.
keywords: workouts monitoring, Ultralytics YOLO11, pose estimation, fitness tracking, exercise assessment, real-time feedback, exercise form, performance metrics
---

# Workouts Monitoring using Ultralytics YOLO11

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-monitor-workouts-using-ultralytics-yolo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Workouts Monitoring In Colab"></a>

Monitoring workouts through pose estimation with [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics/) enhances exercise assessment by accurately tracking key body landmarks and joints in real-time. This technology provides instant feedback on exercise form, tracks workout routines, and measures performance metrics, optimizing training sessions for users and trainers alike.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/Ck7DW96dNok"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Monitor Workout Exercises with Ultralytics YOLO | Squats, Leg Extension, Pushups and More
</p>

## Advantages of Workouts Monitoring

- **Optimized Performance:** Tailoring workouts based on monitoring data for better results.
- **Goal Achievement:** Track and adjust fitness goals for measurable progress.
- **Personalization:** Customized workout plans based on individual data for effectiveness.
- **Health Awareness:** Early detection of patterns indicating health issues or over-training.
- **Informed Decisions:** Data-driven decisions for adjusting routines and setting realistic goals.

## Real World Applications

|                                        Workouts Monitoring                                         |                                        Workouts Monitoring                                         |
| :------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------: |
| ![PushUps Counting](https://github.com/ultralytics/docs/releases/download/0/pushups-counting.avif) | ![PullUps Counting](https://github.com/ultralytics/docs/releases/download/0/pullups-counting.avif) |
|                                          PushUps Counting                                          |                                          PullUps Counting                                          |

!!! example "Workouts Monitoring using Ultralytics YOLO"

    === "CLI"

        ```bash
        # Run a workout example
        yolo solutions workout show=True

        # Pass a source video
        yolo solutions workout source="path/to/video.mp4"

        # Use keypoints for pushups
        yolo solutions workout kpts="[6, 8, 10]"
        ```

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "Error reading video file"

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("workouts_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Init AIGym
        gym = solutions.AIGym(
            show=True,  # display the frame
            kpts=[6, 8, 10],  # keypoints for monitoring specific exercise, by default it's for pushup
            model="yolo11n-pose.pt",  # path to the YOLO11 pose estimation model file
            # line_width=2,  # adjust the line width for bounding boxes and text display
        )

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("Video frame is empty or processing is complete.")
                break

            results = gym(im0)

            # print(results)  # access the output

            video_writer.write(results.plot_im)  # write the processed frame.

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # destroy all opened windows
        ```

### KeyPoints Map

![keyPoints Order Ultralytics YOLO11 Pose](https://github.com/ultralytics/docs/releases/download/0/keypoints-order-ultralytics-yolov8-pose.avif)

### `AIGym` Arguments

Here's a table with the `AIGym` arguments:

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "up_angle", "down_angle", "kpts"]) }}

The `AIGym` solution also supports a range of object tracking parameters:

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

Additionally, the following visualization settings can be applied:

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}

## FAQ

### How do I monitor my workouts using Ultralytics YOLO11?

To monitor your workouts using Ultralytics YOLO11, you can utilize the [pose estimation capabilities](https://docs.ultralytics.com/tasks/pose/) to track and analyze key body landmarks and joints in real-time. This allows you to receive instant feedback on your exercise form, count repetitions, and measure performance metrics. You can start by using the provided example code for push-ups, pull-ups, or ab workouts as shown:

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

gym = solutions.AIGym(
    line_width=2,
    show=True,
    kpts=[6, 8, 10],
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or processing is complete.")
        break
    results = gym(im0)

cv2.destroyAllWindows()
```

For further customization and settings, you can refer to the [AIGym](#aigym-arguments) section in the documentation.

### What are the benefits of using Ultralytics YOLO11 for workout monitoring?

Using Ultralytics YOLO11 for workout monitoring provides several key benefits:

- **Optimized Performance:** By tailoring workouts based on monitoring data, you can achieve better results.
- **Goal Achievement:** Easily track and adjust fitness goals for measurable progress.
- **Personalization:** Get customized workout plans based on your individual data for optimal effectiveness.
- **Health Awareness:** Early detection of patterns that indicate potential health issues or over-training.
- **Informed Decisions:** Make data-driven decisions to adjust routines and set realistic goals.

You can watch a [YouTube video demonstration](https://www.youtube.com/watch?v=LGGxqLZtvuw) to see these benefits in action.

### How accurate is Ultralytics YOLO11 in detecting and tracking exercises?

Ultralytics YOLO11 is highly accurate in detecting and tracking exercises due to its state-of-the-art [pose estimation](https://www.ultralytics.com/blog/how-to-use-ultralytics-yolo11-for-pose-estimation) capabilities. It can accurately track key body landmarks and joints, providing real-time feedback on exercise form and performance metrics. The model's pretrained weights and robust architecture ensure high [precision](https://www.ultralytics.com/glossary/precision) and reliability. For real-world examples, check out the [real-world applications](#real-world-applications) section in the documentation, which showcases push-ups and pull-ups counting.

### Can I use Ultralytics YOLO11 for custom workout routines?

Yes, Ultralytics YOLO11 can be adapted for custom workout routines. The `AIGym` class supports different pose types such as `pushup`, `pullup`, and `abworkout`. You can specify keypoints and angles to detect specific exercises. Here is an example setup:

```python
from ultralytics import solutions

gym = solutions.AIGym(
    line_width=2,
    show=True,
    kpts=[6, 8, 10],  # For pushups - can be customized for other exercises
)
```

For more details on setting arguments, refer to the [Arguments `AIGym`](#aigym-arguments) section. This flexibility allows you to monitor various exercises and customize routines based on your [fitness goals](https://www.ultralytics.com/blog/ai-in-our-day-to-day-health-and-fitness).

### How can I save the workout monitoring output using Ultralytics YOLO11?

To save the workout monitoring output, you can modify the code to include a video writer that saves the processed frames. Here's an example:

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

video_writer = cv2.VideoWriter("workouts.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

gym = solutions.AIGym(
    line_width=2,
    show=True,
    kpts=[6, 8, 10],
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or processing is complete.")
        break
    results = gym(im0)
    video_writer.write(results.plot_im)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
```

This setup writes the monitored video to an output file, allowing you to review your workout performance later or share it with trainers for additional feedback.
>>>>>>> 0e451281e978a7c5f581cc5d9bb3d0f62e0bc632
