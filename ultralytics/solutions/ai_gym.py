<<<<<<< HEAD
# Ultralytics YOLO 🚀, AGPL-3.0 license

import cv2

from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator


class AIGym:
    """A class to manage the gym steps of people in a real-time video stream based on their poses."""

    def __init__(self):
        """Initializes the AIGym with default values for Visual and Image parameters."""

        # Image and line thickness
        self.im0 = None
        self.tf = None

        # Keypoints and count information
        self.keypoints = None
        self.poseup_angle = None
        self.posedown_angle = None
        self.threshold = 0.001

        # Store stage, count and angle information
        self.angle = None
        self.count = None
        self.stage = None
        self.pose_type = 'pushup'
        self.kpts_to_check = None

        # Visual Information
        self.view_img = False
        self.annotator = None

        # Check if environment support imshow
        self.env_check = check_imshow(warn=True)

    def set_args(self,
                 kpts_to_check,
                 line_thickness=2,
                 view_img=False,
                 pose_up_angle=145.0,
                 pose_down_angle=90.0,
                 pose_type='pullup'):
        """
        Configures the AIGym line_thickness, save image and view image parameters
        Args:
            kpts_to_check (list): 3 keypoints for counting
            line_thickness (int): Line thickness for bounding boxes.
            view_img (bool): display the im0
            pose_up_angle (float): Angle to set pose position up
            pose_down_angle (float): Angle to set pose position down
            pose_type: "pushup", "pullup" or "abworkout"
        """
        self.kpts_to_check = kpts_to_check
        self.tf = line_thickness
        self.view_img = view_img
        self.poseup_angle = pose_up_angle
        self.posedown_angle = pose_down_angle
        self.pose_type = pose_type

    def start_counting(self, im0, results, frame_count):
        """
        function used to count the gym steps
        Args:
            im0 (ndarray): Current frame from the video stream.
            results: Pose estimation data
            frame_count: store current frame count
        """
        self.im0 = im0
        if frame_count == 1:
            self.count = [0] * len(results[0])
            self.angle = [0] * len(results[0])
            self.stage = ['-' for _ in results[0]]
        self.keypoints = results[0].keypoints.data
        self.annotator = Annotator(im0, line_width=2)

        for ind, k in enumerate(reversed(self.keypoints)):
            if self.pose_type == 'pushup' or self.pose_type == 'pullup':
                self.angle[ind] = self.annotator.estimate_pose_angle(k[int(self.kpts_to_check[0])].cpu(),
                                                                     k[int(self.kpts_to_check[1])].cpu(),
                                                                     k[int(self.kpts_to_check[2])].cpu())
                self.im0 = self.annotator.draw_specific_points(k, self.kpts_to_check, shape=(640, 640), radius=10)

            if self.pose_type == 'abworkout':
                self.angle[ind] = self.annotator.estimate_pose_angle(k[int(self.kpts_to_check[0])].cpu(),
                                                                     k[int(self.kpts_to_check[1])].cpu(),
                                                                     k[int(self.kpts_to_check[2])].cpu())
                self.im0 = self.annotator.draw_specific_points(k, self.kpts_to_check, shape=(640, 640), radius=10)
                if self.angle[ind] > self.poseup_angle:
                    self.stage[ind] = 'down'
                if self.angle[ind] < self.posedown_angle and self.stage[ind] == 'down':
                    self.stage[ind] = 'up'
                    self.count[ind] += 1
                self.annotator.plot_angle_and_count_and_stage(angle_text=self.angle[ind],
                                                              count_text=self.count[ind],
                                                              stage_text=self.stage[ind],
                                                              center_kpt=k[int(self.kpts_to_check[1])],
                                                              line_thickness=self.tf)

            if self.pose_type == 'pushup':
                if self.angle[ind] > self.poseup_angle:
                    self.stage[ind] = 'up'
                if self.angle[ind] < self.posedown_angle and self.stage[ind] == 'up':
                    self.stage[ind] = 'down'
                    self.count[ind] += 1
                self.annotator.plot_angle_and_count_and_stage(angle_text=self.angle[ind],
                                                              count_text=self.count[ind],
                                                              stage_text=self.stage[ind],
                                                              center_kpt=k[int(self.kpts_to_check[1])],
                                                              line_thickness=self.tf)
            if self.pose_type == 'pullup':
                if self.angle[ind] > self.poseup_angle:
                    self.stage[ind] = 'down'
                if self.angle[ind] < self.posedown_angle and self.stage[ind] == 'down':
                    self.stage[ind] = 'up'
                    self.count[ind] += 1
                self.annotator.plot_angle_and_count_and_stage(angle_text=self.angle[ind],
                                                              count_text=self.count[ind],
                                                              stage_text=self.stage[ind],
                                                              center_kpt=k[int(self.kpts_to_check[1])],
                                                              line_thickness=self.tf)

            self.annotator.kpts(k, shape=(640, 640), radius=1, kpt_line=True)

        if self.env_check and self.view_img:
            cv2.imshow('Ultralytics YOLOv8 AI GYM', self.im0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

        return self.im0


if __name__ == '__main__':
    AIGym()
=======
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from collections import defaultdict
from typing import Any

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults


class AIGym(BaseSolution):
    """
    A class to manage gym steps of people in a real-time video stream based on their poses.

    This class extends BaseSolution to monitor workouts using YOLO pose estimation models. It tracks and counts
    repetitions of exercises based on predefined angle thresholds for up and down positions.

    Attributes:
        states (Dict[float, int, str]): Stores per-track angle, count, and stage for workout monitoring.
        up_angle (float): Angle threshold for considering the 'up' position of an exercise.
        down_angle (float): Angle threshold for considering the 'down' position of an exercise.
        kpts (List[int]): Indices of keypoints used for angle calculation.

    Methods:
        process: Process a frame to detect poses, calculate angles, and count repetitions.

    Examples:
        >>> gym = AIGym(model="yolo11n-pose.pt")
        >>> image = cv2.imread("gym_scene.jpg")
        >>> results = gym.process(image)
        >>> processed_image = results.plot_im
        >>> cv2.imshow("Processed Image", processed_image)
        >>> cv2.waitKey(0)
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize AIGym for workout monitoring using pose estimation and predefined angles.

        Args:
            **kwargs (Any): Keyword arguments passed to the parent class constructor.
                model (str): Model name or path, defaults to "yolo11n-pose.pt".
        """
        kwargs["model"] = kwargs.get("model", "yolo11n-pose.pt")
        super().__init__(**kwargs)
        self.states = defaultdict(lambda: {"angle": 0, "count": 0, "stage": "-"})  # Dict for count, angle and stage

        # Extract details from CFG single time for usage later
        self.up_angle = float(self.CFG["up_angle"])  # Pose up predefined angle to consider up pose
        self.down_angle = float(self.CFG["down_angle"])  # Pose down predefined angle to consider down pose
        self.kpts = self.CFG["kpts"]  # User selected kpts of workouts storage for further usage

    def process(self, im0) -> SolutionResults:
        """
        Monitor workouts using Ultralytics YOLO Pose Model.

        This function processes an input image to track and analyze human poses for workout monitoring. It uses
        the YOLO Pose model to detect keypoints, estimate angles, and count repetitions based on predefined
        angle thresholds.

        Args:
            im0 (np.ndarray): Input image for processing.

        Returns:
            (SolutionResults): Contains processed image `plot_im`,
                'workout_count' (list of completed reps),
                'workout_stage' (list of current stages),
                'workout_angle' (list of angles), and
                'total_tracks' (total number of tracked individuals).

        Examples:
            >>> gym = AIGym()
            >>> image = cv2.imread("workout.jpg")
            >>> results = gym.process(image)
            >>> processed_image = results.plot_im
        """
        annotator = SolutionAnnotator(im0, line_width=self.line_width)  # Initialize annotator

        self.extract_tracks(im0)  # Extract tracks (bounding boxes, classes, and masks)

        if len(self.boxes):
            kpt_data = self.tracks.keypoints.data

            for i, k in enumerate(kpt_data):
                state = self.states[self.track_ids[i]]  # get state details
                # Get keypoints and estimate the angle
                state["angle"] = annotator.estimate_pose_angle(*[k[int(idx)] for idx in self.kpts])
                annotator.draw_specific_kpts(k, self.kpts, radius=self.line_width * 3)

                # Determine stage and count logic based on angle thresholds
                if state["angle"] < self.down_angle:
                    if state["stage"] == "up":
                        state["count"] += 1
                    state["stage"] = "down"
                elif state["angle"] > self.up_angle:
                    state["stage"] = "up"

                # Display angle, count, and stage text
                if self.show_labels:
                    annotator.plot_angle_and_count_and_stage(
                        angle_text=state["angle"],  # angle text for display
                        count_text=state["count"],  # count text for workouts
                        stage_text=state["stage"],  # stage position text
                        center_kpt=k[int(self.kpts[1])],  # center keypoint for display
                    )
        plot_im = annotator.result()
        self.display_output(plot_im)  # Display output image, if environment support display

        # Return SolutionResults
        return SolutionResults(
            plot_im=plot_im,
            workout_count=[v["count"] for v in self.states.values()],
            workout_stage=[v["stage"] for v in self.states.values()],
            workout_angle=[v["angle"] for v in self.states.values()],
            total_tracks=len(self.track_ids),
        )
>>>>>>> 0e451281e978a7c5f581cc5d9bb3d0f62e0bc632
