<<<<<<< HEAD
# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import defaultdict

import cv2
import numpy as np

from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator

check_requirements('shapely>=2.0.0')

from shapely.geometry import Polygon
from shapely.geometry.point import Point


class Heatmap:
    """A class to draw heatmaps in real-time video stream based on their tracks."""

    def __init__(self):
        """Initializes the heatmap class with default values for Visual, Image, track, count and heatmap parameters."""

        # Visual information
        self.annotator = None
        self.view_img = False

        # Image information
        self.imw = None
        self.imh = None
        self.im0 = None

        # Heatmap colormap and heatmap np array
        self.colormap = None
        self.heatmap = None
        self.heatmap_alpha = 0.5

        # Predict/track information
        self.boxes = None
        self.track_ids = None
        self.clss = None
        self.track_history = None

        # Counting info
        self.count_reg_pts = None
        self.count_region = None
        self.in_counts = 0
        self.out_counts = 0
        self.count_list = []
        self.count_txt_thickness = 0
        self.count_reg_color = (0, 255, 0)
        self.region_thickness = 5

        # Check if environment support imshow
        self.env_check = check_imshow(warn=True)

    def set_args(self,
                 imw,
                 imh,
                 colormap=cv2.COLORMAP_JET,
                 heatmap_alpha=0.5,
                 view_img=False,
                 count_reg_pts=None,
                 count_txt_thickness=2,
                 count_reg_color=(255, 0, 255),
                 region_thickness=5):
        """
        Configures the heatmap colormap, width, height and display parameters.

        Args:
            colormap (cv2.COLORMAP): The colormap to be set.
            imw (int): The width of the frame.
            imh (int): The height of the frame.
            heatmap_alpha (float): alpha value for heatmap display
            view_img (bool): Flag indicating frame display
            count_reg_pts (list): Object counting region points
            count_txt_thickness (int): Text thickness for object counting display
            count_reg_color (RGB color): Color of object counting region
            region_thickness (int): Object counting Region thickness
        """
        self.imw = imw
        self.imh = imh
        self.colormap = colormap
        self.heatmap_alpha = heatmap_alpha
        self.view_img = view_img

        self.heatmap = np.zeros((int(self.imw), int(self.imh)), dtype=np.float32)  # Heatmap new frame

        if count_reg_pts is not None:
            self.track_history = defaultdict(list)
            self.count_reg_pts = count_reg_pts
            self.count_region = Polygon(self.count_reg_pts)

        self.count_txt_thickness = count_txt_thickness  # Counting text thickness
        self.count_reg_color = count_reg_color
        self.region_thickness = region_thickness

    def extract_results(self, tracks):
        """
        Extracts results from the provided data.

        Args:
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.boxes = tracks[0].boxes.xyxy.cpu()
        self.clss = tracks[0].boxes.cls.cpu().tolist()
        self.track_ids = tracks[0].boxes.id.int().cpu().tolist()

    def generate_heatmap(self, im0, tracks):
        """
        Generate heatmap based on tracking data.

        Args:
            im0 (nd array): Image
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.im0 = im0
        if tracks[0].boxes.id is None:
            return self.im0

        self.extract_results(tracks)
        self.annotator = Annotator(self.im0, self.count_txt_thickness, None)

        if self.count_reg_pts is not None:
            # Draw counting region
            self.annotator.draw_region(reg_pts=self.count_reg_pts,
                                       color=self.count_reg_color,
                                       thickness=self.region_thickness)

            for box, cls, track_id in zip(self.boxes, self.clss, self.track_ids):
                self.heatmap[int(box[1]):int(box[3]), int(box[0]):int(box[2])] += 1

                # Store tracking hist
                track_line = self.track_history[track_id]
                track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
                if len(track_line) > 30:
                    track_line.pop(0)

                # Count objects
                if self.count_region.contains(Point(track_line[-1])):
                    if track_id not in self.count_list:
                        self.count_list.append(track_id)
                        if box[0] < self.count_region.centroid.x:
                            self.out_counts += 1
                        else:
                            self.in_counts += 1
        else:
            for box, cls in zip(self.boxes, self.clss):
                self.heatmap[int(box[1]):int(box[3]), int(box[0]):int(box[2])] += 1

        # Normalize, apply colormap to heatmap and combine with original image
        heatmap_normalized = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), self.colormap)

        if self.count_reg_pts is not None:
            incount_label = 'InCount : ' + f'{self.in_counts}'
            outcount_label = 'OutCount : ' + f'{self.out_counts}'
            self.annotator.count_labels(in_count=incount_label, out_count=outcount_label)

        im0_with_heatmap = cv2.addWeighted(self.im0, 1 - self.heatmap_alpha, heatmap_colored, self.heatmap_alpha, 0)

        if self.env_check and self.view_img:
            self.display_frames(im0_with_heatmap)

        return im0_with_heatmap

    @staticmethod
    def display_frames(im0_with_heatmap):
        """
        Display heatmap.

        Args:
            im0_with_heatmap (nd array): Original Image with heatmap
        """
        cv2.imshow('Ultralytics Heatmap', im0_with_heatmap)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return


if __name__ == '__main__':
    Heatmap()
=======
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from typing import Any, List

import cv2
import numpy as np

from ultralytics.solutions.object_counter import ObjectCounter
from ultralytics.solutions.solutions import SolutionAnnotator, SolutionResults


class Heatmap(ObjectCounter):
    """
    A class to draw heatmaps in real-time video streams based on object tracks.

    This class extends the ObjectCounter class to generate and visualize heatmaps of object movements in video
    streams. It uses tracked object positions to create a cumulative heatmap effect over time.

    Attributes:
        initialized (bool): Flag indicating whether the heatmap has been initialized.
        colormap (int): OpenCV colormap used for heatmap visualization.
        heatmap (np.ndarray): Array storing the cumulative heatmap data.
        annotator (SolutionAnnotator): Object for drawing annotations on the image.

    Methods:
        heatmap_effect: Calculate and update the heatmap effect for a given bounding box.
        process: Generate and apply the heatmap effect to each frame.

    Examples:
        >>> from ultralytics.solutions import Heatmap
        >>> heatmap = Heatmap(model="yolo11n.pt", colormap=cv2.COLORMAP_JET)
        >>> frame = cv2.imread("frame.jpg")
        >>> processed_frame = heatmap.process(frame)
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the Heatmap class for real-time video stream heatmap generation based on object tracks.

        Args:
            **kwargs (Any): Keyword arguments passed to the parent ObjectCounter class.
        """
        super().__init__(**kwargs)

        self.initialized = False  # Flag for heatmap initialization
        if self.region is not None:  # Check if user provided the region coordinates
            self.initialize_region()

        # Store colormap
        self.colormap = self.CFG["colormap"]
        self.heatmap = None

    def heatmap_effect(self, box: List[float]) -> None:
        """
        Efficiently calculate heatmap area and effect location for applying colormap.

        Args:
            box (List[float]): Bounding box coordinates [x0, y0, x1, y1].
        """
        x0, y0, x1, y1 = map(int, box)
        radius_squared = (min(x1 - x0, y1 - y0) // 2) ** 2

        # Create a meshgrid with region of interest (ROI) for vectorized distance calculations
        xv, yv = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))

        # Calculate squared distances from the center
        dist_squared = (xv - ((x0 + x1) // 2)) ** 2 + (yv - ((y0 + y1) // 2)) ** 2

        # Create a mask of points within the radius
        within_radius = dist_squared <= radius_squared

        # Update only the values within the bounding box in a single vectorized operation
        self.heatmap[y0:y1, x0:x1][within_radius] += 2

    def process(self, im0: np.ndarray) -> SolutionResults:
        """
        Generate heatmap for each frame using Ultralytics tracking.

        Args:
            im0 (np.ndarray): Input image array for processing.

        Returns:
            (SolutionResults): Contains processed image `plot_im`,
                'in_count' (int, count of objects entering the region),
                'out_count' (int, count of objects exiting the region),
                'classwise_count' (dict, per-class object count), and
                'total_tracks' (int, total number of tracked objects).
        """
        if not self.initialized:
            self.heatmap = np.zeros_like(im0, dtype=np.float32) * 0.99
            self.initialized = True  # Initialize heatmap only once

        self.extract_tracks(im0)  # Extract tracks
        self.annotator = SolutionAnnotator(im0, line_width=self.line_width)  # Initialize annotator

        # Iterate over bounding boxes, track ids and classes index
        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            # Apply heatmap effect for the bounding box
            self.heatmap_effect(box)

            if self.region is not None:
                self.annotator.draw_region(reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2)
                self.store_tracking_history(track_id, box)  # Store track history
                # Get previous position if available
                prev_position = None
                if len(self.track_history[track_id]) > 1:
                    prev_position = self.track_history[track_id][-2]
                self.count_objects(self.track_history[track_id][-1], track_id, prev_position, cls)  # object counting

        plot_im = self.annotator.result()
        if self.region is not None:
            self.display_counts(plot_im)  # Display the counts on the frame

        # Normalize, apply colormap to heatmap and combine with original image
        if self.track_data.is_track:
            normalized_heatmap = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            colored_heatmap = cv2.applyColorMap(normalized_heatmap, self.colormap)
            plot_im = cv2.addWeighted(plot_im, 0.5, colored_heatmap, 0.5, 0)

        self.display_output(plot_im)  # Display output with base class function

        # Return SolutionResults
        return SolutionResults(
            plot_im=plot_im,
            in_count=self.in_count,
            out_count=self.out_count,
            classwise_count=dict(self.classwise_count),
            total_tracks=len(self.track_ids),
        )
>>>>>>> 0e451281e978a7c5f581cc5d9bb3d0f62e0bc632
