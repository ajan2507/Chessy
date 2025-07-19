<<<<<<< HEAD
# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import defaultdict

import cv2

from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator, colors

check_requirements('shapely>=2.0.0')

from shapely.geometry import Polygon
from shapely.geometry.point import Point


class ObjectCounter:
    """A class to manage the counting of objects in a real-time video stream based on their tracks."""

    def __init__(self):
        """Initializes the Counter with default values for various tracking and counting parameters."""

        # Mouse events
        self.is_drawing = False
        self.selected_point = None

        # Region Information
        self.reg_pts = None
        self.counting_region = None
        self.region_color = (255, 255, 255)

        # Image and annotation Information
        self.im0 = None
        self.tf = None
        self.view_img = False

        self.names = None  # Classes names
        self.annotator = None  # Annotator

        # Object counting Information
        self.in_counts = 0
        self.out_counts = 0
        self.counting_list = []

        # Tracks info
        self.track_history = defaultdict(list)
        self.track_thickness = 2
        self.draw_tracks = False

        # Check if environment support imshow
        self.env_check = check_imshow(warn=True)

    def set_args(self,
                 classes_names,
                 reg_pts,
                 region_color=None,
                 line_thickness=2,
                 track_thickness=2,
                 view_img=False,
                 draw_tracks=False):
        """
        Configures the Counter's image, bounding box line thickness, and counting region points.

        Args:
            line_thickness (int): Line thickness for bounding boxes.
            view_img (bool): Flag to control whether to display the video stream.
            reg_pts (list): Initial list of points defining the counting region.
            classes_names (dict): Classes names
            region_color (tuple): color for region line
            track_thickness (int): Track thickness
            draw_tracks (Bool): draw tracks
        """
        self.tf = line_thickness
        self.view_img = view_img
        self.track_thickness = track_thickness
        self.draw_tracks = draw_tracks
        self.reg_pts = reg_pts
        self.counting_region = Polygon(self.reg_pts)
        self.names = classes_names
        self.region_color = region_color if region_color else self.region_color

    def mouse_event_for_region(self, event, x, y, flags, params):
        """
        This function is designed to move region with mouse events in a real-time video stream.

        Args:
            event (int): The type of mouse event (e.g., cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, etc.).
            x (int): The x-coordinate of the mouse pointer.
            y (int): The y-coordinate of the mouse pointer.
            flags (int): Any flags associated with the event (e.g., cv2.EVENT_FLAG_CTRLKEY,
                cv2.EVENT_FLAG_SHIFTKEY, etc.).
            params (dict): Additional parameters you may want to pass to the function.
        """
        # global is_drawing, selected_point
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, point in enumerate(self.reg_pts):
                if isinstance(point, (tuple, list)) and len(point) >= 2:
                    if abs(x - point[0]) < 10 and abs(y - point[1]) < 10:
                        self.selected_point = i
                        self.is_drawing = True
                        break

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing and self.selected_point is not None:
                self.reg_pts[self.selected_point] = (x, y)
                self.counting_region = Polygon(self.reg_pts)

        elif event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False
            self.selected_point = None

    def extract_and_process_tracks(self, tracks):
        boxes = tracks[0].boxes.xyxy.cpu()
        clss = tracks[0].boxes.cls.cpu().tolist()
        track_ids = tracks[0].boxes.id.int().cpu().tolist()

        self.annotator = Annotator(self.im0, self.tf, self.names)
        self.annotator.draw_region(reg_pts=self.reg_pts, color=(0, 255, 0))

        for box, track_id, cls in zip(boxes, track_ids, clss):
            self.annotator.box_label(box, label=self.names[cls], color=colors(int(cls), True))  # Draw bounding box

            # Draw Tracks
            track_line = self.track_history[track_id]
            track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
            if len(track_line) > 30:
                track_line.pop(0)

            if self.draw_tracks:
                self.annotator.draw_centroid_and_tracks(track_line,
                                                        color=(0, 255, 0),
                                                        track_thickness=self.track_thickness)

            # Count objects
            if self.counting_region.contains(Point(track_line[-1])):
                if track_id not in self.counting_list:
                    self.counting_list.append(track_id)
                    if box[0] < self.counting_region.centroid.x:
                        self.out_counts += 1
                    else:
                        self.in_counts += 1

        if self.env_check and self.view_img:
            incount_label = 'InCount : ' + f'{self.in_counts}'
            outcount_label = 'OutCount : ' + f'{self.out_counts}'
            self.annotator.count_labels(in_count=incount_label, out_count=outcount_label)
            cv2.namedWindow('Ultralytics YOLOv8 Object Counter')
            cv2.setMouseCallback('Ultralytics YOLOv8 Object Counter', self.mouse_event_for_region,
                                 {'region_points': self.reg_pts})
            cv2.imshow('Ultralytics YOLOv8 Object Counter', self.im0)
            # Break Window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

    def start_counting(self, im0, tracks):
        """
        Main function to start the object counting process.

        Args:
            im0 (ndarray): Current frame from the video stream.
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.im0 = im0  # store image
        if tracks[0].boxes.id is None:
            return
        self.extract_and_process_tracks(tracks)
        return self.im0


if __name__ == '__main__':
    ObjectCounter()
=======
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from collections import defaultdict
from typing import Any, Optional, Tuple

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class ObjectCounter(BaseSolution):
    """
    A class to manage the counting of objects in a real-time video stream based on their tracks.

    This class extends the BaseSolution class and provides functionality for counting objects moving in and out of a
    specified region in a video stream. It supports both polygonal and linear regions for counting.

    Attributes:
        in_count (int): Counter for objects moving inward.
        out_count (int): Counter for objects moving outward.
        counted_ids (List[int]): List of IDs of objects that have been counted.
        classwise_counts (Dict[str, Dict[str, int]]): Dictionary for counts, categorized by object class.
        region_initialized (bool): Flag indicating whether the counting region has been initialized.
        show_in (bool): Flag to control display of inward count.
        show_out (bool): Flag to control display of outward count.
        margin (int): Margin for background rectangle size to display counts properly.

    Methods:
        count_objects: Count objects within a polygonal or linear region based on their tracks.
        display_counts: Display object counts on the frame.
        process: Process input data and update counts.

    Examples:
        >>> counter = ObjectCounter()
        >>> frame = cv2.imread("frame.jpg")
        >>> results = counter.process(frame)
        >>> print(f"Inward count: {counter.in_count}, Outward count: {counter.out_count}")
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the ObjectCounter class for real-time object counting in video streams."""
        super().__init__(**kwargs)

        self.in_count = 0  # Counter for objects moving inward
        self.out_count = 0  # Counter for objects moving outward
        self.counted_ids = []  # List of IDs of objects that have been counted
        self.classwise_count = defaultdict(lambda: {"IN": 0, "OUT": 0})  # Dictionary for counts, categorized by class
        self.region_initialized = False  # Flag indicating whether the region has been initialized

        self.show_in = self.CFG["show_in"]
        self.show_out = self.CFG["show_out"]
        self.margin = self.line_width * 2  # Scales the background rectangle size to display counts properly

    def count_objects(
        self,
        current_centroid: Tuple[float, float],
        track_id: int,
        prev_position: Optional[Tuple[float, float]],
        cls: int,
    ) -> None:
        """
        Count objects within a polygonal or linear region based on their tracks.

        Args:
            current_centroid (Tuple[float, float]): Current centroid coordinates (x, y) in the current frame.
            track_id (int): Unique identifier for the tracked object.
            prev_position (Tuple[float, float], optional): Last frame position coordinates (x, y) of the track.
            cls (int): Class index for classwise count updates.

        Examples:
            >>> counter = ObjectCounter()
            >>> track_line = {1: [100, 200], 2: [110, 210], 3: [120, 220]}
            >>> box = [130, 230, 150, 250]
            >>> track_id_num = 1
            >>> previous_position = (120, 220)
            >>> class_to_count = 0  # In COCO model, class 0 = person
            >>> counter.count_objects((140, 240), track_id_num, previous_position, class_to_count)
        """
        if prev_position is None or track_id in self.counted_ids:
            return

        if len(self.region) == 2:  # Linear region (defined as a line segment)
            if self.r_s.intersects(self.LineString([prev_position, current_centroid])):
                # Determine orientation of the region (vertical or horizontal)
                if abs(self.region[0][0] - self.region[1][0]) < abs(self.region[0][1] - self.region[1][1]):
                    # Vertical region: Compare x-coordinates to determine direction
                    if current_centroid[0] > prev_position[0]:  # Moving right
                        self.in_count += 1
                        self.classwise_count[self.names[cls]]["IN"] += 1
                    else:  # Moving left
                        self.out_count += 1
                        self.classwise_count[self.names[cls]]["OUT"] += 1
                # Horizontal region: Compare y-coordinates to determine direction
                elif current_centroid[1] > prev_position[1]:  # Moving downward
                    self.in_count += 1
                    self.classwise_count[self.names[cls]]["IN"] += 1
                else:  # Moving upward
                    self.out_count += 1
                    self.classwise_count[self.names[cls]]["OUT"] += 1
                self.counted_ids.append(track_id)

        elif len(self.region) > 2:  # Polygonal region
            if self.r_s.contains(self.Point(current_centroid)):
                # Determine motion direction for vertical or horizontal polygons
                region_width = max(p[0] for p in self.region) - min(p[0] for p in self.region)
                region_height = max(p[1] for p in self.region) - min(p[1] for p in self.region)

                if (
                    region_width < region_height
                    and current_centroid[0] > prev_position[0]
                    or region_width >= region_height
                    and current_centroid[1] > prev_position[1]
                ):  # Moving right or downward
                    self.in_count += 1
                    self.classwise_count[self.names[cls]]["IN"] += 1
                else:  # Moving left or upward
                    self.out_count += 1
                    self.classwise_count[self.names[cls]]["OUT"] += 1
                self.counted_ids.append(track_id)

    def display_counts(self, plot_im) -> None:
        """
        Display object counts on the input image or frame.

        Args:
            plot_im (np.ndarray): The image or frame to display counts on.

        Examples:
            >>> counter = ObjectCounter()
            >>> frame = cv2.imread("image.jpg")
            >>> counter.display_counts(frame)
        """
        labels_dict = {
            str.capitalize(key): f"{'IN ' + str(value['IN']) if self.show_in else ''} "
            f"{'OUT ' + str(value['OUT']) if self.show_out else ''}".strip()
            for key, value in self.classwise_count.items()
            if value["IN"] != 0 or value["OUT"] != 0 and (self.show_in or self.show_out)
        }
        if labels_dict:
            self.annotator.display_analytics(plot_im, labels_dict, (104, 31, 17), (255, 255, 255), self.margin)

    def process(self, im0) -> SolutionResults:
        """
        Process input data (frames or object tracks) and update object counts.

        This method initializes the counting region, extracts tracks, draws bounding boxes and regions, updates
        object counts, and displays the results on the input image.

        Args:
            im0 (np.ndarray): The input image or frame to be processed.

        Returns:
            (SolutionResults): Contains processed image `im0`, 'in_count' (int, count of objects entering the region),
                'out_count' (int, count of objects exiting the region), 'classwise_count' (dict, per-class object count),
                and 'total_tracks' (int, total number of tracked objects).

        Examples:
            >>> counter = ObjectCounter()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> results = counter.process(frame)
        """
        if not self.region_initialized:
            self.initialize_region()
            self.region_initialized = True

        self.extract_tracks(im0)  # Extract tracks
        self.annotator = SolutionAnnotator(im0, line_width=self.line_width)  # Initialize annotator

        self.annotator.draw_region(
            reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2
        )  # Draw region

        # Iterate over bounding boxes, track ids and classes index
        for box, track_id, cls, conf in zip(self.boxes, self.track_ids, self.clss, self.confs):
            # Draw bounding box and counting region
            self.annotator.box_label(box, label=self.adjust_box_label(cls, conf, track_id), color=colors(cls, True))
            self.store_tracking_history(track_id, box)  # Store track history

            # Store previous position of track for object counting
            prev_position = None
            if len(self.track_history[track_id]) > 1:
                prev_position = self.track_history[track_id][-2]
            self.count_objects(self.track_history[track_id][-1], track_id, prev_position, cls)  # object counting

        plot_im = self.annotator.result()
        self.display_counts(plot_im)  # Display the counts on the frame
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
