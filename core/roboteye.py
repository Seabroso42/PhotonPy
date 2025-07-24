import os
import cv2 as cv
import numpy as np
import requests
import random
import matplotlib.pyplot as plt
from typing import Optional, Union, Tuple, Literal

class RobotEye:
    rootpath = os.getcwd()

    def __init__(self, outpath: Optional[str], connection: Union[None, int, str, cv.Mat] = None):
        """Initialize RobotEye with output path and image/video source."""
        self._connection = connection
        self._outpath = outpath if outpath else os.path.join(self.rootpath, 'output')
        self._source_type = self._determine_source_type(connection)
        self._capture = None
        self._capture_device = None
        self._current_frame = None

        self._validate_output_path(self._outpath)
        self._setup_source(connection)

    # ==================== CORE METHODS ====================
    def _validate_output_path(self, path: str) -> None:
        """Create output directory if it doesn't exist."""
        os.makedirs(path, exist_ok=True)

    def _determine_source_type(self, source) -> Literal['camera', 'file', 'direct']:
        """Determine the type of image source."""
        if source is None:
            return 'camera'
        elif isinstance(source, int):
            return 'camera'
        elif isinstance(source, str):
            return 'file'
        elif isinstance(source, cv.Mat):
            return 'direct'
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

    def _setup_source(self, source) -> None:
        """Configure the image source based on type."""
        if self._source_type == 'camera':
            self._init_camera(source if isinstance(source, int) else 0)
        elif self._source_type == 'file':
            self._load_image_file(source)
        elif self._source_type == 'direct':
            self._current_frame = source

    def _init_camera(self, camera_index: int) -> None:
        """Initialize camera capture device."""
        self._capture_device = cv.VideoCapture(camera_index)
        if not self._capture_device.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_index}")
        self._update_frame()

    def _load_image_file(self, file_path: str) -> None:
        """Load image from file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        self._current_frame = cv.imread(file_path)
        if self._current_frame is None:
            raise ValueError(f"Failed to read image: {file_path}")

    def _update_frame(self) -> None:
        """Update current frame (for camera sources)."""
        if self._source_type == 'camera':
            ret, frame = self._capture_device.read()
            if ret:
                self._current_frame = frame
            else:
                raise RuntimeError("Failed to capture frame from camera")

    # ==================== IMAGE ACQUISITION ====================
    @staticmethod
    def read_image(image_name: str) -> np.ndarray:
        """Read image from default computer vision images directory."""
        img_path = os.path.join(RobotEye.rootpath, f'computervision/images/{image_name}')
        image = cv.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found or invalid: {img_path}")
        return image

    @staticmethod
    def read_n_show(image_name: str) -> None:
        """Read and display image using matplotlib."""
        image = RobotEye.read_image(image_name)
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        plt.figure()
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.show()

    # ==================== API FETCH METHODS ====================
    @staticmethod
    def pokefetch(name: Optional[str] = None) -> np.ndarray:
        """Fetch Pokémon image from PokeAPI."""
        try:
            url = (f"https://pokeapi.co/api/v2/pokemon/{name.lower()}"
                  if name else
                  f"https://pokeapi.co/api/v2/pokemon/{random.randint(1, 1025)}")
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            img_url = (data.get("sprites", {}).get("other", {}).get("official-artwork", {}).get("front_default"))
            if not img_url:
                raise ValueError("No image available for this Pokémon")

            img_resp = requests.get(img_url, timeout=10)
            img_resp.raise_for_status()
            img_array = np.asarray(bytearray(img_resp.content), dtype=np.uint8)
            return cv.imdecode(img_array, cv.IMREAD_COLOR)
        except Exception as e:
            raise ValueError(f"Failed to fetch Pokémon: {str(e)}")

    @staticmethod
    def magic_gather(name: Optional[str] = None, image_type: str = "art_crop") -> np.ndarray:
        """Fetch Magic: The Gathering card from Scryfall API."""
        try:
            url = (f"https://api.scryfall.com/cards/named?fuzzy={name}"
                  if name else
                  "https://api.scryfall.com/cards/random")
            resp = requests.get(url, headers={"User-Agent": "PhotonApp/1.0"}, timeout=10)
            resp.raise_for_status()
            card = resp.json()

            img_url = (card.get("image_uris", {}).get(image_type)
                      or card.get("card_faces", [{}])[0].get("image_uris", {}).get(image_type))
            if not img_url:
                raise ValueError(f"Image type '{image_type}' not available")

            img_resp = requests.get(img_url, timeout=10)
            img_resp.raise_for_status()
            img_array = np.asarray(bytearray(img_resp.content), dtype=np.uint8)
            return cv.imdecode(img_array, cv.IMREAD_COLOR)
        except Exception as e:
            raise ValueError(f"Failed to fetch Magic card: {str(e)}")

    # ==================== REGION OF INTEREST ====================
    def read_roi(self, init_point: Tuple[int, int], end_point: Tuple[int, int]) -> np.ndarray:
        """Extract Region of Interest from current frame."""
        if self._current_frame is None:
            raise ValueError("No image available for ROI extraction")

        x1, y1 = init_point
        x2, y2 = end_point
        height, width = self._current_frame.shape[:2]

        # Validate coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)

        if x1 >= x2 or y1 >= y2:
            raise ValueError("Invalid ROI coordinates")

        region = self._current_frame[y1:y2, x1:x2]
        return cv.cvtColor(region, cv.COLOR_BGR2RGB)

    # ==================== VIDEO METHODS ====================
    def video_webcam(self) -> None:
        """Display live webcam feed."""
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError("Failed to capture frame")
                cv.imshow('Webcam Feed', frame)
                if cv.waitKey(1) == ord('q'):
                    break
        finally:
            cap.release()
            cv.destroyAllWindows()

    def webcam_to_file(self, filename: str = 'webcam.avi') -> None:
        """Record webcam video to file."""
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam")

        fourcc = cv.VideoWriter_fourcc(*'XVID')
        out_path = os.path.join(self._outpath, filename)
        out = cv.VideoWriter(out_path, fourcc, 20.0, (640, 480))

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError("Failed to capture frame")

                out.write(frame)
                cv.imshow("Recording... (Press Q to stop)", frame)
                if cv.waitKey(1) == ord('q'):
                    break
        finally:
            cap.release()
            out.release()
            cv.destroyAllWindows()

    def video_from_file(self, vid_name: str) -> None:
        """Play video from file."""
        vid_path = os.path.join(self._outpath, vid_name)
        cap = cv.VideoCapture(vid_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {vid_path}")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                cv.imshow(vid_name, frame)
                if cv.waitKey(int(1000/30)) == ord('q'):  # 30 FPS
                    break
        finally:
            cap.release()
            cv.destroyAllWindows()

    # ==================== UTILITY METHODS ====================
    def return_capture(self, pattern: Literal['bgr', 'rgb', 'hsv', 'gray']) -> np.ndarray:
        """Return capture in specified color pattern."""
        if self._current_frame is None:
            raise ValueError("No frame available for conversion")

        match pattern:
            case 'bgr':
                return self._current_frame.copy()
            case 'rgb':
                return cv.cvtColor(self._current_frame, cv.COLOR_BGR2RGB)
            case 'hsv':
                return cv.cvtColor(self._current_frame, cv.COLOR_BGR2HSV)
            case 'gray':
                return cv.cvtColor(self._current_frame, cv.COLOR_BGR2GRAY)
            case _:
                raise ValueError(f"Unsupported color pattern: {pattern}")

    def save_image(self, filename: str) -> None:
        """Save current frame to file."""
        if self._current_frame is None:
            raise ValueError("No image available to save")

        out_path = os.path.join(self._outpath, filename)
        if not cv.imwrite(out_path, self._current_frame):
            raise RuntimeError(f"Failed to save image to {out_path}")

    # ==================== PROPERTIES ====================
    @property
    def current_frame(self) -> np.ndarray:
        """Get current frame with proper error checking."""
        if self._current_frame is None:
            raise ValueError("No frame available")
        return self._current_frame.copy()

    @property
    def output_path(self) -> str:
        """Get output directory path."""
        return self._outpath