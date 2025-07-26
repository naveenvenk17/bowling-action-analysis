"""
Image analysis utilities for Cricket Analysis System.
Complementary tools for image processing and analysis.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any


class ImageProcessor:
    """Utility class for image processing operations."""

    @staticmethod
    def enhance_image_quality(image: np.ndarray) -> np.ndarray:
        """
        Enhance image quality for better pose detection.

        Args:
            image: Input image as numpy array

        Returns:
            Enhanced image
        """
        # Convert to LAB color space for better processing
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])

        # Convert back to BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return enhanced

    @staticmethod
    def resize_maintain_aspect_ratio(
        image: np.ndarray,
        target_size: Tuple[int, int] = (1280, 720)
    ) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio.

        Args:
            image: Input image
            target_size: Target (width, height)

        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        target_w, target_h = target_size

        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)

        # New dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize image
        resized = cv2.resize(image, (new_w, new_h),
                             interpolation=cv2.INTER_AREA)

        # Create new image with target size
        result = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        # Center the resized image
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        return result

    @staticmethod
    def extract_person_roi(
        image: np.ndarray,
        person_bbox: Tuple[int, int, int, int],
        padding: float = 0.1
    ) -> np.ndarray:
        """
        Extract region of interest around detected person.

        Args:
            image: Input image
            person_bbox: Bounding box (x1, y1, x2, y2)
            padding: Padding factor around bbox

        Returns:
            ROI image
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = person_bbox

        # Add padding
        pad_w = int((x2 - x1) * padding)
        pad_h = int((y2 - y1) * padding)

        # Expand bbox with padding
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)

        return image[y1:y2, x1:x2]

    @staticmethod
    def create_pose_overlay(
        image: np.ndarray,
        keypoints: Dict[str, Tuple[int, int]],
        connections: Optional[list] = None
    ) -> np.ndarray:
        """
        Create pose overlay on image.

        Args:
            image: Base image
            keypoints: Dictionary of keypoint names to (x, y) coordinates
            connections: List of keypoint pairs to connect

        Returns:
            Image with pose overlay
        """
        overlay = image.copy()

        # Default connections for cricket pose analysis
        if connections is None:
            connections = [
                ('left_shoulder', 'right_shoulder'),
                ('left_shoulder', 'left_elbow'),
                ('left_elbow', 'left_wrist'),
                ('right_shoulder', 'right_elbow'),
                ('right_elbow', 'right_wrist'),
                ('left_hip', 'right_hip'),
                ('left_shoulder', 'left_hip'),
                ('right_shoulder', 'right_hip'),
                ('left_hip', 'left_knee'),
                ('left_knee', 'left_ankle'),
                ('right_hip', 'right_knee'),
                ('right_knee', 'right_ankle')
            ]

        # Draw connections
        for start_point, end_point in connections:
            if start_point in keypoints and end_point in keypoints:
                start_pos = keypoints[start_point]
                end_pos = keypoints[end_point]
                cv2.line(overlay, start_pos, end_pos, (0, 255, 0), 2)

        # Draw keypoints
        for name, (x, y) in keypoints.items():
            cv2.circle(overlay, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(overlay, name, (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        return overlay


class FrameExtractor:
    """Utility for extracting and processing video frames."""

    def __init__(self, video_path: str):
        """Initialize with video path."""
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

    def get_frame_count(self) -> int:
        """Get total frame count."""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_fps(self) -> float:
        """Get video FPS."""
        return self.cap.get(cv2.CAP_PROP_FPS)

    def extract_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """
        Extract specific frame.

        Args:
            frame_index: Index of frame to extract

        Returns:
            Frame image or None if failed
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        return frame if ret else None

    def extract_frames_range(
        self,
        start_frame: int,
        end_frame: int,
        step: int = 1
    ) -> list:
        """
        Extract frames in a range.

        Args:
            start_frame: Starting frame index
            end_frame: Ending frame index
            step: Step size between frames

        Returns:
            List of frame images
        """
        frames = []

        for frame_idx in range(start_frame, end_frame + 1, step):
            frame = self.extract_frame(frame_idx)
            if frame is not None:
                frames.append(frame)

        return frames

    def extract_frames_around_point(
        self,
        center_frame: int,
        window_size: int = 10
    ) -> Dict[int, np.ndarray]:
        """
        Extract frames around a specific point.

        Args:
            center_frame: Center frame index
            window_size: Number of frames before and after center

        Returns:
            Dictionary mapping frame indices to images
        """
        start_frame = max(0, center_frame - window_size)
        end_frame = min(self.get_frame_count() - 1, center_frame + window_size)

        frames = {}
        for frame_idx in range(start_frame, end_frame + 1):
            frame = self.extract_frame(frame_idx)
            if frame is not None:
                frames[frame_idx] = frame

        return frames

    def save_frame(
        self,
        frame_index: int,
        output_path: str,
        enhance: bool = True
    ) -> bool:
        """
        Save specific frame to file.

        Args:
            frame_index: Index of frame to save
            output_path: Output file path
            enhance: Whether to enhance image quality

        Returns:
            True if successful, False otherwise
        """
        frame = self.extract_frame(frame_index)
        if frame is None:
            return False

        if enhance:
            frame = ImageProcessor.enhance_image_quality(frame)

        return cv2.imwrite(output_path, frame)

    def __del__(self):
        """Clean up video capture."""
        if hasattr(self, 'cap'):
            self.cap.release()
