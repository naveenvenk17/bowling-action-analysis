"""
Data models for Cricket Analysis System.
Defines the structure and types for analysis data.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd


@dataclass
class VideoInfo:
    """Information about a video file."""
    path: str
    frame_count: int
    fps: float
    duration: float

    @classmethod
    def from_video_path(cls, video_path: str) -> 'VideoInfo':
        """Create VideoInfo from a video file path."""
        import cv2
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0
        cap.release()

        return cls(
            path=video_path,
            frame_count=frame_count,
            fps=fps,
            duration=duration
        )


@dataclass
class AnalysisResult:
    """Result of video analysis."""
    video_name: str
    result_dir: str
    analyzed_path: Optional[str] = None
    angles_file: Optional[str] = None
    success: bool = False
    suggested_release_frame: Optional[int] = None
    manual_release_frame: Optional[int] = None

    @property
    def release_frame(self) -> Optional[int]:
        """Get the final release frame (manual takes precedence)."""
        return self.manual_release_frame or self.suggested_release_frame


@dataclass
class ReleasePointData:
    """Data extracted at a release point."""
    video_name: str
    frame_index: int
    angles_data: Dict[str, float]
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV export."""
        result = {
            'video_name': self.video_name,
            'release_frame': self.frame_index,
            'timestamp': self.timestamp
        }
        result.update(self.angles_data)
        return result


@dataclass
class YOLODetectionResult:
    """Result from YOLO release point detection."""
    video_name: str
    release_frame: Optional[int]
    confidence: Optional[float]
    detection_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisConfig:
    """Configuration for analysis parameters."""
    # Sports2D Configuration
    nb_persons_to_detect: int = 1
    person_ordering_method: str = 'highest_likelihood'
    first_person_height: float = 1.75
    pose_model: str = 'HALPE_26'
    mode: str = 'balanced'
    det_frequency: int = 4
    use_gpu: bool = True

    # YOLO Configuration
    yolo_confidence_threshold: float = 0.25
    yolo_model_path: Optional[str] = None

    # Processing Configuration
    save_vid: bool = True
    save_img: bool = True
    save_pose: bool = True
    calculate_angles: bool = True
    save_angles: bool = True

    # Output Configuration
    show_realtime_results: bool = False
    make_c3d: bool = True

    # Angle Configuration
    flip_left_right: bool = False  # Set to false to avoid frame of reference issues

    def to_sports2d_config(self, video_path: str, result_dir: str) -> Dict[str, Any]:
        """Convert to Sports2D configuration format."""
        return {
            'base': {
                'video_input': str(video_path),
                'result_dir': str(result_dir),
                'save_vid': self.save_vid,
                'save_img': self.save_img,
                'save_pose': self.save_pose,
                'calculate_angles': self.calculate_angles,
                'save_angles': self.save_angles,
                'nb_persons_to_detect': self.nb_persons_to_detect,
                'person_ordering_method': self.person_ordering_method,
                'first_person_height': self.first_person_height,
                'visible_side': ['auto'],
                'time_range': [],
                'video_dir': '',
                'show_realtime_results': self.show_realtime_results,
                'load_trc_px': '',
                'compare': False,
                'webcam_id': 0,
                'input_size': [1280, 720]
            },
            'pose': {
                'slowmo_factor': 1,
                'pose_model': self.pose_model,
                'mode': self.mode,
                'det_frequency': self.det_frequency,
                'device': 'cuda' if self.use_gpu else 'cpu',
                'backend': 'onnxruntime' if self.use_gpu else 'auto',
                'tracking_mode': 'sports2d',
                'keypoint_likelihood_threshold': 0.1,
                'average_likelihood_threshold': 0.3,
                'keypoint_number_threshold': 0.1
            },
            'px_to_meters_conversion': {
                'to_meters': True,
                'make_c3d': self.make_c3d,
                'save_calib': True,
                'floor_angle': 'auto',
                'xy_origin': ['auto'],
                'calib_file': ''
            },
            'angles': {
                'display_angle_values_on': ['body'],
                'fontSize': 0.3,
                'joint_angles': ['Right ankle', 'Left ankle', 'Right knee', 'Left knee',
                                 'Right hip', 'Left hip', 'Right shoulder', 'Left shoulder',
                                 'Right elbow', 'Left elbow', 'Right wrist', 'Left wrist'],
                'segment_angles': ['Right foot', 'Left foot', 'Right shank', 'Left shank',
                                   'Right thigh', 'Left thigh', 'Trunk', 'Right arm', 'Left arm',
                                   'Right forearm', 'Left forearm'],
                'flip_left_right': self.flip_left_right,
                'correct_segment_angles_with_floor_angle': True
            },
            'post-processing': {
                'interpolate': True,
                'interp_gap_smaller_than': 10,
                'fill_large_gaps_with': 'last_value',
                'filter': True,
                'show_graphs': False,
                'filter_type': 'butterworth',
                'butterworth': {'order': 4, 'cut_off_frequency': 6},
                'gaussian': {'sigma_kernel': 1},
                'loess': {'nb_values_used': 5},
                'median': {'kernel_size': 3}
            },
            'kinematics': {
                'do_ik': False,
                'use_augmentation': False,
                'feet_on_floor': False,
                'use_simple_model': False,
                'participant_mass': [70.0],
                'right_left_symmetry': True,
                'default_height': 1.7,
                'fastest_frames_to_remove_percent': 0.1,
                'close_to_zero_speed_px': 50,
                'close_to_zero_speed_m': 0.2,
                'large_hip_knee_angles': 45,
                'trimmed_extrema_percent': 0.5,
                'remove_individual_scaling_setup': True,
                'remove_individual_ik_setup': True
            },
            'logging': {'use_custom_logging': False}
        }
