"""
Video analysis module for Cricket Analysis System.
Handles Sports2D integration and video processing.
"""

import cv2
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from Sports2D.Sports2D import process as sports2d_process

from .data_models import VideoInfo, AnalysisResult, AnalysisConfig


class VideoAnalyzer:
    """Handles video analysis using Sports2D pose estimation."""

    def __init__(self, config: AnalysisConfig, output_dir: str = "analysis_output"):
        """
        Initialize the video analyzer.

        Args:
            config: Analysis configuration
            output_dir: Output directory for analysis results
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Cache for video information to avoid repeated reads
        self._video_info_cache = {}

    def get_video_info(self, video_path: str) -> Optional[VideoInfo]:
        """
        Get video information including frame count, FPS, etc.

        Args:
            video_path: Path to the video file

        Returns:
            VideoInfo object or None if failed
        """
        # Use cache if available
        if video_path in self._video_info_cache:
            return self._video_info_cache[video_path]

        try:
            # Validate file exists
            if not Path(video_path).exists():
                print(f"Video file not found: {video_path}")
                return None

            video_info = VideoInfo.from_video_path(video_path)

            # Validate video info
            if video_info.frame_count <= 0:
                print(
                    f"Invalid frame count for {video_path}: {video_info.frame_count}")
                return None

            # Cache the result
            self._video_info_cache[video_path] = video_info
            return video_info

        except Exception as e:
            print(f"Error getting video info for {video_path}: {e}")
            return None

    def analyze_video(self, video_path: str, output_dir: str) -> Optional[AnalysisResult]:
        """
        Run Sports2D analysis on a video.

        Args:
            video_path: Path to the input video
            output_dir: Directory for output files

        Returns:
            AnalysisResult object or None if failed
        """
        try:
            print(f"Analyzing video: {video_path}")

            # Validate input
            if not Path(video_path).exists():
                print(f"Video file not found: {video_path}")
                return None

            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Generate Sports2D configuration
            sports2d_config = self.config.to_sports2d_config(
                video_path, str(output_path))

            # Run Sports2D analysis
            sports2d_process(sports2d_config)

            # Parse results
            video_name = Path(video_path).stem
            result_dir = output_path / f"{video_name}_Sports2D"

            if not result_dir.exists():
                print(f"Sports2D output directory not found: {result_dir}")
                return None

            # Find generated files
            angles_file = result_dir / \
                f"{video_name}_Sports2D_angles_person00.mot"
            analyzed_video = result_dir / f"{video_name}_Sports2D.mp4"

            return AnalysisResult(
                video_name=video_name,
                result_dir=str(result_dir),
                angles_file=str(angles_file) if angles_file.exists() else None,
                analyzed_path=str(
                    analyzed_video) if analyzed_video.exists() else None,
                success=True
            )

        except Exception as e:
            print(f"Error analyzing video {video_path}: {e}")
            return None

    def get_frame_at_index(self, video_path: str, frame_index: int) -> Optional[any]:
        """
        Extract a specific frame from a video with improved error handling.

        Args:
            video_path: Path to the video file
            frame_index: Index of the frame to extract

        Returns:
            Frame as numpy array or None if failed
        """
        try:
            # Validate inputs
            if not video_path or not isinstance(video_path, str):
                print(f"Invalid video path: {video_path}")
                return None

            if not Path(video_path).exists():
                print(f"Video file not found: {video_path}")
                return None

            if frame_index < 0:
                print(f"Invalid frame index: {frame_index} (must be >= 0)")
                return None

            # Get video info to validate frame index
            video_info = self.get_video_info(video_path)
            if video_info is None:
                print(f"Could not get video info for: {video_path}")
                return None

            if frame_index >= video_info.frame_count:
                print(
                    f"Frame index {frame_index} exceeds video frame count {video_info.frame_count} for {video_path}")
                return None

            # Open video capture
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"Could not open video: {video_path}")
                return None

            # Set frame position
            ret_set = cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            if not ret_set:
                print(
                    f"Could not set frame position to {frame_index} for {video_path}")
                cap.release()
                return None

            # Read frame
            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                print(f"Could not read frame {frame_index} from {video_path}")
                return None

            # Validate frame
            if frame.size == 0:
                print(f"Empty frame at index {frame_index} for {video_path}")
                return None

            return frame

        except Exception as e:
            print(
                f"Error extracting frame {frame_index} from {video_path}: {e}")
            return None

    def get_all_frames(self, video_path: str) -> Tuple[List[any], int]:
        """
        Extract all frames from a video.

        Args:
            video_path: Path to the video file

        Returns:
            Tuple of (frames_list, frame_count)
        """
        try:
            # Validate input
            if not Path(video_path).exists():
                print(f"Video file not found: {video_path}")
                return [], 0

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"Could not open video: {video_path}")
                return [], 0

            frames = []
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            for i in range(frame_count):
                ret, frame = cap.read()
                if ret and frame is not None:
                    frames.append(frame)
                else:
                    print(f"Failed to read frame {i} from {video_path}")
                    break

            cap.release()
            return frames, len(frames)

        except Exception as e:
            print(f"Error extracting frames from {video_path}: {e}")
            return [], 0

    def validate_frame_access(self, video_path: str, frame_index: int) -> Dict[str, Any]:
        """
        Validate frame access and return diagnostic information.

        Args:
            video_path: Path to the video file
            frame_index: Index of the frame to validate

        Returns:
            Dictionary with validation results and diagnostic info
        """
        result = {
            'valid': False,
            'video_exists': False,
            'video_openable': False,
            'frame_index_valid': False,
            'frame_readable': False,
            'video_info': None,
            'error_message': None
        }

        try:
            # Check file existence
            if not Path(video_path).exists():
                result['error_message'] = f"Video file not found: {video_path}"
                return result
            result['video_exists'] = True

            # Get video info
            video_info = self.get_video_info(video_path)
            if video_info is None:
                result['error_message'] = f"Could not read video info: {video_path}"
                return result
            result['video_info'] = video_info

            # Check if video can be opened
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                result['error_message'] = f"Could not open video: {video_path}"
                cap.release()
                return result
            result['video_openable'] = True

            # Validate frame index
            if frame_index < 0 or frame_index >= video_info.frame_count:
                result['error_message'] = f"Frame index {frame_index} out of range [0, {video_info.frame_count-1}]"
                cap.release()
                return result
            result['frame_index_valid'] = True

            # Test frame reading
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                result['error_message'] = f"Could not read frame {frame_index}"
                return result
            result['frame_readable'] = True
            result['valid'] = True

        except Exception as e:
            result['error_message'] = f"Exception during validation: {e}"

        return result


class AnglesDataProcessor:
    """Processes angle data from Sports2D output files."""

    @staticmethod
    def read_angles_file(angles_file_path: str) -> pd.DataFrame:
        """
        Read and parse a Sports2D angles file (.mot format).

        Args:
            angles_file_path: Path to the angles file

        Returns:
            DataFrame containing angle data
        """
        angles_path = Path(angles_file_path)

        if not angles_path.exists():
            # Try to find alternative files
            result_dir = angles_path.parent
            video_name = angles_path.stem.replace(
                '_Sports2D_angles_person00', '')

            # Look for any .mot files
            mot_files = list(result_dir.glob('*.mot'))
            if mot_files:
                angles_path = mot_files[0]
            else:
                # Try alternative patterns
                alt_patterns = [
                    f"{video_name}_angles_person00.mot",
                    f"{video_name}_Sports2D_angles.mot",
                    f"{video_name}_angles.mot"
                ]

                for pattern in alt_patterns:
                    alt_path = result_dir / pattern
                    if alt_path.exists():
                        angles_path = alt_path
                        break
                else:
                    return pd.DataFrame()

        try:
            return AnglesDataProcessor._parse_mot_file(angles_path)
        except Exception as e:
            print(f"Error reading angles file {angles_path}: {e}")
            return pd.DataFrame()

    @staticmethod
    def _parse_mot_file(file_path: Path) -> pd.DataFrame:
        """
        Parse a .mot file format used by Sports2D.

        Args:
            file_path: Path to the .mot file

        Returns:
            DataFrame with parsed angle data
        """
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Find data start after 'endheader'
        data_start = None
        for i, line in enumerate(lines):
            if line.strip().startswith('endheader'):
                data_start = i + 1
                break

        if data_start is None:
            raise ValueError("Could not find 'endheader' marker in .mot file")

        # Parse header and data
        header_line = lines[data_start].strip()
        columns = header_line.split('\t')

        # Parse data lines
        data_start += 1
        data = []
        for line in lines[data_start:]:
            if line.strip():
                data.append(line.strip().split('\t'))

        df = pd.DataFrame(data, columns=columns)

        # Convert numeric columns
        numeric_columns = df.columns[1:]  # Skip first column (usually time)
        df[numeric_columns] = df[numeric_columns].apply(
            pd.to_numeric, errors='coerce')

        return df

    @staticmethod
    def extract_frame_data(angles_df: pd.DataFrame, frame_index: int) -> Dict[str, float]:
        """
        Extract angle data for a specific frame.

        Args:
            angles_df: DataFrame containing angle data
            frame_index: Index of the frame

        Returns:
            Dictionary of angle measurements
        """
        if angles_df.empty or frame_index >= len(angles_df):
            return {}

        frame_data = angles_df.iloc[frame_index].to_dict()

        # Convert to float and handle NaN values
        cleaned_data = {}
        for key, value in frame_data.items():
            try:
                if pd.isna(value):
                    cleaned_data[key] = 0.0
                else:
                    cleaned_data[key] = float(value)
            except (ValueError, TypeError):
                cleaned_data[key] = 0.0

        return cleaned_data
