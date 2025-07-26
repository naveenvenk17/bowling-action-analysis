"""
Video analysis utilities for Cricket Analysis System.
Complementary tools for video processing and analysis.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import json
from collections import defaultdict

from .analyse_image import FrameExtractor, ImageProcessor


class VideoAnalysisHelper:
    """Helper utilities for advanced video analysis."""

    @staticmethod
    def detect_motion_patterns(
        video_path: str,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> Dict[str, Any]:
        """
        Detect motion patterns in video for bowling action analysis.

        Args:
            video_path: Path to video file
            start_frame: Starting frame for analysis
            end_frame: Ending frame (None for full video)
            roi: Region of interest (x, y, w, h)

        Returns:
            Dictionary with motion analysis results
        """
        extractor = FrameExtractor(video_path)

        if end_frame is None:
            end_frame = extractor.get_frame_count() - 1

        # Initialize background subtractor
        backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

        motion_data = {
            'frame_motion': [],
            'motion_intensity': [],
            'motion_centers': [],
            'dominant_motion_frame': None
        }

        prev_frame = None
        max_motion = 0
        max_motion_frame = start_frame

        for frame_idx in range(start_frame, end_frame + 1):
            frame = extractor.extract_frame(frame_idx)
            if frame is None:
                continue

            # Apply ROI if specified
            if roi:
                x, y, w, h = roi
                frame = frame[y:y+h, x:x+w]

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Background subtraction
            fg_mask = backSub.apply(frame)

            # Calculate motion intensity
            motion_pixels = cv2.countNonZero(fg_mask)
            motion_intensity = motion_pixels / \
                (frame.shape[0] * frame.shape[1])

            # Find motion center
            moments = cv2.moments(fg_mask)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                motion_center = (cx, cy)
            else:
                motion_center = None

            motion_data['frame_motion'].append(motion_pixels)
            motion_data['motion_intensity'].append(motion_intensity)
            motion_data['motion_centers'].append(motion_center)

            # Track maximum motion frame
            if motion_intensity > max_motion:
                max_motion = motion_intensity
                max_motion_frame = frame_idx

            prev_frame = gray

        motion_data['dominant_motion_frame'] = max_motion_frame
        motion_data['max_motion_intensity'] = max_motion

        return motion_data

    @staticmethod
    def analyze_bowling_phases(
        motion_data: Dict[str, Any],
        fps: float = 30.0
    ) -> Dict[str, Any]:
        """
        Analyze bowling action phases based on motion patterns.

        Args:
            motion_data: Motion analysis data from detect_motion_patterns
            fps: Video frames per second

        Returns:
            Dictionary with phase analysis
        """
        motion_intensities = motion_data['motion_intensity']

        if not motion_intensities:
            return {'error': 'No motion data available'}

        # Smooth motion data
        window_size = max(3, int(fps * 0.1))  # 0.1 second window
        smoothed = []
        for i in range(len(motion_intensities)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(motion_intensities), i + window_size // 2 + 1)
            smoothed.append(np.mean(motion_intensities[start_idx:end_idx]))

        # Find peaks and valleys
        peaks = []
        valleys = []

        for i in range(1, len(smoothed) - 1):
            if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]:
                peaks.append(i)
            elif smoothed[i] < smoothed[i-1] and smoothed[i] < smoothed[i+1]:
                valleys.append(i)

        # Identify phases
        phases = {
            'approach_phase': None,
            'delivery_stride': None,
            'release_point': None,
            'follow_through': None,
            'phase_transitions': []
        }

        if peaks:
            # Delivery stride is typically the highest peak
            delivery_peak = max(peaks, key=lambda x: smoothed[x])
            phases['delivery_stride'] = delivery_peak

            # Release point is shortly after delivery stride peak
            release_candidates = [p for p in peaks if p > delivery_peak]
            if release_candidates:
                phases['release_point'] = min(release_candidates)

            # Approach phase is before delivery stride
            approach_candidates = [p for p in peaks if p < delivery_peak]
            if approach_candidates:
                phases['approach_phase'] = max(approach_candidates)

            # Follow through is after release
            if phases['release_point']:
                followthrough_candidates = [
                    p for p in peaks if p > phases['release_point']]
                if followthrough_candidates:
                    phases['follow_through'] = min(followthrough_candidates)

        return phases

    @staticmethod
    def extract_key_frames(
        video_path: str,
        analysis_result: Dict[str, Any],
        output_dir: str = "key_frames"
    ) -> Dict[str, str]:
        """
        Extract key frames from bowling action analysis.

        Args:
            video_path: Path to video file
            analysis_result: Results from analyze_bowling_phases
            output_dir: Directory to save key frames

        Returns:
            Dictionary mapping phase names to saved image paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        extractor = FrameExtractor(video_path)
        video_name = Path(video_path).stem

        saved_frames = {}

        # Extract frames for each identified phase
        for phase_name, frame_index in analysis_result.items():
            if frame_index is not None and isinstance(frame_index, int):
                frame = extractor.extract_frame(frame_index)
                if frame is not None:
                    # Enhance frame quality
                    enhanced_frame = ImageProcessor.enhance_image_quality(
                        frame)

                    # Save frame
                    filename = f"{video_name}_{phase_name}_frame_{frame_index:06d}.png"
                    file_path = output_path / filename

                    if cv2.imwrite(str(file_path), enhanced_frame):
                        saved_frames[phase_name] = str(file_path)

        return saved_frames


class VideoComparator:
    """Compare multiple videos for bowling action analysis."""

    def __init__(self, video_paths: List[str]):
        """
        Initialize with list of video paths.

        Args:
            video_paths: List of video file paths to compare
        """
        self.video_paths = video_paths
        self.video_names = [Path(path).stem for path in video_paths]
        self.analysis_cache = {}

    def compare_motion_patterns(self) -> Dict[str, Any]:
        """
        Compare motion patterns across all videos.

        Returns:
            Dictionary with comparative analysis
        """
        all_motion_data = {}

        # Analyze each video
        for video_path in self.video_paths:
            video_name = Path(video_path).stem
            motion_data = VideoAnalysisHelper.detect_motion_patterns(
                video_path)
            all_motion_data[video_name] = motion_data

        # Comparative analysis
        comparison = {
            'individual_analysis': all_motion_data,
            'comparative_metrics': {},
            'similarity_scores': {}
        }

        # Calculate comparative metrics
        max_intensities = {name: max(data['motion_intensity']) if data['motion_intensity'] else 0
                           for name, data in all_motion_data.items()}

        mean_intensities = {name: np.mean(data['motion_intensity']) if data['motion_intensity'] else 0
                            for name, data in all_motion_data.items()}

        comparison['comparative_metrics'] = {
            'max_motion_intensities': max_intensities,
            'mean_motion_intensities': mean_intensities,
            'dominant_motion_frames': {name: data.get('dominant_motion_frame')
                                       for name, data in all_motion_data.items()}
        }

        # Calculate similarity scores between videos
        for i, name1 in enumerate(self.video_names):
            for name2 in self.video_names[i+1:]:
                similarity = self._calculate_motion_similarity(
                    all_motion_data[name1], all_motion_data[name2]
                )
                comparison['similarity_scores'][f"{name1}_vs_{name2}"] = similarity

        return comparison

    def _calculate_motion_similarity(
        self,
        motion_data1: Dict[str, Any],
        motion_data2: Dict[str, Any]
    ) -> float:
        """
        Calculate similarity score between two motion patterns.

        Args:
            motion_data1: Motion data from first video
            motion_data2: Motion data from second video

        Returns:
            Similarity score between 0 and 1
        """
        if not motion_data1['motion_intensity'] or not motion_data2['motion_intensity']:
            return 0.0

        # Normalize sequences to same length
        seq1 = np.array(motion_data1['motion_intensity'])
        seq2 = np.array(motion_data2['motion_intensity'])

        min_len = min(len(seq1), len(seq2))
        seq1 = seq1[:min_len]
        seq2 = seq2[:min_len]

        # Normalize intensity values
        seq1 = (seq1 - np.min(seq1)) / (np.max(seq1) - np.min(seq1) + 1e-8)
        seq2 = (seq2 - np.min(seq2)) / (np.max(seq2) - np.min(seq2) + 1e-8)

        # Calculate correlation coefficient
        correlation = np.corrcoef(seq1, seq2)[0, 1]

        # Handle NaN case
        if np.isnan(correlation):
            return 0.0

        # Convert to similarity score (0 to 1)
        return (correlation + 1) / 2

    def generate_comparison_report(self, output_path: str = "video_comparison.json") -> str:
        """
        Generate comprehensive comparison report.

        Args:
            output_path: Output file path for the report

        Returns:
            Path to the generated report
        """
        comparison_data = self.compare_motion_patterns()

        # Add metadata
        report = {
            'metadata': {
                'total_videos': len(self.video_paths),
                'video_names': self.video_names,
                'analysis_timestamp': str(np.datetime64('now'))
            },
            'motion_analysis': comparison_data
        }

        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        return output_path
