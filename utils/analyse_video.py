#!/usr/bin/env python3
"""
Cricket Analysis System - Advanced Video Analysis Utilities

This module provides advanced video processing and motion analysis utilities
for the Cricket Analysis System. It includes classes and methods for:

- Motion pattern detection and analysis
- Bowling phase identification (approach, delivery, release, follow-through)
- Key frame extraction from bowling actions
- Multi-video comparison and similarity analysis
- Motion intensity tracking and dominant motion frame detection
- Background subtraction for motion analysis

Classes:
    VideoAnalysisHelper: Static methods for advanced video analysis
    VideoComparator: Compare multiple videos for bowling action analysis

Dependencies:
    - OpenCV (cv2): Video processing and background subtraction
    - NumPy: Numerical operations and statistical analysis
    - pathlib: Path handling
    - json: Data serialization
    - collections: Data structures

Usage:
    Can be run as a standalone module for testing:
    python utils/analyse_video.py --test-video path/to/video.mp4
    
    Or imported and used programmatically:
    from utils.analyse_video import VideoAnalysisHelper, VideoComparator
    
Author: Cricket Analysis System
Version: 1.0
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import json
from collections import defaultdict
import argparse
import sys

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


if __name__ == "__main__":
    """
    Test the video analysis utilities with command-line arguments.

    Usage examples:
    python utils/analyse_video.py --test-video input/video/set1/sample.mp4
    python utils/analyse_video.py --compare-videos video1.mp4 video2.mp4
    python utils/analyse_video.py --help
    """
    parser = argparse.ArgumentParser(
        description="Test Cricket Analysis Video Processing Utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python utils/analyse_video.py --test-video input/video/set1/sample.mp4
  python utils/analyse_video.py --compare-videos video1.mp4 video2.mp4 video3.mp4
  python utils/analyse_video.py --analyze-phases input/video/bowling.mp4
  python utils/analyse_video.py --create-test-video
        """
    )

    parser.add_argument('--test-video', type=str,
                        help='Test motion analysis with specified video file')
    parser.add_argument('--compare-videos', nargs='+',
                        help='Compare motion patterns across multiple videos')
    parser.add_argument('--analyze-phases', type=str,
                        help='Analyze bowling phases in specified video')
    parser.add_argument('--create-test-video', action='store_true',
                        help='Create a sample test video for demonstration')
    parser.add_argument('--output-dir', type=str, default='test_output',
                        help='Output directory for test results (default: test_output)')
    parser.add_argument('--roi', type=int, nargs=4, metavar=('X', 'Y', 'W', 'H'),
                        help='Region of interest for motion analysis (x y width height)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("🏏 Testing Cricket Analysis Video Processing Utilities")
    print("=" * 65)

    try:
        # Test 1: Single Video Motion Analysis
        if args.test_video:
            print(f"\n🎥 Testing Motion Analysis with: {args.test_video}")

            if not Path(args.test_video).exists():
                print(f"❌ Error: Video file not found: {args.test_video}")
                sys.exit(1)

            # Convert ROI to tuple if provided
            roi = tuple(args.roi) if args.roi else None
            if roi:
                print(
                    f"   Using ROI: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")

            # Analyze motion patterns
            print("   🔧 Analyzing motion patterns...")
            motion_data = VideoAnalysisHelper.detect_motion_patterns(
                args.test_video, roi=roi
            )

            print(f"   ✅ Motion analysis completed!")
            print(
                f"   📊 Total frames analyzed: {len(motion_data['frame_motion'])}")
            print(
                f"   📈 Max motion intensity: {motion_data['max_motion_intensity']:.4f}")
            print(
                f"   🎯 Dominant motion frame: {motion_data['dominant_motion_frame']}")

            # Save motion data
            motion_file = output_dir / "motion_analysis.json"
            with open(motion_file, 'w') as f:
                json.dump(motion_data, f, indent=2, default=str)
            print(f"   💾 Motion data saved: {motion_file}")

            # Analyze bowling phases
            print("   🔧 Analyzing bowling phases...")
            extractor = FrameExtractor(args.test_video)
            fps = extractor.get_fps()
            phases = VideoAnalysisHelper.analyze_bowling_phases(
                motion_data, fps)

            print(f"   ✅ Phase analysis completed!")
            for phase_name, frame_idx in phases.items():
                if isinstance(frame_idx, int):
                    timestamp = frame_idx / fps if fps > 0 else 0
                    print(
                        f"   🏏 {phase_name}: Frame {frame_idx} (t={timestamp:.2f}s)")

            # Extract key frames
            print("   🔧 Extracting key frames...")
            key_frames_dir = output_dir / "key_frames"
            saved_frames = VideoAnalysisHelper.extract_key_frames(
                args.test_video, phases, str(key_frames_dir)
            )

            print(f"   ✅ Key frames extracted: {len(saved_frames)}")
            for phase, path in saved_frames.items():
                print(f"   🖼️  {phase}: {path}")

        # Test 2: Video Comparison
        if args.compare_videos:
            print(
                f"\n🔄 Testing Video Comparison with {len(args.compare_videos)} videos:")
            for i, video in enumerate(args.compare_videos, 1):
                print(f"   {i}. {video}")

            # Check all videos exist
            missing_videos = [
                v for v in args.compare_videos if not Path(v).exists()]
            if missing_videos:
                print(f"❌ Error: Videos not found: {missing_videos}")
                sys.exit(1)

            # Create comparator
            print("   🔧 Creating video comparator...")
            comparator = VideoComparator(args.compare_videos)

            # Compare motion patterns
            print("   🔧 Comparing motion patterns...")
            comparison = comparator.compare_motion_patterns()

            print(f"   ✅ Comparison completed!")

            # Display results
            metrics = comparison['comparative_metrics']
            similarities = comparison['similarity_scores']

            print(f"   📊 Motion Intensity Comparison:")
            for video, intensity in metrics['max_motion_intensities'].items():
                print(f"      {video}: {intensity:.4f}")

            print(f"   🔗 Similarity Scores:")
            for pair, score in similarities.items():
                print(f"      {pair}: {score:.3f}")

            # Save comparison report
            report_file = output_dir / "video_comparison_report.json"
            report_path = comparator.generate_comparison_report(
                str(report_file))
            print(f"   💾 Comparison report saved: {report_path}")

        # Test 3: Phase Analysis Only
        if args.analyze_phases:
            print(f"\n🎯 Testing Phase Analysis with: {args.analyze_phases}")

            if not Path(args.analyze_phases).exists():
                print(f"❌ Error: Video file not found: {args.analyze_phases}")
                sys.exit(1)

            # Analyze motion and phases
            motion_data = VideoAnalysisHelper.detect_motion_patterns(
                args.analyze_phases)
            extractor = FrameExtractor(args.analyze_phases)
            phases = VideoAnalysisHelper.analyze_bowling_phases(
                motion_data, extractor.get_fps())

            # Create detailed phase report
            phase_report = {
                'video_path': args.analyze_phases,
                'fps': extractor.get_fps(),
                'total_frames': extractor.get_frame_count(),
                'motion_data': motion_data,
                'phases': phases
            }

            phase_file = output_dir / "phase_analysis.json"
            with open(phase_file, 'w') as f:
                json.dump(phase_report, f, indent=2, default=str)
            print(f"   💾 Phase analysis saved: {phase_file}")

        # Test 4: Create Test Video
        if args.create_test_video:
            print("\n🛠️  Creating sample test video...")

            # Create a simple test video with moving objects
            test_video_path = output_dir / "sample_test_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                str(test_video_path), fourcc, 30.0, (640, 480))

            # Create frames with moving circle (simulating bowling motion)
            for frame_num in range(150):  # 5 seconds at 30fps
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

                # Moving circle to simulate bowling action
                x = int(50 + (frame_num * 4) % 540)  # Move horizontally
                # Slight vertical motion
                y = int(240 + 50 * np.sin(frame_num * 0.1))

                # Draw person (rectangle) and ball (circle)
                cv2.rectangle(frame, (x-20, y-40),
                              (x+20, y+40), (0, 255, 0), -1)
                cv2.circle(frame, (x+30, y), 8, (0, 0, 255), -1)

                # Add frame number
                cv2.putText(frame, f"Frame {frame_num}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                writer.write(frame)

            writer.release()
            print(f"   ✅ Sample test video created: {test_video_path}")

            # Test with created video
            motion_data = VideoAnalysisHelper.detect_motion_patterns(
                str(test_video_path))
            print(f"   ✅ Motion analysis on test video completed!")
            print(
                f"   📊 Dominant motion frame: {motion_data['dominant_motion_frame']}")

        # Default behavior if no arguments
        if not any([args.test_video, args.compare_videos, args.analyze_phases, args.create_test_video]):
            print("\n🎯 Running default tests...")

            # Look for sample videos
            sample_dirs = [Path("input/video/set1"), Path("input/video")]
            sample_videos = []

            for sample_dir in sample_dirs:
                if sample_dir.exists():
                    video_files = list(sample_dir.glob(
                        "*.mp4")) + list(sample_dir.glob("*.avi"))
                    # Take first 2 videos
                    sample_videos.extend(video_files[:2])

            if sample_videos:
                print(
                    f"   Found sample videos: {[str(v) for v in sample_videos]}")

                # Test with first video
                test_video = sample_videos[0]
                motion_data = VideoAnalysisHelper.detect_motion_patterns(
                    str(test_video))

                motion_file = output_dir / "default_motion_analysis.json"
                with open(motion_file, 'w') as f:
                    json.dump(motion_data, f, indent=2, default=str)
                print(f"   ✅ Default motion analysis saved: {motion_file}")

                # If multiple videos, compare them
                if len(sample_videos) > 1:
                    comparator = VideoComparator(
                        [str(v) for v in sample_videos])
                    report_path = comparator.generate_comparison_report(
                        str(output_dir / "default_comparison.json")
                    )
                    print(
                        f"   ✅ Default comparison report saved: {report_path}")
            else:
                print("   No sample videos found, creating test video...")
                # Create and test with synthetic video
                test_video_path = output_dir / "default_test_video.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(
                    str(test_video_path), fourcc, 30.0, (320, 240))

                for i in range(60):  # 2 seconds
                    frame = np.random.randint(
                        0, 255, (240, 320, 3), dtype=np.uint8)
                    writer.write(frame)
                writer.release()

                print(f"   ✅ Default test video created: {test_video_path}")

        print(f"\n🎉 All tests completed successfully!")
        print(f"📁 Check output directory: {output_dir}")
        print("\n📖 Usage examples:")
        print("   python utils/analyse_video.py --test-video input/video/sample.mp4")
        print("   python utils/analyse_video.py --compare-videos video1.mp4 video2.mp4")
        print("   python utils/analyse_video.py --create-test-video")

    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
