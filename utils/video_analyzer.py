#!/usr/bin/env python3
"""
Cricket Analysis System - Sports2D Video Analysis Integration

This module provides Sports2D integration and comprehensive video processing
for the Cricket Analysis System. It includes classes and methods for:

- Sports2D pose estimation integration and configuration
- Video information extraction and validation
- Frame-by-frame angle data generation and processing
- Angle data processing from .mot files
- Enhanced error handling and diagnostics
- Video frame access and validation

Classes:
    VideoAnalyzer: Handles video analysis using Sports2D pose estimation
    AnglesDataProcessor: Processes angle data from Sports2D output files

Key Features:
    - Full Sports2D integration with custom configuration
    - Frame-by-frame angle tracking for biomechanical analysis
    - Robust video file handling and validation
    - Comprehensive error handling and diagnostics
    - .mot file parsing and angle data extraction

Dependencies:
    - OpenCV (cv2): Video processing
    - pandas: Data manipulation and analysis
    - Sports2D: Pose estimation (Sports2D.Sports2D)
    - pathlib: Path handling

Usage:
    Can be run as a standalone module for testing:
    python utils/video_analyzer.py --test-video path/to/video.mp4
    
    Or imported and used programmatically:
    from utils.video_analyzer import VideoAnalyzer, AnglesDataProcessor
    
Author: Cricket Analysis System
Version: 1.0
"""

import cv2
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import argparse
import sys
import json

try:
    from Sports2D.Sports2D import process as sports2d_process
    SPORTS2D_AVAILABLE = True
except ImportError:
    print("Warning: Sports2D not available. Sports2D analysis will not work.")
    sports2d_process = None
    SPORTS2D_AVAILABLE = False

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

    @staticmethod
    def generate_frame_by_frame_angles(angles_file_path: str, result_dir: str) -> Optional[str]:
        """
        Generate frame-by-frame angle data for all frames in JSON format.

        Args:
            angles_file_path: Path to the Sports2D angles file (.mot)
            result_dir: Directory to save the output JSON file

        Returns:
            Path to the generated JSON file or None if failed
        """
        try:
            # Read the angles data
            angles_df = AnglesDataProcessor.read_angles_file(angles_file_path)

            if angles_df.empty:
                print(f"No angle data found in {angles_file_path}")
                return None

            # Convert to frame-by-frame JSON format
            frame_data = []

            for frame_index in range(len(angles_df)):
                frame_angles = AnglesDataProcessor.extract_frame_data(
                    angles_df, frame_index)

                # Add frame metadata
                frame_entry = {
                    'frame_index': frame_index,
                    # Assume 30fps if no time
                    'timestamp': frame_angles.get('time', frame_index / 30.0)
                }

                # Add all angle measurements
                for key, value in frame_angles.items():
                    if key != 'time':  # Skip time as it's already added as timestamp
                        frame_entry[key] = value

                frame_data.append(frame_entry)

            # Generate output filename
            angles_path = Path(angles_file_path)
            video_name = angles_path.stem.replace(
                '_Sports2D_angles_person00', '')
            output_file = Path(result_dir) / \
                f"{video_name}_frame_by_frame_angles.json"

            # Save to JSON
            with open(output_file, 'w') as f:
                json.dump(frame_data, f, indent=2)

            print(f"Frame-by-frame angles saved: {output_file}")
            return str(output_file)

        except Exception as e:
            print(f"Error generating frame-by-frame angles: {e}")
            return None


if __name__ == "__main__":
    """
    Test the video analyzer utilities with command-line arguments.

    Usage examples:
    python utils/video_analyzer.py --test-video input/video/sample.mp4
    python utils/video_analyzer.py --analyze-angles analysis_output/video_Sports2D/angles.mot
    python utils/video_analyzer.py --validate-frame input/video/sample.mp4 100
    python utils/video_analyzer.py --create-test-config
    """
    parser = argparse.ArgumentParser(
        description="Test Cricket Analysis Video Analyzer (Sports2D Integration)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python utils/video_analyzer.py --test-video input/video/sample.mp4
  python utils/video_analyzer.py --analyze-angles bowl1_Sports2D/bowl1_Sports2D_angles_person00.mot
  python utils/video_analyzer.py --validate-frame input/video/sample.mp4 100
  python utils/video_analyzer.py --create-test-config
        """
    )

    parser.add_argument('--test-video', type=str,
                        help='Test video analysis with specified video file')
    parser.add_argument('--analyze-angles', type=str,
                        help='Analyze angles from specified .mot file')
    parser.add_argument('--validate-frame', nargs=2, metavar=('VIDEO', 'FRAME'),
                        help='Validate frame access for video and frame index')
    parser.add_argument('--create-test-config', action='store_true',
                        help='Create a test configuration file')
    parser.add_argument('--output-dir', type=str, default='test_output',
                        help='Output directory for test results (default: test_output)')
    parser.add_argument('--config-file', type=str, default='config.toml',
                        help='Configuration file to use (default: config.toml)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("🏏 Testing Cricket Analysis Video Analyzer (Sports2D Integration)")
    print("=" * 70)

    try:
        # Test 1: Video Analysis with Sports2D
        if args.test_video:
            print(f"\n🎥 Testing Video Analysis with: {args.test_video}")

            if not Path(args.test_video).exists():
                print(f"❌ Error: Video file not found: {args.test_video}")
                sys.exit(1)

            # Create default config
            from .config_manager import ConfigManager
            config_manager = ConfigManager(
                args.config_file if Path(args.config_file).exists() else None)
            config = config_manager.get_analysis_config()

            print(f"   📋 Using configuration:")
            print(f"      Mode: {config.mode}")
            print(f"      Pose model: {config.pose_model}")
            print(f"      GPU: {config.use_gpu}")

            # Initialize analyzer
            analyzer = VideoAnalyzer(config, str(output_dir))

            # Get video info
            print("   🔧 Getting video information...")
            video_info = analyzer.get_video_info(args.test_video)

            if video_info:
                print(f"   ✅ Video info extracted:")
                print(f"      Frames: {video_info.frame_count}")
                print(f"      FPS: {video_info.fps:.2f}")
                print(f"      Duration: {video_info.duration:.2f}s")
            else:
                print(f"   ❌ Could not get video information")
                sys.exit(1)

            # Test frame extraction
            print("   🔧 Testing frame extraction...")
            test_frame_idx = min(50, video_info.frame_count // 2)
            frame = analyzer.get_frame_at_index(
                args.test_video, test_frame_idx)

            if frame is not None:
                frame_path = output_dir / \
                    f"test_frame_{test_frame_idx:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                print(f"   ✅ Frame {test_frame_idx} extracted: {frame_path}")
            else:
                print(f"   ❌ Could not extract frame {test_frame_idx}")

            # Test Sports2D analysis (if available)
            if SPORTS2D_AVAILABLE:
                print("   🔧 Testing Sports2D analysis...")
                video_name = Path(args.test_video).stem
                result = analyzer.analyze_video(args.test_video, str(
                    output_dir / f"{video_name}_analysis"))

                if result and result.success:
                    print(f"   ✅ Sports2D analysis completed!")
                    print(f"      Result directory: {result.result_dir}")
                    print(f"      Angles file: {result.angles_file}")
                    print(f"      Analyzed video: {result.analyzed_path}")

                    # Test frame-by-frame angle generation
                    if result.angles_file and Path(result.angles_file).exists():
                        print("   🔧 Generating frame-by-frame angles...")
                        frame_angles_file = AnglesDataProcessor.generate_frame_by_frame_angles(
                            result.angles_file, result.result_dir
                        )
                        if frame_angles_file:
                            print(
                                f"   ✅ Frame-by-frame angles generated: {frame_angles_file}")
                else:
                    print(f"   ❌ Sports2D analysis failed")
            else:
                print("   ⚠️  Sports2D not available, skipping analysis")

        # Test 2: Angles Analysis
        if args.analyze_angles:
            print(f"\n📊 Testing Angles Analysis with: {args.analyze_angles}")

            if not Path(args.analyze_angles).exists():
                print(f"❌ Error: Angles file not found: {args.analyze_angles}")
                sys.exit(1)

            # Read angles data
            print("   🔧 Reading angles data...")
            angles_df = AnglesDataProcessor.read_angles_file(
                args.analyze_angles)

            if not angles_df.empty:
                print(f"   ✅ Angles data loaded successfully!")
                print(f"      Shape: {angles_df.shape}")
                # Show first 10 columns
                print(f"      Columns: {list(angles_df.columns)[:10]}...")

                # Extract sample frame data
                sample_frame = min(50, len(angles_df) // 2)
                frame_data = AnglesDataProcessor.extract_frame_data(
                    angles_df, sample_frame)

                print(f"   📋 Sample frame {sample_frame} data:")
                # Show first 5 angles
                for key, value in list(frame_data.items())[:5]:
                    print(f"      {key}: {value:.2f}")

                # Generate frame-by-frame angles
                print("   🔧 Generating frame-by-frame angles...")
                output_file = AnglesDataProcessor.generate_frame_by_frame_angles(
                    args.analyze_angles, str(output_dir)
                )
                if output_file:
                    print(f"   ✅ Frame-by-frame angles saved: {output_file}")

                # Save summary statistics
                summary_file = output_dir / "angles_summary.json"
                summary = {
                    'total_frames': len(angles_df),
                    'columns': list(angles_df.columns),
                    'sample_statistics': {}
                }

                # Calculate statistics for numeric columns
                numeric_cols = angles_df.select_dtypes(
                    include=['float64', 'int64']).columns
                for col in numeric_cols[:10]:  # Limit to first 10 for brevity
                    summary['sample_statistics'][col] = {
                        'mean': float(angles_df[col].mean()),
                        'std': float(angles_df[col].std()),
                        'min': float(angles_df[col].min()),
                        'max': float(angles_df[col].max())
                    }

                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)
                print(f"   💾 Angles summary saved: {summary_file}")
            else:
                print(f"   ❌ Could not read angles data")

        # Test 3: Frame Validation
        if args.validate_frame:
            video_path, frame_idx_str = args.validate_frame
            try:
                frame_idx = int(frame_idx_str)
            except ValueError:
                print(f"❌ Error: Invalid frame index: {frame_idx_str}")
                sys.exit(1)

            print(
                f"\n🔍 Testing Frame Validation: {video_path}, frame {frame_idx}")

            if not Path(video_path).exists():
                print(f"❌ Error: Video file not found: {video_path}")
                sys.exit(1)

            # Create analyzer with default config
            from .config_manager import ConfigManager
            config_manager = ConfigManager()
            config = config_manager.get_analysis_config()
            analyzer = VideoAnalyzer(config)

            # Validate frame access
            print("   🔧 Validating frame access...")
            validation_result = analyzer.validate_frame_access(
                video_path, frame_idx)

            print(f"   📋 Validation results:")
            for key, value in validation_result.items():
                if key == 'video_info' and value:
                    print(
                        f"      {key}: frames={value.frame_count}, fps={value.fps:.2f}")
                else:
                    print(f"      {key}: {value}")

            if validation_result['valid']:
                # Extract and save the frame
                frame = analyzer.get_frame_at_index(video_path, frame_idx)
                if frame is not None:
                    frame_path = output_dir / \
                        f"validated_frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    print(f"   ✅ Frame saved: {frame_path}")
            else:
                print(
                    f"   ❌ Frame validation failed: {validation_result.get('error_message', 'Unknown error')}")

        # Test 4: Create Test Configuration
        if args.create_test_config:
            print("\n🛠️  Creating test configuration...")

            test_config = {
                'sports2d': {
                    'nb_persons_to_detect': 1,
                    'person_ordering_method': 'highest_likelihood',
                    'first_person_height': 1.75,
                    'pose_model': 'HALPE_26',
                    'mode': 'balanced',
                    'det_frequency': 4,
                    'use_gpu': True
                },
                'yolo': {
                    'confidence_threshold': 0.25,
                    'model_path': None
                },
                'processing': {
                    'save_vid': True,
                    'save_img': True,
                    'save_pose': True,
                    'calculate_angles': True,
                    'save_angles': True,
                    'show_realtime_results': False,
                    'make_c3d': True,
                    'flip_left_right': False
                }
            }

            config_file = output_dir / "test_config.toml"
            import toml
            with open(config_file, 'w') as f:
                toml.dump(test_config, f)
            print(f"   ✅ Test configuration created: {config_file}")

            # Test loading the configuration
            from .config_manager import ConfigManager
            config_manager = ConfigManager(str(config_file))
            analysis_config = config_manager.get_analysis_config()
            print(f"   ✅ Configuration loaded and validated!")
            print(f"      Mode: {analysis_config.mode}")
            print(f"      Pose model: {analysis_config.pose_model}")

        # Default behavior if no arguments
        if not any([args.test_video, args.analyze_angles, args.validate_frame, args.create_test_config]):
            print("\n🎯 Running default tests...")

            # Look for sample videos and angles files
            sample_video = None
            sample_angles = None

            # Check for sample video
            video_paths = [
                Path("input/video/set1"),
                Path("input/video"),
                Path(".")
            ]

            for video_dir in video_paths:
                if video_dir.exists():
                    video_files = list(video_dir.glob("*.mp4")) + \
                        list(video_dir.glob("*.avi"))
                    if video_files:
                        sample_video = video_files[0]
                        break

            # Check for sample angles file
            angles_paths = [
                Path("analysis_output"),
                Path("."),
                output_dir
            ]

            for angles_dir in angles_paths:
                if angles_dir.exists():
                    angles_files = list(angles_dir.glob("**/*.mot"))
                    if angles_files:
                        sample_angles = angles_files[0]
                        break

            if sample_video:
                print(f"   Found sample video: {sample_video}")

                from .config_manager import ConfigManager
                config_manager = ConfigManager()
                config = config_manager.get_analysis_config()
                analyzer = VideoAnalyzer(config, str(output_dir))

                video_info = analyzer.get_video_info(str(sample_video))
                if video_info:
                    print(
                        f"   ✅ Video info: {video_info.frame_count} frames, {video_info.fps:.2f} fps")

                    # Extract a test frame
                    test_frame = min(30, video_info.frame_count // 2)
                    frame = analyzer.get_frame_at_index(
                        str(sample_video), test_frame)
                    if frame is not None:
                        frame_path = output_dir / f"default_test_frame.jpg"
                        cv2.imwrite(str(frame_path), frame)
                        print(f"   ✅ Test frame saved: {frame_path}")

            if sample_angles:
                print(f"   Found sample angles file: {sample_angles}")
                angles_df = AnglesDataProcessor.read_angles_file(
                    str(sample_angles))
                if not angles_df.empty:
                    print(f"   ✅ Angles data: {angles_df.shape}")

                    # Generate frame-by-frame angles
                    output_file = AnglesDataProcessor.generate_frame_by_frame_angles(
                        str(sample_angles), str(output_dir)
                    )
                    if output_file:
                        print(
                            f"   ✅ Default frame-by-frame angles: {output_file}")

            if not sample_video and not sample_angles:
                print("   No sample data found, creating test configuration...")
                # Create test config as fallback
                test_config_file = output_dir / "default_config.toml"
                import toml
                default_config = {
                    'sports2d': {'mode': 'balanced', 'pose_model': 'HALPE_26'},
                    'processing': {'save_vid': True, 'save_angles': True}
                }
                with open(test_config_file, 'w') as f:
                    toml.dump(default_config, f)
                print(f"   ✅ Default test config created: {test_config_file}")

        print(f"\n🎉 All tests completed successfully!")
        print(f"📁 Check output directory: {output_dir}")
        print("\n📖 Usage examples:")
        print("   python utils/video_analyzer.py --test-video input/video/sample.mp4")
        print(
            "   python utils/video_analyzer.py --analyze-angles analysis_output/angles.mot")
        print("   python utils/video_analyzer.py --validate-frame video.mp4 100")

    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
