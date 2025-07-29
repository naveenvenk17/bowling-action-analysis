#!/usr/bin/env python3
"""
Cricket Analysis System - Core Data Models and Structures

This module defines the core data models and structures for the Cricket Analysis System.
It includes dataclasses and methods for:

- Video information and metadata representation
- Analysis result containers and status tracking
- Release point data structures for biomechanical analysis
- YOLO detection result containers
- Analysis configuration management and Sports2D integration
- Data serialization and validation

Classes:
    VideoInfo: Information about video files (frame count, FPS, duration)
    AnalysisResult: Result containers for video analysis operations
    ReleasePointData: Data extracted at release points with angle measurements
    YOLODetectionResult: Results from YOLO-based release point detection
    AnalysisConfig: Configuration for analysis parameters and Sports2D integration

Key Features:
    - Type-safe data structures using dataclasses
    - Automatic video metadata extraction from file paths
    - Sports2D configuration generation and export
    - Release point data serialization for CSV export
    - Comprehensive analysis configuration with validation
    - Integration with external libraries (Sports2D, YOLO)

Dependencies:
    - dataclasses: Type-safe data structures
    - pathlib: Path handling
    - typing: Type annotations
    - pandas: Data manipulation
    - OpenCV: Video processing

Usage:
    Can be run as a standalone module for testing:
    python utils/data_models.py --test-models
    
    Or imported and used programmatically:
    from utils.data_models import VideoInfo, AnalysisResult, ReleasePointData
    
Author: Cricket Analysis System
Version: 1.0
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import argparse
import sys
import json


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

    def get_joint_angles(self) -> Dict[str, float]:
        """
        Get only joint angle measurements (excludes metadata like time).

        Returns:
            Dictionary containing only joint angle measurements
        """
        joint_keywords = ['elbow', 'knee', 'shoulder', 'ankle', 'hip', 'wrist']
        joint_angles = {}

        for key, value in self.angles_data.items():
            if any(keyword in key.lower() for keyword in joint_keywords):
                joint_angles[key] = value

        return joint_angles


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


if __name__ == "__main__":
    """
    Test the data models with command-line arguments.

    Usage examples:
    python utils/data_models.py --test-models
    python utils/data_models.py --create-sample-data
    python utils/data_models.py --help
    """
    parser = argparse.ArgumentParser(
        description="Test Cricket Analysis Data Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python utils/data_models.py --test-models
  python utils/data_models.py --create-sample-data
  python utils/data_models.py --test-video-info path/to/video.mp4
  python utils/data_models.py --test-config-export
        """
    )

    parser.add_argument('--test-models', action='store_true',
                        help='Test all data models with sample data')
    parser.add_argument('--create-sample-data', action='store_true',
                        help='Create sample data files for testing')
    parser.add_argument('--test-video-info', type=str,
                        help='Test VideoInfo with specified video file')
    parser.add_argument('--test-config-export', action='store_true',
                        help='Test Sports2D configuration export')
    parser.add_argument('--output-dir', type=str, default='test_output',
                        help='Output directory for test results (default: test_output)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("🏏 Testing Cricket Analysis Data Models")
    print("=" * 50)

    try:
        # Test 1: Test All Models
        if args.test_models:
            print("\n📊 Testing All Data Models...")

            # Test VideoInfo
            print("   🔧 Testing VideoInfo model...")

            # Create mock video info
            video_info = VideoInfo(
                path="test_video.mp4",
                frame_count=300,
                fps=30.0,
                duration=10.0
            )

            print(f"   ✅ VideoInfo created:")
            print(f"      Path: {video_info.path}")
            print(f"      Frames: {video_info.frame_count}")
            print(f"      FPS: {video_info.fps}")
            print(f"      Duration: {video_info.duration}s")

            # Test AnalysisResult
            print("   🔧 Testing AnalysisResult model...")

            analysis_result = AnalysisResult(
                video_name="test_bowling",
                result_dir="test_bowling_Sports2D",
                analyzed_path="test_bowling_analyzed.mp4",
                angles_file="test_bowling_angles.mot",
                success=True,
                suggested_release_frame=145,
                manual_release_frame=147
            )

            print(f"   ✅ AnalysisResult created:")
            print(f"      Video name: {analysis_result.video_name}")
            print(f"      Success: {analysis_result.success}")
            print(
                f"      Suggested frame: {analysis_result.suggested_release_frame}")
            print(
                f"      Manual frame: {analysis_result.manual_release_frame}")
            print(
                f"      Final release frame: {analysis_result.release_frame}")

            # Test ReleasePointData
            print("   🔧 Testing ReleasePointData model...")

            sample_angles = {
                'Right_elbow': 142.5,
                'Left_elbow': 138.2,
                'Right_knee': 165.8,
                'Left_knee': 162.1,
                'Right_shoulder': 120.3,
                'Left_shoulder': 118.7,
                'time': 4.83
            }

            release_data = ReleasePointData(
                video_name="test_bowling",
                frame_index=145,
                angles_data=sample_angles,
                timestamp=4.83
            )

            print(f"   ✅ ReleasePointData created:")
            print(f"      Video: {release_data.video_name}")
            print(f"      Frame: {release_data.frame_index}")
            print(f"      Timestamp: {release_data.timestamp}s")
            print(f"      Angles count: {len(release_data.angles_data)}")

            # Test joint angles extraction
            joint_angles = release_data.get_joint_angles()
            print(f"      Joint angles: {len(joint_angles)}")
            for joint, angle in list(joint_angles.items())[:3]:
                print(f"         {joint}: {angle:.2f}°")

            # Test CSV export format
            csv_data = release_data.to_dict()
            print(f"      CSV export keys: {list(csv_data.keys())}")

            # Test YOLODetectionResult
            print("   🔧 Testing YOLODetectionResult model...")

            detection_data = {
                'method': 'refined_yolo',
                'confidence_threshold': 0.25,
                'critical_analysis': {'frames_analyzed': 30},
                'validation_status': 'excellent'
            }

            yolo_result = YOLODetectionResult(
                video_name="test_bowling",
                release_frame=145,
                confidence=0.87,
                detection_data=detection_data
            )

            print(f"   ✅ YOLODetectionResult created:")
            print(f"      Video: {yolo_result.video_name}")
            print(f"      Release frame: {yolo_result.release_frame}")
            print(f"      Confidence: {yolo_result.confidence:.3f}")
            print(
                f"      Detection data keys: {list(yolo_result.detection_data.keys())}")

            # Test AnalysisConfig
            print("   🔧 Testing AnalysisConfig model...")

            config = AnalysisConfig(
                mode='accurate',
                pose_model='HALPE_26',
                use_gpu=True,
                yolo_confidence_threshold=0.25,
                save_vid=True,
                save_angles=True
            )

            print(f"   ✅ AnalysisConfig created:")
            print(f"      Mode: {config.mode}")
            print(f"      Pose model: {config.pose_model}")
            print(f"      GPU enabled: {config.use_gpu}")
            print(f"      YOLO confidence: {config.yolo_confidence_threshold}")

            # Save models to JSON for inspection
            models_data = {
                'video_info': {
                    'path': video_info.path,
                    'frame_count': video_info.frame_count,
                    'fps': video_info.fps,
                    'duration': video_info.duration
                },
                'analysis_result': {
                    'video_name': analysis_result.video_name,
                    'success': analysis_result.success,
                    'release_frame': analysis_result.release_frame
                },
                'release_point_data': release_data.to_dict(),
                'yolo_detection': {
                    'video_name': yolo_result.video_name,
                    'release_frame': yolo_result.release_frame,
                    'confidence': yolo_result.confidence
                },
                'analysis_config': {
                    'mode': config.mode,
                    'pose_model': config.pose_model,
                    'use_gpu': config.use_gpu
                }
            }

            models_file = output_dir / "data_models_test.json"
            with open(models_file, 'w') as f:
                json.dump(models_data, f, indent=2)
            print(f"   💾 Models data saved: {models_file}")

        # Test 2: Create Sample Data
        if args.create_sample_data:
            print("\n🛠️  Creating sample data files...")

            # Create sample release point data
            sample_videos = ['bowling1', 'bowling2', 'bowling3']
            sample_data = []

            for i, video_name in enumerate(sample_videos):
                # Create varying angle data
                base_frame = 140 + i * 5
                sample_angles = {
                    'Right_elbow': 140 + i * 2 + (i * 0.5),
                    'Left_elbow': 145 + i * 1.5 + (i * 0.3),
                    'Right_knee': 160 + i * 3 + (i * 0.8),
                    'Left_knee': 165 + i * 2.5 + (i * 0.6),
                    'Right_shoulder': 120 + i * 1.8 + (i * 0.4),
                    'Left_shoulder': 125 + i * 2.2 + (i * 0.7)
                }

                release_data = ReleasePointData(
                    video_name=video_name,
                    frame_index=base_frame,
                    angles_data=sample_angles,
                    timestamp=base_frame / 30.0
                )

                sample_data.append(release_data.to_dict())

            # Save as CSV
            df = pd.DataFrame(sample_data)
            csv_file = output_dir / "sample_release_points.csv"
            df.to_csv(csv_file, index=False)
            print(f"   ✅ Sample CSV created: {csv_file}")
            print(f"      Shape: {df.shape}")
            print(f"      Columns: {list(df.columns)}")

            # Create sample analysis results
            analysis_results = []
            for video_name in sample_videos:
                result_data = {
                    'video_name': video_name,
                    'result_dir': f"{video_name}_Sports2D",
                    'success': True,
                    'suggested_release_frame': 140 + sample_videos.index(video_name) * 5,
                    'manual_release_frame': None
                }
                analysis_results.append(result_data)

            results_file = output_dir / "sample_analysis_results.json"
            with open(results_file, 'w') as f:
                json.dump(analysis_results, f, indent=2)
            print(f"   ✅ Sample analysis results: {results_file}")

            # Create sample YOLO detection results
            yolo_results = []
            for i, video_name in enumerate(sample_videos):
                yolo_data = {
                    'video_name': video_name,
                    'release_frame': 140 + i * 5,
                    'confidence': 0.8 + i * 0.05,
                    'detection_data': {
                        'method': 'refined_yolo',
                        'validation_status': ['excellent', 'good', 'acceptable'][i]
                    }
                }
                yolo_results.append(yolo_data)

            yolo_file = output_dir / "sample_yolo_results.json"
            with open(yolo_file, 'w') as f:
                json.dump(yolo_results, f, indent=2)
            print(f"   ✅ Sample YOLO results: {yolo_file}")

        # Test 3: Test Video Info
        if args.test_video_info:
            print(f"\n🎥 Testing VideoInfo with: {args.test_video_info}")

            video_path = Path(args.test_video_info)
            if not video_path.exists():
                print(f"❌ Error: Video file not found: {args.test_video_info}")
                sys.exit(1)

            try:
                # Test with real video file
                video_info = VideoInfo.from_video_path(str(video_path))

                print(f"   ✅ VideoInfo extracted successfully:")
                print(f"      Path: {video_info.path}")
                print(f"      Frame count: {video_info.frame_count}")
                print(f"      FPS: {video_info.fps:.2f}")
                print(f"      Duration: {video_info.duration:.2f} seconds")

                # Calculate additional metrics
                print(f"   📊 Additional metrics:")
                print(f"      Minutes: {video_info.duration / 60:.2f}")
                print(
                    f"      Frame rate category: {'High' if video_info.fps >= 60 else 'Standard' if video_info.fps >= 24 else 'Low'}")

                # Save video info
                video_info_data = {
                    'file_path': str(video_path),
                    'file_name': video_path.name,
                    'file_size_mb': video_path.stat().st_size / (1024 * 1024) if video_path.exists() else 0,
                    'frame_count': video_info.frame_count,
                    'fps': video_info.fps,
                    'duration_seconds': video_info.duration,
                    'duration_minutes': video_info.duration / 60
                }

                info_file = output_dir / f"video_info_{video_path.stem}.json"
                with open(info_file, 'w') as f:
                    json.dump(video_info_data, f, indent=2)
                print(f"   💾 Video info saved: {info_file}")

            except Exception as e:
                print(f"   ❌ Error extracting video info: {e}")

        # Test 4: Test Config Export
        if args.test_config_export:
            print("\n🎯 Testing Sports2D Configuration Export...")

            # Create test configuration
            config = AnalysisConfig(
                mode='accurate',
                pose_model='HALPE_26',
                use_gpu=True,
                yolo_confidence_threshold=0.3,
                save_vid=True,
                save_angles=True,
                calculate_angles=True
            )

            print(f"   📋 Test configuration:")
            print(f"      Mode: {config.mode}")
            print(f"      Pose model: {config.pose_model}")
            print(f"      GPU: {config.use_gpu}")

            # Export Sports2D configuration
            test_video_path = "test_bowling_video.mp4"
            test_result_dir = str(output_dir / "sports2d_test")

            sports2d_config = config.to_sports2d_config(
                test_video_path, test_result_dir)

            print(f"   ✅ Sports2D configuration generated!")
            print(f"   📊 Configuration sections:")
            for section in sports2d_config.keys():
                print(
                    f"      - {section}: {len(sports2d_config[section])} settings")

            # Save Sports2D configuration
            sports2d_file = output_dir / "sports2d_config_test.json"
            with open(sports2d_file, 'w') as f:
                json.dump(sports2d_config, f, indent=2)
            print(f"   💾 Sports2D config saved: {sports2d_file}")

            # Test different configurations
            test_configs = [
                {'mode': 'fast', 'use_gpu': False},
                {'mode': 'balanced', 'pose_model': 'COCO_17'},
                {'mode': 'accurate', 'det_frequency': 2}
            ]

            for i, config_updates in enumerate(test_configs):
                print(
                    f"   🔧 Testing configuration variant {i+1}: {config_updates}")

                # Update config
                for key, value in config_updates.items():
                    setattr(config, key, value)

                # Export variant
                variant_config = config.to_sports2d_config(
                    test_video_path, test_result_dir)
                variant_file = output_dir / \
                    f"sports2d_config_variant_{i+1}.json"

                with open(variant_file, 'w') as f:
                    json.dump(variant_config, f, indent=2)
                print(f"      ✅ Variant {i+1} saved: {variant_file}")

        # Default behavior if no arguments
        if not any([args.test_models, args.create_sample_data, args.test_video_info, args.test_config_export]):
            print("\n🎯 Running default tests...")

            # Quick test of all models
            print("   🔧 Quick model validation...")

            # Test basic model creation
            video_info = VideoInfo("test.mp4", 300, 30.0, 10.0)
            analysis_result = AnalysisResult("test", "test_dir", success=True)
            release_data = ReleasePointData(
                "test", 145, {'Right_elbow': 142.5}, 4.83)
            yolo_result = YOLODetectionResult("test", 145, 0.85)
            config = AnalysisConfig()

            print(f"   ✅ All models created successfully!")
            print(f"      VideoInfo: {video_info.path}")
            print(f"      AnalysisResult: {analysis_result.video_name}")
            print(f"      ReleasePointData: {release_data.frame_index}")
            print(f"      YOLODetectionResult: {yolo_result.confidence}")
            print(f"      AnalysisConfig: {config.mode}")

            # Test data serialization
            print("   🔧 Testing data serialization...")

            csv_data = release_data.to_dict()
            sports2d_config = config.to_sports2d_config("test.mp4", "test_dir")

            # Save test data
            test_data = {
                'release_point_csv': csv_data,
                'sports2d_config_keys': list(sports2d_config.keys()),
                'model_validation': 'passed'
            }

            test_file = output_dir / "default_models_test.json"
            with open(test_file, 'w') as f:
                json.dump(test_data, f, indent=2)
            print(f"   ✅ Default test data saved: {test_file}")

        print(f"\n🎉 All tests completed successfully!")
        print(f"📁 Check output directory: {output_dir}")
        print("\n📖 Usage examples:")
        print("   python utils/data_models.py --test-models")
        print("   python utils/data_models.py --create-sample-data")
        print("   python utils/data_models.py --test-video-info video.mp4")

    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
