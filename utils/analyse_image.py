#!/usr/bin/env python3
"""
Cricket Analysis System - Image Processing Utilities

This module provides standalone image processing and frame extraction utilities
for the Cricket Analysis System. It includes classes and methods for:

- Image quality enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Aspect ratio preserving image resizing
- Person region of interest extraction
- Pose overlay visualization
- Video frame extraction and processing
- Frame saving with enhancement options

Classes:
    ImageProcessor: Static methods for image processing operations
    FrameExtractor: Video frame extraction and processing utilities

Dependencies:
    - OpenCV (cv2): Image and video processing
    - NumPy: Numerical operations
    - pathlib: Path handling

Usage:
    Can be run as a standalone module for testing:
    python utils/analyse_image.py --test-image path/to/image.jpg
    
    Or imported and used programmatically:
    from utils.analyse_image import ImageProcessor, FrameExtractor
    
Author: Cricket Analysis System
Version: 1.0
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import argparse
import sys


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


if __name__ == "__main__":
    """
    Test the image processing utilities with command-line arguments.

    Usage examples:
    python utils/analyse_image.py --test-image input/image/cummins1.png
    python utils/analyse_video.py --test-video input/video/set1/sample.mp4
    python utils/analyse_image.py --help
    """
    parser = argparse.ArgumentParser(
        description="Test Cricket Analysis Image Processing Utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python utils/analyse_image.py --test-image input/image/cummins1.png
  python utils/analyse_image.py --test-video input/video/set1/sample.mp4
  python utils/analyse_image.py --create-test-data
        """
    )

    parser.add_argument('--test-image', type=str,
                        help='Test image processing with specified image file')
    parser.add_argument('--test-video', type=str,
                        help='Test frame extraction with specified video file')
    parser.add_argument('--create-test-data', action='store_true',
                        help='Create sample test data for demonstration')
    parser.add_argument('--output-dir', type=str, default='test_output',
                        help='Output directory for test results (default: test_output)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("🏏 Testing Cricket Analysis Image Processing Utilities")
    print("=" * 60)

    try:
        # Test 1: Image Processing
        if args.test_image:
            print(f"\n📷 Testing Image Processing with: {args.test_image}")

            if not Path(args.test_image).exists():
                print(f"❌ Error: Image file not found: {args.test_image}")
                sys.exit(1)

            # Load image
            image = cv2.imread(args.test_image)
            if image is None:
                print(f"❌ Error: Could not load image: {args.test_image}")
                sys.exit(1)

            print(f"   Original image shape: {image.shape}")

            # Test image enhancement
            print("   🔧 Testing image enhancement...")
            enhanced = ImageProcessor.enhance_image_quality(image)
            enhanced_path = output_dir / "enhanced_image.jpg"
            cv2.imwrite(str(enhanced_path), enhanced)
            print(f"   ✅ Enhanced image saved: {enhanced_path}")

            # Test aspect ratio resize
            print("   🔧 Testing aspect ratio resize...")
            resized = ImageProcessor.resize_maintain_aspect_ratio(
                image, (640, 480))
            resized_path = output_dir / "resized_image.jpg"
            cv2.imwrite(str(resized_path), resized)
            print(f"   ✅ Resized image saved: {resized_path}")
            print(f"   New image shape: {resized.shape}")

            # Test pose overlay (with dummy keypoints)
            print("   🔧 Testing pose overlay...")
            dummy_keypoints = {
                'left_shoulder': (100, 150),
                'right_shoulder': (200, 150),
                'left_elbow': (80, 200),
                'right_elbow': (220, 200),
                'left_wrist': (60, 250),
                'right_wrist': (240, 250)
            }
            pose_overlay = ImageProcessor.create_pose_overlay(
                resized, dummy_keypoints)
            overlay_path = output_dir / "pose_overlay.jpg"
            cv2.imwrite(str(overlay_path), pose_overlay)
            print(f"   ✅ Pose overlay saved: {overlay_path}")

        # Test 2: Video Frame Extraction
        if args.test_video:
            print(f"\n🎥 Testing Frame Extraction with: {args.test_video}")

            if not Path(args.test_video).exists():
                print(f"❌ Error: Video file not found: {args.test_video}")
                sys.exit(1)

            try:
                # Initialize frame extractor
                extractor = FrameExtractor(args.test_video)

                print(f"   Video frame count: {extractor.get_frame_count()}")
                print(f"   Video FPS: {extractor.get_fps():.2f}")

                # Test single frame extraction
                print("   🔧 Testing single frame extraction...")
                frame_index = min(100, extractor.get_frame_count() // 2)
                frame = extractor.extract_frame(frame_index)

                if frame is not None:
                    frame_path = output_dir / \
                        f"extracted_frame_{frame_index:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    print(f"   ✅ Frame {frame_index} saved: {frame_path}")
                else:
                    print(f"   ❌ Could not extract frame {frame_index}")

                # Test frame range extraction
                print("   🔧 Testing frame range extraction...")
                start_frame = max(0, frame_index - 5)
                end_frame = min(extractor.get_frame_count() -
                                1, frame_index + 5)
                frames = extractor.extract_frames_range(
                    start_frame, end_frame, step=2)
                print(
                    f"   ✅ Extracted {len(frames)} frames from range {start_frame}-{end_frame}")

                # Test frames around point
                print("   🔧 Testing frames around point...")
                frames_dict = extractor.extract_frames_around_point(
                    frame_index, window_size=3)
                print(
                    f"   ✅ Extracted {len(frames_dict)} frames around frame {frame_index}")

                # Save a few frames
                for i, (idx, frame) in enumerate(list(frames_dict.items())[:3]):
                    frame_path = output_dir / \
                        f"around_point_frame_{idx:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    print(f"   💾 Saved frame {idx}: {frame_path}")

            except Exception as e:
                print(f"   ❌ Error during frame extraction: {e}")

        # Test 3: Create test data
        if args.create_test_data:
            print("\n🛠️  Creating sample test data...")

            # Create a simple test image
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)

            # Add some content
            cv2.rectangle(test_image, (100, 100), (300, 300), (0, 255, 0), -1)
            cv2.circle(test_image, (200, 200), 50, (255, 0, 0), -1)
            cv2.putText(test_image, "Test Image", (150, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            test_image_path = output_dir / "sample_test_image.jpg"
            cv2.imwrite(str(test_image_path), test_image)
            print(f"   ✅ Sample test image created: {test_image_path}")

            # Test with created image
            enhanced = ImageProcessor.enhance_image_quality(test_image)
            enhanced_test_path = output_dir / "enhanced_test_image.jpg"
            cv2.imwrite(str(enhanced_test_path), enhanced)
            print(f"   ✅ Enhanced test image saved: {enhanced_test_path}")

        # Default behavior if no arguments
        if not any([args.test_image, args.test_video, args.create_test_data]):
            print("\n🎯 Running default tests...")

            # Test with existing sample if available
            sample_image_path = Path("input/image/cummins1.png")
            if sample_image_path.exists():
                print(f"   Found sample image: {sample_image_path}")
                image = cv2.imread(str(sample_image_path))
                if image is not None:
                    enhanced = ImageProcessor.enhance_image_quality(image)
                    enhanced_path = output_dir / "default_enhanced.jpg"
                    cv2.imwrite(str(enhanced_path), enhanced)
                    print(
                        f"   ✅ Default enhanced image saved: {enhanced_path}")
            else:
                print("   No sample image found, creating test data...")
                # Create test data as fallback
                test_image = np.random.randint(
                    0, 255, (480, 640, 3), dtype=np.uint8)
                test_path = output_dir / "random_test_image.jpg"
                cv2.imwrite(str(test_path), test_image)
                print(f"   ✅ Random test image created: {test_path}")

        print(f"\n🎉 All tests completed successfully!")
        print(f"📁 Check output directory: {output_dir}")
        print("\n📖 Usage examples:")
        print("   python utils/analyse_image.py --test-image input/image/cummins1.png")
        print("   python utils/analyse_image.py --create-test-data")

    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
