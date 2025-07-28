"""
Core analyzer module for Cricket Analysis System.
Lightweight orchestrator that coordinates all analysis components.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import cv2

from utils.config_manager import ConfigManager
from utils.data_models import AnalysisConfig, AnalysisResult, VideoInfo, YOLODetectionResult
from utils.video_analyzer import VideoAnalyzer
from utils.release_point_detector import ReleasePointDetector
from utils.result_generator import ResultGenerator


class CricketAnalyzer:
    """
    Main analyzer that orchestrates all cricket analysis components.

    This is a lightweight coordinator that delegates work to specialized modules.
    """

    def __init__(self, output_dir: str = "analysis_output", config_path: Optional[str] = None):
        """
        Initialize the cricket analyzer.

        Args:
            output_dir: Directory for analysis outputs
            config_path: Optional path to configuration file
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_analysis_config()

        # Initialize components
        self.video_analyzer = VideoAnalyzer(self.config, str(self.output_dir))
        self.release_detector = ReleasePointDetector(self.config)
        self.result_generator = ResultGenerator(debug_mode=True)

        # State management
        self.analysis_results: Dict[str, AnalysisResult] = {}
        self.release_frames: Dict[str, int] = {}
        self.suggested_frames: Dict[str, YOLODetectionResult] = {}

    def analyze_video(self, video_path: str, video_name: Optional[str] = None) -> AnalysisResult:
        """
        Analyze a single video.

        Args:
            video_path: Path to the video file
            video_name: Optional custom name for the video

        Returns:
            AnalysisResult object
        """
        if video_name is None:
            video_name = Path(video_path).stem

        print(f"Analyzing video: {video_name}")

        # Perform Sports2D analysis
        result = self.video_analyzer.analyze_video(video_path, video_name)
        self.analysis_results[video_name] = result

        # Get AI suggestion for release point
        suggestion = self.get_ai_release_suggestion(video_name)
        if suggestion.release_frame is not None:
            result.suggested_release_frame = suggestion.release_frame
            self.suggested_frames[video_name] = suggestion
            print(
                f"AI suggested release point: Frame {suggestion.release_frame}")

        return result

    def analyze_multiple_videos(self, video_paths: List[str]) -> Dict[str, AnalysisResult]:
        """
        Analyze multiple videos.

        Args:
            video_paths: List of video file paths

        Returns:
            Dictionary mapping video names to AnalysisResult objects
        """
        print(f"Starting analysis of {len(video_paths)} videos...")

        results = {}
        for i, video_path in enumerate(video_paths, 1):
            video_name = Path(video_path).stem
            print(f"Processing video {i}/{len(video_paths)}: {video_name}")

            try:
                result = self.analyze_video(video_path, video_name)
                results[video_name] = result
                print(f"Successfully analyzed: {video_name}")
            except Exception as e:
                print(f"Failed to analyze {video_name}: {e}")
                # Still add a minimal result for consistency
                results[video_name] = AnalysisResult(
                    video_name=video_name,
                    original_path=video_path,
                    analyzed_path="",
                    angles_file="",
                    result_dir=""
                )

        print(f"Analysis complete! Processed {len(results)} videos.")
        return results

    def get_ai_release_suggestion(self, video_name: str) -> YOLODetectionResult:
        """
        Get AI-powered release point suggestion for a video.

        Args:
            video_name: Name of the video

        Returns:
            YOLODetectionResult with suggestion
        """
        if video_name not in self.analysis_results:
            return YOLODetectionResult(video_name, None, None, {'error': 'Video not analyzed'})

        result = self.analysis_results[video_name]

        # Find images directory
        result_dir = Path(result.result_dir)
        sports2d_dir = result_dir / f"{video_name}_Sports2D"
        images_dir = sports2d_dir / f"{video_name}_Sports2D_img"

        if not images_dir.exists():
            return YOLODetectionResult(video_name, None, None, {'error': 'Images directory not found'})

        print(f"Running AI analysis for {video_name}...")
        suggestion = self.release_detector.detect_release_point(
            video_name, str(images_dir))

        return suggestion

    def set_release_frame(self, video_name: str, frame_index: int) -> None:
        """
        Set the release frame for a video (manual override).

        Args:
            video_name: Name of the video
            frame_index: Frame index for release point
        """
        self.release_frames[video_name] = frame_index

        # Update the analysis result if it exists
        if video_name in self.analysis_results:
            self.analysis_results[video_name].manual_release_frame = frame_index

        print(f"Release point set for {video_name}: Frame {frame_index}")

    def generate_csv_analysis(self, output_path: str = "release_point_analysis.csv") -> Optional[str]:
        """
        Generate CSV analysis of all release points.

        Args:
            output_path: Path for the output CSV file

        Returns:
            Path to generated CSV file or None if failed
        """
        if not self.release_frames:
            print("WARNING: No release points set. Cannot generate analysis.")
            return None

        print(
            f"Generating CSV analysis for {len(self.release_frames)} videos...")

        csv_path = self.result_generator.generate_release_point_analysis(
            self.analysis_results,
            self.release_frames,
            output_path
        )

        if csv_path:
            print(f"CSV analysis saved to: {csv_path}")
        else:
            print("Failed to generate CSV analysis")

        return csv_path

    def generate_comparison_report(self, output_path: str = "comparison_report.json") -> Optional[str]:
        """
        Generate detailed comparison report.

        Args:
            output_path: Path for the output JSON file

        Returns:
            Path to generated report file or None if failed
        """
        print("Generating detailed comparison report...")

        report_path = self.result_generator.generate_comparison_report(
            self.analysis_results,
            self.release_frames,
            output_path
        )

        if report_path:
            print(f"Comparison report saved to: {report_path}")
        else:
            print("Failed to generate comparison report")

        return report_path

    def get_video_info(self, video_path: str) -> VideoInfo:
        """
        Get video information.

        Args:
            video_path: Path to the video file

        Returns:
            VideoInfo object
        """
        return self.video_analyzer.get_video_info(video_path)

    def get_frame_at_index(self, video_path: str, frame_index: int) -> Optional[Any]:
        """
        Extract a specific frame from a video.

        Args:
            video_path: Path to the video file
            frame_index: Index of the frame to extract

        Returns:
            Frame as numpy array or None if failed
        """
        return self.video_analyzer.get_frame_at_index(video_path, frame_index)

    def get_analysis_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all analysis performed.

        Returns:
            Dictionary with summary information
        """
        return {
            'total_videos_analyzed': len(self.analysis_results),
            'videos_with_release_points': len(self.release_frames),
            'ai_suggestions_available': len(self.suggested_frames),
            'video_names': list(self.analysis_results.keys()),
            'release_points_set': {name: frame for name, frame in self.release_frames.items()},
            'ai_suggestions': {
                name: result.release_frame
                for name, result in self.suggested_frames.items()
                if result.release_frame is not None
            }
        }

    def export_all_data(self, export_dir: str = "full_export") -> Dict[str, List[str]]:
        """
        Export all analysis data in various formats.

        Args:
            export_dir: Directory for exports

        Returns:
            Dictionary with exported file paths by type
        """
        export_path = Path(export_dir)
        export_path.mkdir(exist_ok=True)

        exported = {
            'csv_files': [],
            'json_files': [],
            'angles_files': []
        }

        # Generate main CSV analysis
        if self.release_frames:
            csv_path = self.generate_csv_analysis(
                str(export_path / "release_point_analysis.csv"))
            if csv_path:
                exported['csv_files'].append(csv_path)

        # Generate comparison report
        json_path = self.generate_comparison_report(
            str(export_path / "comparison_report.json"))
        if json_path:
            exported['json_files'].append(json_path)

        # Export individual angles files
        angles_files = self.result_generator.export_angles_data(
            self.analysis_results,
            str(export_path / "angles_data")
        )
        exported['angles_files'].extend(angles_files)

        print(f"Full export completed to: {export_dir}")
        print(f"  CSV files: {len(exported['csv_files'])}")
        print(f"  JSON files: {len(exported['json_files'])}")
        print(f"  Angles files: {len(exported['angles_files'])}")

        return exported

    def load_existing_analysis_results(self, base_dir: str = ".") -> Dict[str, AnalysisResult]:
        """
        Load existing analysis results from bowl directories.

        Args:
            base_dir: Base directory to search for bowl* directories

        Returns:
            Dictionary of loaded analysis results
        """
        base_path = Path(base_dir)
        loaded_results = {}

        # Look for bowl* directories
        bowl_dirs = list(base_path.glob("bowl*"))

        for bowl_dir in bowl_dirs:
            if bowl_dir.is_dir():
                try:
                    # Look for Sports2D subdirectory
                    sports2d_dirs = list(bowl_dir.glob("*_Sports2D"))

                    if sports2d_dirs:
                        sports2d_dir = sports2d_dirs[0]
                        video_name = bowl_dir.name

                        # Find the analyzed video file
                        analyzed_video = sports2d_dir / \
                            f"{sports2d_dir.name}.mp4"

                        # Find the angles file
                        angles_files = list(
                            sports2d_dir.glob("*_angles_person00.mot"))
                        angles_file = angles_files[0] if angles_files else None

                        # Create analysis result
                        result = AnalysisResult(
                            video_name=video_name,
                            result_dir=str(sports2d_dir),
                            analyzed_path=str(
                                analyzed_video) if analyzed_video.exists() else None,
                            angles_file=str(
                                angles_file) if angles_file and angles_file.exists() else None,
                            success=True
                        )

                        loaded_results[video_name] = result
                        self.analysis_results[video_name] = result

                        print(f"Loaded existing analysis for {video_name}")

                except Exception as e:
                    print(f"Error loading analysis for {bowl_dir.name}: {e}")

        print(f"Loaded {len(loaded_results)} existing analysis results")
        return loaded_results
