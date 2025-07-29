#!/usr/bin/env python3
"""
Cricket Analysis System - Result Generation and Data Export

This module provides comprehensive result generation and data export utilities
for the Cricket Analysis System. It includes classes and methods for:

- CSV export and data analysis of release point measurements
- Comprehensive comparison reports in JSON format
- Angle data processing and export
- Summary statistics calculation and reporting
- Debug information logging and analysis
- Multi-video comparison and analysis

Classes:
    ResultGenerator: Generates analysis results and exports data in various formats

Key Features:
    - Release point CSV analysis generation with comprehensive data
    - Multi-video comparison reports with statistical analysis
    - Angle data export in multiple formats
    - Summary statistics calculation for biomechanical analysis
    - Debug mode with detailed logging and error reporting
    - Metadata integration for enhanced analysis context

Dependencies:
    - pandas: Data manipulation and CSV export
    - pathlib: Path handling
    - json: Data serialization
    - datetime: Timestamp generation

Usage:
    Can be run as a standalone module for testing:
    python utils/result_generator.py --test-data analysis_output/
    
    Or imported and used programmatically:
    from utils.result_generator import ResultGenerator
    
Author: Cricket Analysis System
Version: 1.0
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
from datetime import datetime
import argparse
import sys
import numpy as np

from .data_models import AnalysisResult, ReleasePointData
from .video_analyzer import AnglesDataProcessor


class ResultGenerator:
    """Generates analysis results and exports data in various formats."""

    def __init__(self, debug_mode: bool = False):
        """
        Initialize the result generator.

        Args:
            debug_mode: Whether to generate debug information
        """
        self.debug_mode = debug_mode
        self.debug_info = []

    def generate_release_point_analysis(
        self,
        analysis_results: Dict[str, AnalysisResult],
        release_frames: Dict[str, int],
        output_path: str = "release_point_analysis.csv"
    ) -> Optional[str]:
        """
        Generate comprehensive CSV analysis of release points.

        Args:
            analysis_results: Dictionary of analysis results by video name
            release_frames: Dictionary of release frame indices by video name
            output_path: Path for the output CSV file

        Returns:
            Path to generated CSV file or None if failed
        """
        self.debug_info = []
        all_data = []

        for video_name, frame_index in release_frames.items():
            self._log_debug(f"Processing {video_name}, frame {frame_index}")

            if video_name not in analysis_results:
                self._log_debug(
                    f"  ERROR: {video_name} not in analysis_results")
                continue

            analysis_result = analysis_results[video_name]
            angles_file = analysis_result.angles_file

            self._log_debug(f"  Reading angles file: {angles_file}")

            # Read angle data
            angles_df = AnglesDataProcessor.read_angles_file(angles_file)
            self._log_debug(f"  Angles DataFrame shape: {angles_df.shape}")

            if angles_df.empty:
                self._log_debug(f"  ERROR: Empty DataFrame for {video_name}")
                continue

            if frame_index >= len(angles_df):
                self._log_debug(
                    f"  ERROR: Frame index {frame_index} >= DataFrame length {len(angles_df)}")
                continue

            # Extract frame data
            frame_angles = AnglesDataProcessor.extract_frame_data(
                angles_df, frame_index)

            # Calculate timestamp (if time column exists)
            timestamp = self._calculate_timestamp(angles_df, frame_index)

            # Create release point data
            release_data = ReleasePointData(
                video_name=video_name,
                frame_index=frame_index,
                angles_data=frame_angles,
                timestamp=timestamp
            )

            all_data.append(release_data.to_dict())
            self._log_debug(f"  SUCCESS: Added data for {video_name}")

        # Write debug information if enabled
        if self.debug_mode:
            debug_path = Path(output_path).parent / "csv_generation_debug.txt"
            self._write_debug_info(debug_path)

        # Generate CSV
        if all_data:
            df = pd.DataFrame(all_data)

            # Reorder columns for better readability
            df = self._reorder_columns(df)

            # Add metadata
            df = self._add_metadata(df, analysis_results)

            # Save to CSV
            df.to_csv(output_path, index=False)
            return output_path

        return None

    def generate_comparison_report(
        self,
        analysis_results: Dict[str, AnalysisResult],
        release_frames: Dict[str, int],
        output_path: str = "comparison_report.json"
    ) -> Optional[str]:
        """
        Generate a detailed comparison report of all analyzed videos.

        Args:
            analysis_results: Dictionary of analysis results by video name
            release_frames: Dictionary of release frame indices by video name
            output_path: Path for the output JSON file

        Returns:
            Path to generated report file or None if failed
        """
        report_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_videos': len(analysis_results),
                'videos_with_release_points': len(release_frames)
            },
            'videos': {},
            'summary_statistics': {}
        }

        all_release_data = []

        # Process each video
        for video_name, result in analysis_results.items():
            video_report = {
                'analysis_result': {
                    'original_path': result.original_path,
                    'analyzed_path': result.analyzed_path,
                    'angles_file': result.angles_file,
                    'suggested_release_frame': result.suggested_release_frame,
                    'manual_release_frame': result.manual_release_frame,
                    'final_release_frame': result.release_frame
                },
                'release_point_data': None
            }

            # Add release point data if available
            if video_name in release_frames:
                frame_index = release_frames[video_name]
                angles_df = AnglesDataProcessor.read_angles_file(
                    result.angles_file)

                if not angles_df.empty and frame_index < len(angles_df):
                    frame_data = AnglesDataProcessor.extract_frame_data(
                        angles_df, frame_index)
                    timestamp = self._calculate_timestamp(
                        angles_df, frame_index)

                    release_data = ReleasePointData(
                        video_name=video_name,
                        frame_index=frame_index,
                        angles_data=frame_data,
                        timestamp=timestamp
                    )

                    video_report['release_point_data'] = release_data.to_dict()
                    all_release_data.append(release_data)

            report_data['videos'][video_name] = video_report

        # Generate summary statistics
        if all_release_data:
            report_data['summary_statistics'] = self._generate_summary_statistics(
                all_release_data)

        # Save report
        try:
            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            return output_path
        except Exception as e:
            print(f"Error saving comparison report: {e}")
            return None

    def export_angles_data(
        self,
        analysis_results: Dict[str, AnalysisResult],
        output_dir: str = "angles_export"
    ) -> List[str]:
        """
        Export all angles data as separate CSV files.

        Args:
            analysis_results: Dictionary of analysis results by video name
            output_dir: Directory to save exported files

        Returns:
            List of exported file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        exported_files = []

        for video_name, result in analysis_results.items():
            angles_df = AnglesDataProcessor.read_angles_file(
                result.angles_file)

            if not angles_df.empty:
                export_file = output_path / f"{video_name}_angles.csv"
                angles_df.to_csv(export_file, index=False)
                exported_files.append(str(export_file))

        return exported_files

    def _calculate_timestamp(self, angles_df: pd.DataFrame, frame_index: int) -> float:
        """Calculate timestamp for a frame."""
        try:
            # Look for time column (usually first column)
            time_columns = ['time', 'Time', 'frame_time', 'timestamp']

            for col in time_columns:
                if col in angles_df.columns:
                    return float(angles_df.iloc[frame_index][col])

            # If no time column, assume 30 FPS
            return frame_index / 30.0

        except:
            return frame_index / 30.0

    def _reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reorder columns for better readability."""
        # Define preferred column order
        priority_columns = ['video_name', 'release_frame', 'timestamp']

        # Get remaining columns
        other_columns = [
            col for col in df.columns if col not in priority_columns]

        # Reorder
        new_order = priority_columns + sorted(other_columns)
        return df[new_order]

    def _add_metadata(self, df: pd.DataFrame, analysis_results: Dict[str, AnalysisResult]) -> pd.DataFrame:
        """Add metadata columns to the dataframe."""
        # Add analysis metadata
        for video_name in df['video_name']:
            if video_name in analysis_results:
                result = analysis_results[video_name]
                mask = df['video_name'] == video_name

                # Add suggested vs manual release frame info
                if result.suggested_release_frame is not None:
                    df.loc[mask, 'ai_suggested_frame'] = result.suggested_release_frame
                if result.manual_release_frame is not None:
                    df.loc[mask, 'manually_set_frame'] = result.manual_release_frame

        return df

    def _generate_summary_statistics(self, release_data_list: List[ReleasePointData]) -> Dict[str, Any]:
        """Generate summary statistics from release point data."""
        if not release_data_list:
            return {}

        # Create DataFrame for analysis
        df = pd.DataFrame([data.to_dict() for data in release_data_list])

        # Extract numeric columns (angle measurements)
        numeric_columns = df.select_dtypes(
            include=['float64', 'int64']).columns
        angle_columns = [col for col in numeric_columns if col not in [
            'release_frame', 'timestamp']]

        summary = {
            'total_samples': len(release_data_list),
            'angle_statistics': {}
        }

        # Calculate statistics for each angle
        for col in angle_columns:
            if col in df.columns:
                values = df[col].dropna()
                if not values.empty:
                    summary['angle_statistics'][col] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'median': float(values.median()),
                        'count': len(values)
                    }

        # Add timestamp statistics
        if 'timestamp' in df.columns:
            timestamps = df['timestamp'].dropna()
            if not timestamps.empty:
                summary['timing_statistics'] = {
                    'mean_release_time': float(timestamps.mean()),
                    'std_release_time': float(timestamps.std()),
                    'earliest_release': float(timestamps.min()),
                    'latest_release': float(timestamps.max())
                }

        return summary

    def _log_debug(self, message: str) -> None:
        """Log debug information."""
        if self.debug_mode:
            self.debug_info.append(message)

    def _write_debug_info(self, debug_path: Path) -> None:
        """Write debug information to file."""
        try:
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write("\n".join(self.debug_info))
        except Exception as e:
            print(f"Warning: Could not write debug info to {debug_path}: {e}")


if __name__ == "__main__":
    """
    Test the result generator with command-line arguments.

    Usage examples:
    python utils/result_generator.py --test-data analysis_output/
    python utils/result_generator.py --create-test-data
    python utils/result_generator.py --generate-csv sample_results.json
    python utils/result_generator.py --compare-videos video1 video2 video3
        """
    parser = argparse.ArgumentParser(
        description="Test Cricket Analysis Result Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python utils/result_generator.py --test-data analysis_output/
  python utils/result_generator.py --create-test-data
  python utils/result_generator.py --generate-csv results.json
  python utils/result_generator.py --compare-videos video1 video2 video3
        """
    )

    parser.add_argument('--test-data', type=str,
                        help='Test result generation with data from specified directory')
    parser.add_argument('--create-test-data', action='store_true',
                        help='Create sample test data for demonstration')
    parser.add_argument('--generate-csv', type=str,
                        help='Generate CSV from JSON results file')
    parser.add_argument('--compare-videos', nargs='+',
                        help='Generate comparison report for specified videos')
    parser.add_argument('--output-dir', type=str, default='test_output',
                        help='Output directory for test results (default: test_output)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with detailed logging')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("🏏 Testing Cricket Analysis Result Generator")
    print("=" * 60)

    try:
        # Initialize result generator
        generator = ResultGenerator(debug_mode=args.debug)

        # Test 1: Test with Real Data
        if args.test_data:
            print(f"\n📊 Testing Result Generation with: {args.test_data}")

            data_dir = Path(args.test_data)
            if not data_dir.exists():
                print(f"❌ Error: Data directory not found: {args.test_data}")
                sys.exit(1)

            # Look for Sports2D output directories
            sports2d_dirs = list(data_dir.glob("*_Sports2D"))
            if not sports2d_dirs:
                print(
                    f"   ⚠️  No Sports2D output directories found in {args.test_data}")
                print(f"   Looking for directories matching pattern: *_Sports2D")
                sys.exit(1)

            print(
                f"   Found {len(sports2d_dirs)} Sports2D output directories:")
            for dir_path in sports2d_dirs:
                print(f"      - {dir_path.name}")

            # Create mock analysis results and release frames
            analysis_results = {}
            release_frames = {}

            for sports2d_dir in sports2d_dirs:
                video_name = sports2d_dir.name.replace('_Sports2D', '')

                # Look for angles file
                angles_files = list(sports2d_dir.glob("*_angles_person00.mot"))
                if not angles_files:
                    angles_files = list(sports2d_dir.glob("*.mot"))

                if angles_files:
                    angles_file = angles_files[0]
                    print(f"   📋 Processing {video_name}: {angles_file.name}")

                    # Create analysis result
                    analysis_results[video_name] = AnalysisResult(
                        video_name=video_name,
                        result_dir=str(sports2d_dir),
                        angles_file=str(angles_file),
                        analyzed_path=str(
                            sports2d_dir / f"{video_name}_Sports2D.mp4"),
                        success=True
                    )

                    # Use a sample release frame (145 is typical)
                    release_frames[video_name] = 145
                else:
                    print(f"   ⚠️  No angles file found for {video_name}")

            if analysis_results:
                # Generate CSV analysis
                print("   🔧 Generating release point CSV analysis...")
                csv_path = generator.generate_release_point_analysis(
                    analysis_results,
                    release_frames,
                    str(output_dir / "release_point_analysis.csv")
                )

                if csv_path:
                    print(f"   ✅ CSV analysis generated: {csv_path}")

                    # Show CSV preview
                    df = pd.read_csv(csv_path)
                    print(f"   📋 CSV Preview (shape: {df.shape}):")
                    # Show first 10 columns
                    print(f"      Columns: {list(df.columns)[:10]}...")
                    if not df.empty:
                        print(f"      Sample row:")
                        # Show first 5 columns
                        for col in list(df.columns)[:5]:
                            print(f"         {col}: {df.iloc[0][col]}")
                else:
                    print(f"   ❌ Failed to generate CSV analysis")

                # Generate comparison report
                print("   🔧 Generating comparison report...")
                report_path = generator.generate_comparison_report(
                    analysis_results,
                    release_frames,
                    str(output_dir / "comparison_report.json")
                )

                if report_path:
                    print(f"   ✅ Comparison report generated: {report_path}")

                    # Show report preview
                    with open(report_path, 'r') as f:
                        report_data = json.load(f)

                    print(f"   📋 Report Preview:")
                    print(
                        f"      Total videos: {report_data['metadata']['total_videos']}")
                    print(
                        f"      Videos with release points: {report_data['metadata']['videos_with_release_points']}")

                    if 'summary_statistics' in report_data:
                        stats = report_data['summary_statistics']
                        print(
                            f"      Summary statistics available: {len(stats)}")
                else:
                    print(f"   ❌ Failed to generate comparison report")

                # Export angles data
                print("   🔧 Exporting angles data...")
                angles_dir = output_dir / "angles_export"
                exported_files = generator.export_angles_data(
                    analysis_results, str(angles_dir))

                print(f"   ✅ Exported {len(exported_files)} angles files:")
                for file_path in exported_files:
                    print(f"      - {Path(file_path).name}")
            else:
                print(f"   ❌ No valid analysis results found")

        # Test 2: Create Test Data
        if args.create_test_data:
            print("\n🛠️  Creating sample test data...")

            # Create sample angles data
            test_data_dir = output_dir / "sample_data"
            test_data_dir.mkdir(exist_ok=True)

            # Create sample angle data
            sample_angles = {
                'time': [i/30.0 for i in range(200)],  # 200 frames at 30fps
                'Right_elbow': [140 + 10*np.sin(i*0.1) for i in range(200)],
                'Left_elbow': [145 + 8*np.cos(i*0.1) for i in range(200)],
                'Right_knee': [160 + 5*np.sin(i*0.05) for i in range(200)],
                'Left_knee': [165 + 5*np.cos(i*0.05) for i in range(200)],
                'Right_shoulder': [120 + 15*np.sin(i*0.08) for i in range(200)],
                'Left_shoulder': [125 + 12*np.cos(i*0.08) for i in range(200)]
            }

            # Create sample videos
            sample_videos = ['sample_video1', 'sample_video2', 'sample_video3']
            analysis_results = {}
            release_frames = {}

            for video_name in sample_videos:
                # Create Sports2D directory structure
                sports2d_dir = test_data_dir / f"{video_name}_Sports2D"
                sports2d_dir.mkdir(exist_ok=True)

                # Create angles file
                angles_file = sports2d_dir / \
                    f"{video_name}_Sports2D_angles_person00.mot"

                # Create .mot file content
                mot_content = """name sample_angles
datacolumns 7
datarows 200
range 0 6.666666666666667
endheader
time	Right_elbow	Left_elbow	Right_knee	Left_knee	Right_shoulder	Left_shoulder
"""

                for i in range(200):
                    row = [
                        f"{sample_angles['time'][i]:.6f}",
                        f"{sample_angles['Right_elbow'][i]:.6f}",
                        f"{sample_angles['Left_elbow'][i]:.6f}",
                        f"{sample_angles['Right_knee'][i]:.6f}",
                        f"{sample_angles['Left_knee'][i]:.6f}",
                        f"{sample_angles['Right_shoulder'][i]:.6f}",
                        f"{sample_angles['Left_shoulder'][i]:.6f}"
                    ]
                    mot_content += "\t".join(row) + "\n"

                with open(angles_file, 'w') as f:
                    f.write(mot_content)

                # Create analysis result
                analysis_results[video_name] = AnalysisResult(
                    video_name=video_name,
                    result_dir=str(sports2d_dir),
                    angles_file=str(angles_file),
                    analyzed_path=str(
                        sports2d_dir / f"{video_name}_Sports2D.mp4"),
                    success=True
                )

                # Random release frame around 145
                import random
                release_frames[video_name] = 145 + random.randint(-10, 10)

            print(f"   ✅ Created sample data for {len(sample_videos)} videos")

            # Test with created data
            print("   🔧 Testing with created sample data...")

            # Generate CSV
            csv_path = generator.generate_release_point_analysis(
                analysis_results,
                release_frames,
                str(output_dir / "sample_release_analysis.csv")
            )

            if csv_path:
                print(f"   ✅ Sample CSV generated: {csv_path}")

            # Generate comparison report
            report_path = generator.generate_comparison_report(
                analysis_results,
                release_frames,
                str(output_dir / "sample_comparison_report.json")
            )

            if report_path:
                print(
                    f"   ✅ Sample comparison report generated: {report_path}")

        # Test 3: Generate CSV from JSON
        if args.generate_csv:
            print(f"\n📄 Generating CSV from JSON file: {args.generate_csv}")

            json_file = Path(args.generate_csv)
            if not json_file.exists():
                print(f"❌ Error: JSON file not found: {args.generate_csv}")
                sys.exit(1)

            try:
                with open(json_file, 'r') as f:
                    json_data = json.load(f)

                # Convert JSON to CSV format
                if isinstance(json_data, list):
                    df = pd.DataFrame(json_data)
                elif isinstance(json_data, dict):
                    # Handle different JSON structures
                    if 'videos' in json_data:  # Comparison report format
                        rows = []
                        for video_name, video_data in json_data['videos'].items():
                            if 'release_point_data' in video_data and video_data['release_point_data']:
                                rows.append(video_data['release_point_data'])
                        df = pd.DataFrame(rows)
                    else:
                        df = pd.DataFrame([json_data])

                # Save CSV
                csv_output = output_dir / f"{json_file.stem}_converted.csv"
                df.to_csv(csv_output, index=False)

                print(f"   ✅ CSV generated from JSON: {csv_output}")
                print(f"   📊 Shape: {df.shape}")
                print(f"   📋 Columns: {list(df.columns)}")

            except Exception as e:
                print(f"   ❌ Error processing JSON file: {e}")

        # Test 4: Compare Videos
        if args.compare_videos:
            print(
                f"\n🔄 Generating comparison for videos: {args.compare_videos}")

            # Create mock analysis results for comparison
            analysis_results = {}
            release_frames = {}

            for i, video_name in enumerate(args.compare_videos):
                # Create mock data
                analysis_results[video_name] = AnalysisResult(
                    video_name=video_name,
                    result_dir=f"mock_dir_{video_name}",
                    angles_file=f"mock_angles_{video_name}.mot",
                    success=True
                )

                # Varying release frames
                release_frames[video_name] = 140 + i * 5

            # Generate comparison report
            report_path = generator.generate_comparison_report(
                analysis_results,
                release_frames,
                str(output_dir / "video_comparison.json")
            )

            if report_path:
                print(f"   ✅ Video comparison report generated: {report_path}")

                # Show comparison summary
                with open(report_path, 'r') as f:
                    report_data = json.load(f)

                print(f"   📋 Comparison Summary:")
                print(f"      Videos compared: {len(args.compare_videos)}")
                print(f"      Release frames: {list(release_frames.values())}")

        # Default behavior if no arguments
        if not any([args.test_data, args.create_test_data, args.generate_csv, args.compare_videos]):
            print("\n🎯 Running default tests...")

            # Look for existing analysis output
            analysis_dirs = [
                Path("analysis_output"),
                Path("."),
                output_dir
            ]

            found_data = False

            for analysis_dir in analysis_dirs:
                if analysis_dir.exists():
                    sports2d_dirs = list(analysis_dir.glob("*_Sports2D"))
                    if sports2d_dirs:
                        print(
                            f"   Found Sports2D directories in {analysis_dir}: {len(sports2d_dirs)}")

                        # Test with first directory
                        test_dir = sports2d_dirs[0]
                        video_name = test_dir.name.replace('_Sports2D', '')

                        # Look for angles file
                        angles_files = list(test_dir.glob("*.mot"))
                        if angles_files:
                            angles_file = angles_files[0]

                            # Create mock analysis result
                            analysis_results = {
                                video_name: AnalysisResult(
                                    video_name=video_name,
                                    result_dir=str(test_dir),
                                    angles_file=str(angles_file),
                                    success=True
                                )
                            }
                            release_frames = {video_name: 145}

                            # Generate CSV
                            csv_path = generator.generate_release_point_analysis(
                                analysis_results,
                                release_frames,
                                str(output_dir / "default_analysis.csv")
                            )

                            if csv_path:
                                print(f"   ✅ Default CSV analysis: {csv_path}")
                                found_data = True
                                break

            if not found_data:
                print("   No existing data found, creating minimal test...")

                # Create minimal test data
                test_data = [
                    {
                        'video_name': 'test_video',
                        'release_frame': 145,
                        'timestamp': 4.83,
                        'Right_elbow': 142.5,
                        'Left_elbow': 138.2,
                        'Right_knee': 165.8,
                        'Left_knee': 162.1
                    }
                ]

                df = pd.DataFrame(test_data)
                csv_path = output_dir / "minimal_test.csv"
                df.to_csv(csv_path, index=False)

                print(f"   ✅ Minimal test CSV created: {csv_path}")

        print(f"\n🎉 All tests completed successfully!")
        print(f"📁 Check output directory: {output_dir}")
        print("\n📖 Usage examples:")
        print("   python utils/result_generator.py --test-data analysis_output/")
        print("   python utils/result_generator.py --create-test-data")
        print("   python utils/result_generator.py --generate-csv results.json")

    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
