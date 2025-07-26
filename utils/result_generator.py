"""
Result generation module for Cricket Analysis System.
Handles CSV export and data analysis of release point measurements.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
from datetime import datetime

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
