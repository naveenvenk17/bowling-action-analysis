"""
Cricket Video Analysis System - Main Entry Point

A lightweight entry point that orchestrates the modular cricket analysis system.
Supports both UI and command-line interfaces.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from core.analyzer import CricketAnalyzer


def run_ui():
    """Launch the Streamlit UI"""
    import subprocess
    subprocess.run([sys.executable, "-m", "streamlit",
                   "run", "ui/streamlit_app.py"])


def run_backend_analysis(
    video_paths: List[str],
    output_dir: str = "analysis_output",
    config_path: Optional[str] = None
) -> CricketAnalyzer:
    """
    Run backend analysis standalone.

    Args:
        video_paths: List of video file paths to analyze
        output_dir: Output directory for results
        config_path: Optional configuration file path

    Returns:
        CricketAnalyzer instance with completed analysis
    """
    print(f"Initializing Cricket Analysis System...")
    analyzer = CricketAnalyzer(output_dir, config_path)

    print(f"Starting analysis of {len(video_paths)} videos...")
    results = analyzer.analyze_multiple_videos(video_paths)

    print(f"\nAnalysis Results:")
    for video_name, result in results.items():
        if result.analyzed_path:
            print(f"  SUCCESS: {video_name}: {result.analyzed_path}")
            if result.suggested_release_frame is not None:
                print(
                    f"           AI suggested release: Frame {result.suggested_release_frame}")
        else:
            print(f"  FAILED: {video_name}: Analysis failed")

    print(f"\nAll results saved to: {output_dir}")
    return analyzer


def main():
    """Main CLI entry point with enhanced argument handling."""
    parser = argparse.ArgumentParser(
        description="Cricket Video Analysis System - AI-Powered Biomechanical Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Launch UI interface
    python main.py --mode ui
    
    # Analyze videos via command line
    python main.py --mode backend --videos video1.mp4 video2.mp4
    
    # Analyze with custom output directory and config
    python main.py --mode backend --videos *.mp4 --output results --config config.toml
    
    # Set specific release points and generate CSV
    python main.py --mode backend --videos video1.mp4 video2.mp4 --release-frames 120 95
        """)

    parser.add_argument(
        "--mode",
        choices=["ui", "backend"],
        default="ui",
        help="Run UI interface or backend analysis (default: ui)"
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        help="Video file paths for backend analysis"
    )
    parser.add_argument(
        "--output",
        default="analysis_output",
        help="Output directory for analysis results (default: analysis_output)"
    )
    parser.add_argument(
        "--config",
        help="Path to configuration file (TOML format)"
    )
    parser.add_argument(
        "--release-frames",
        nargs="+",
        type=int,
        help="Manual release frame indices (same order as videos)"
    )
    parser.add_argument(
        "--csv-output",
        default="release_point_analysis.csv",
        help="Output CSV file for release point analysis (default: release_point_analysis.csv)"
    )
    parser.add_argument(
        "--export-all",
        action="store_true",
        help="Export all data formats (CSV, JSON, individual angles)"
    )

    args = parser.parse_args()

    if args.mode == "ui":
        print("Launching Cricket Analysis UI...")
        run_ui()

    elif args.mode == "backend":
        if not args.videos:
            print("ERROR: --videos required for backend mode")
            print("Use --help for usage examples")
            sys.exit(1)

        # Validate video files
        video_paths = []
        for video_pattern in args.videos:
            resolved_path = Path(video_pattern).resolve()
            if not resolved_path.exists():
                print(f"ERROR: Video file not found: {video_pattern}")
                sys.exit(1)
            video_paths.append(str(resolved_path))

        # Run analysis
        analyzer = run_backend_analysis(video_paths, args.output, args.config)

        # Set manual release frames if provided
        if args.release_frames:
            if len(args.release_frames) != len(video_paths):
                print("ERROR: Number of release frames must match number of videos")
                sys.exit(1)

            print("\nSetting manual release points...")
            for video_path, frame_index in zip(video_paths, args.release_frames):
                video_name = Path(video_path).stem
                analyzer.set_release_frame(video_name, frame_index)
                print(f"  {video_name}: Frame {frame_index}")

        # Generate outputs
        if analyzer.release_frames:
            if args.export_all:
                print("\nExporting all data formats...")
                exports = analyzer.export_all_data(args.output + "_export")
                total_files = len(
                    exports['csv_files']) + len(exports['json_files']) + len(exports['angles_files'])
                print(f"Export complete: {total_files} files generated")
            else:
                print("\nGenerating CSV analysis...")
                csv_path = analyzer.generate_csv_analysis(args.csv_output)
                if csv_path:
                    print(f"Release point analysis saved to: {csv_path}")
                else:
                    print("Failed to generate release point analysis")
        else:
            print(
                "\nWARNING: No release points set. Use --release-frames to set manual points.")
            print("AI suggestions may be available in analysis results.")

        # Display summary
        summary = analyzer.get_analysis_summary()
        print(f"\nAnalysis Summary:")
        print(f"  Videos analyzed: {summary['total_videos_analyzed']}")
        print(f"  AI suggestions: {summary['ai_suggestions_available']}")
        print(f"  Release points set: {summary['videos_with_release_points']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
