"""
Streamlit UI for Cricket Analysis System.
Clean interface that uses the modular core analyzer.
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import tempfile
import os

from core.analyzer import CricketAnalyzer


class CricketAnalysisUI:
    """Streamlit-based UI for cricket video analysis."""

    def __init__(self):
        """Initialize the UI with session state management."""
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initialize all session state variables."""
        if 'analyzer' not in st.session_state:
            st.session_state.analyzer = CricketAnalyzer()
        if 'page' not in st.session_state:
            st.session_state.page = 'selection'
        if 'selected_videos' not in st.session_state:
            st.session_state.selected_videos = []
        if 'analysis_complete' not in st.session_state:
            st.session_state.analysis_complete = False
        if 'current_video_index' not in st.session_state:
            st.session_state.current_video_index = 0
        if 'current_frame_index' not in st.session_state:
            st.session_state.current_frame_index = 0
        if 'video_info_cache' not in st.session_state:
            st.session_state.video_info_cache = {}

    def run(self):
        """Main UI entry point."""
        st.set_page_config(
            page_title="Cricket Video Analysis",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Header
        st.title("Cricket Bowling Analysis System")
        st.markdown(
            "**AI-Powered Release Point Detection & Biomechanical Analysis**")

        # Navigation
        if st.session_state.page == 'selection':
            self._video_selection_page()
        elif st.session_state.page == 'analysis':
            self._frame_analysis_page()

    def _video_selection_page(self):
        """Video selection and analysis initiation page."""
        st.header("Step 1: Select Videos for Analysis")

        with st.container():
            st.info("Upload 2-10 cricket bowling videos for analysis. The system will analyze each video using AI pose estimation and suggest optimal release points.")

            uploaded_files = st.file_uploader(
                "Choose video files",
                type=['mp4', 'avi', 'mov', 'mkv'],
                accept_multiple_files=True,
                key="video_uploader",
                help="Select between 2 and 10 videos for comparative analysis"
            )

            if uploaded_files:
                self._display_upload_summary(uploaded_files)

                if self._validate_uploads(uploaded_files):
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button("Start Analysis", type="primary", use_container_width=True):
                            self._process_uploaded_videos(uploaded_files)

    def _display_upload_summary(self, uploaded_files):
        """Display summary of uploaded files."""
        st.subheader("Selected Videos")

        for i, file in enumerate(uploaded_files):
            file_size_mb = len(file.getvalue()) / (1024 * 1024)
            st.write(f"**{i+1}.** {file.name} ({file_size_mb:.1f} MB)")

    def _validate_uploads(self, uploaded_files):
        """Validate uploaded files."""
        if len(uploaded_files) < 2:
            st.warning(
                "Please select at least 2 videos for comparison analysis")
            return False
        elif len(uploaded_files) > 10:
            st.error("Maximum 10 videos allowed to ensure optimal performance")
            return False
        else:
            st.success(f"{len(uploaded_files)} videos ready for analysis")
            return True

    def _process_uploaded_videos(self, uploaded_files):
        """Process uploaded videos through the analysis pipeline."""
        temp_dir = Path(tempfile.mkdtemp())
        video_paths = []

        # Create progress tracking
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Save uploaded files
            status_text.text("Saving uploaded files...")
            for i, uploaded_file in enumerate(uploaded_files):
                temp_video_path = temp_dir / uploaded_file.name
                with open(temp_video_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                video_paths.append(str(temp_video_path))
                progress_bar.progress((i + 1) / len(uploaded_files) * 0.2)

            # Run analysis
            status_text.text(
                "Running AI analysis (this may take several minutes)...")
            st.session_state.analyzer.analyze_multiple_videos(video_paths)

            progress_bar.progress(1.0)
            status_text.text("Analysis complete!")

        # Store results and transition
        st.session_state.selected_videos = video_paths
        st.session_state.analysis_complete = True
        st.session_state.page = 'analysis'

        st.success(
            "Analysis completed successfully! Redirecting to frame analysis...")
        st.rerun()

    def _frame_analysis_page(self):
        """Frame analysis and release point selection page."""
        if not st.session_state.analysis_complete:
            st.error(
                "No analysis data available. Please return to video selection.")
            if st.button("Back to Video Selection"):
                self._reset_session()
            return

        st.header("Step 2: Review & Adjust Release Points")

        # Sidebar controls
        self._render_sidebar()

        # Main content area
        col1, col2 = st.columns([3, 2])

        with col1:
            self._render_video_viewer()

        with col2:
            self._render_control_panel()

    def _render_sidebar(self):
        """Render the sidebar with video selection and navigation."""
        with st.sidebar:
            st.header("Navigation")

            # Video selector
            video_names = [
                Path(path).stem for path in st.session_state.selected_videos]

            selected_video = st.selectbox(
                "Select Video",
                options=range(len(video_names)),
                format_func=lambda x: f"{x+1}. {video_names[x]}",
                key="video_selector"
            )

            if selected_video != st.session_state.current_video_index:
                self._switch_video(selected_video)

            # Video info
            video_name = video_names[selected_video]
            self._display_video_info(video_name)

            # AI suggestion info
            self._display_ai_suggestion(video_name)

            # Analysis summary
            st.markdown("---")
            self._display_analysis_summary()

    def _render_video_viewer(self):
        """Render the main video viewer area."""
        video_names = [
            Path(path).stem for path in st.session_state.selected_videos]
        current_video_name = video_names[st.session_state.current_video_index]

        st.subheader(f"Video: {current_video_name}")

        # Get current frame
        frame = self._get_current_frame()

        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(
                frame_rgb,
                caption=f"Frame {st.session_state.current_frame_index}",
                use_container_width=True
            )
        else:
            st.error("Could not load frame. Please check video file integrity.")

    def _render_control_panel(self):
        """Render the control panel for frame navigation and release point setting."""
        video_names = [
            Path(path).stem for path in st.session_state.selected_videos]
        current_video_name = video_names[st.session_state.current_video_index]

        st.subheader("Frame Controls")

        # Frame slider
        video_info = self._get_video_info(current_video_name)

        if video_info:
            frame_index = st.slider(
                "Frame Position",
                min_value=0,
                max_value=video_info.frame_count - 1,
                value=st.session_state.current_frame_index,
                key=f"frame_slider_{st.session_state.current_video_index}"
            )

            st.session_state.current_frame_index = frame_index

            # Navigation buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("-10", disabled=frame_index < 10):
                    st.session_state.current_frame_index = max(
                        0, frame_index - 10)
                    st.rerun()
            with col2:
                if st.button("-1", disabled=frame_index <= 0):
                    st.session_state.current_frame_index = frame_index - 1
                    st.rerun()
            with col3:
                if st.button("+1", disabled=frame_index >= video_info.frame_count - 1):
                    st.session_state.current_frame_index = frame_index + 1
                    st.rerun()

            col4, col5 = st.columns(2)
            with col4:
                if st.button("+10", disabled=frame_index >= video_info.frame_count - 10):
                    st.session_state.current_frame_index = min(
                        video_info.frame_count - 1, frame_index + 10)
                    st.rerun()

        st.markdown("---")

        # Release point setting
        st.subheader("Release Point")

        if st.button("Set Release Point", type="primary", use_container_width=True):
            st.session_state.analyzer.set_release_frame(
                current_video_name, frame_index)
            st.success(f"Release point set at frame {frame_index}")
            st.rerun()

        # CSV Generation
        st.markdown("---")
        st.subheader("Export Results")

        release_count = len(st.session_state.analyzer.release_frames)

        if release_count > 0:
            st.info(f"{release_count} release points set")

            if st.button("Generate CSV Analysis", type="secondary", use_container_width=True):
                self._generate_and_download_csv()
        else:
            st.warning("No release points set yet")

    def _switch_video(self, video_index):
        """Switch to a different video."""
        st.session_state.current_video_index = video_index

        # Set frame to AI suggestion if available
        video_names = [
            Path(path).stem for path in st.session_state.selected_videos]
        video_name = video_names[video_index]

        if video_name in st.session_state.analyzer.suggested_frames:
            suggestion = st.session_state.analyzer.suggested_frames[video_name]
            if suggestion.release_frame is not None:
                st.session_state.current_frame_index = suggestion.release_frame
            else:
                st.session_state.current_frame_index = 0
        else:
            st.session_state.current_frame_index = 0

    def _get_current_frame(self):
        """Get the current frame for display."""
        current_video_path = st.session_state.selected_videos[st.session_state.current_video_index]
        video_names = [
            Path(path).stem for path in st.session_state.selected_videos]
        current_video_name = video_names[st.session_state.current_video_index]

        # Try analyzed video first
        if current_video_name in st.session_state.analyzer.analysis_results:
            analyzed_path = st.session_state.analyzer.analysis_results[
                current_video_name].analyzed_path
            frame = st.session_state.analyzer.get_frame_at_index(
                analyzed_path, st.session_state.current_frame_index)
            if frame is not None:
                return frame

        # Fallback to original video
        return st.session_state.analyzer.get_frame_at_index(current_video_path, st.session_state.current_frame_index)

    def _get_video_info(self, video_name):
        """Get cached video info."""
        if video_name not in st.session_state.video_info_cache:
            video_path = None
            for path in st.session_state.selected_videos:
                if Path(path).stem == video_name:
                    video_path = path
                    break

            if video_path:
                st.session_state.video_info_cache[video_name] = st.session_state.analyzer.get_video_info(
                    video_path)

        return st.session_state.video_info_cache.get(video_name)

    def _display_video_info(self, video_name):
        """Display video information in sidebar."""
        video_info = self._get_video_info(video_name)

        if video_info:
            st.markdown("**Video Info**")
            st.write(f"Frames: {video_info.frame_count}")
            st.write(f"FPS: {video_info.fps:.1f}")
            st.write(f"Duration: {video_info.duration:.1f}s")

    def _display_ai_suggestion(self, video_name):
        """Display AI suggestion information."""
        if video_name in st.session_state.analyzer.suggested_frames:
            suggestion = st.session_state.analyzer.suggested_frames[video_name]
            if suggestion.release_frame is not None:
                st.success(f"AI Suggestion: Frame {suggestion.release_frame}")
                if suggestion.confidence:
                    st.write(f"Confidence: {suggestion.confidence:.2f}")
            else:
                st.warning("No AI suggestion available")
        else:
            st.info("AI analysis pending...")

    def _display_analysis_summary(self):
        """Display analysis summary in sidebar."""
        summary = st.session_state.analyzer.get_analysis_summary()

        st.markdown("**Analysis Summary**")
        st.write(f"Videos analyzed: {summary['total_videos_analyzed']}")
        st.write(
            f"Release points set: {summary['videos_with_release_points']}")
        st.write(f"AI suggestions: {summary['ai_suggestions_available']}")

        if summary['release_points_set']:
            with st.expander("Release Points Set"):
                for name, frame in summary['release_points_set'].items():
                    st.write(f"**{name}:** Frame {frame}")

    def _generate_and_download_csv(self):
        """Generate CSV analysis and provide download."""
        with st.spinner("Generating CSV analysis..."):
            csv_path = st.session_state.analyzer.generate_csv_analysis()

            if csv_path:
                with open(csv_path, 'rb') as f:
                    csv_data = f.read()

                st.download_button(
                    label="Download CSV Analysis",
                    data=csv_data,
                    file_name="cricket_release_point_analysis.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                st.success("CSV analysis generated successfully!")
            else:
                st.error("Failed to generate CSV analysis")

    def _reset_session(self):
        """Reset session state and return to video selection."""
        st.session_state.page = 'selection'
        st.session_state.analysis_complete = False
        st.session_state.selected_videos = []
        st.session_state.current_video_index = 0
        st.session_state.current_frame_index = 0
        st.session_state.video_info_cache = {}
        st.session_state.analyzer = CricketAnalyzer()
        st.rerun()


def main():
    """Main entry point for Streamlit UI."""
    ui = CricketAnalysisUI()
    ui.run()


if __name__ == "__main__":
    main()
