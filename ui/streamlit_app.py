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
import math
from PIL import Image
import io

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
        if 'collage_data' not in st.session_state:
            st.session_state.collage_data = None

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
        elif st.session_state.page == 'collage':
            self._collage_page()

    def _video_selection_page(self):
        """Video selection and analysis initiation page."""
        st.header("Step 1: Select Videos for Analysis")

        # Option to load existing results
        with st.expander("🔄 Load Existing Analysis Results (for testing)"):
            st.info("Load pre-existing analysis results from bowl1-bowl9 directories")
            if st.button("Load Existing Bowl Analysis Results"):
                self._load_existing_results()

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

    def _load_existing_results(self):
        """Load existing analysis results from bowl directories."""
        with st.spinner("Loading existing analysis results..."):
            results = st.session_state.analyzer.load_existing_analysis_results()

            if results:
                # Create dummy video paths for existing results
                st.session_state.selected_videos = [
                    f"{name}.mp4" for name in results.keys()]
                st.session_state.analysis_complete = True
                st.session_state.page = 'analysis'

                st.success(f"Loaded {len(results)} existing analysis results!")
                st.info("Transitioning to frame analysis page...")
                st.rerun()
            else:
                st.error(
                    "No existing analysis results found. Make sure bowl1-bowl9 directories exist.")

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

        # Header with debug toggle
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(f"Video: {current_video_name}")
        with col2:
            st.session_state.debug_mode = st.checkbox(
                "Debug Mode", value=st.session_state.get('debug_mode', False))

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
            st.error(
                "⚠️ Could not load frame. Enable Debug Mode for detailed diagnostics.")

            # Offer troubleshooting options
            if st.button("🔧 Run Video Diagnostics"):
                self._run_video_diagnostics(current_video_name)

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

        # Generate Analysis - Updated to go to collage page
        st.markdown("---")
        st.subheader("Generate Analysis")

        release_count = len(st.session_state.analyzer.release_frames)

        if release_count > 0:
            st.info(f"{release_count} release points set")

            if st.button("Generate Analysis", type="secondary", use_container_width=True):
                self._generate_collage_data()
                st.session_state.page = 'collage'
                st.rerun()
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
        """Get the current frame for display with improved error handling."""
        current_video_path = st.session_state.selected_videos[st.session_state.current_video_index]
        video_names = [
            Path(path).stem for path in st.session_state.selected_videos]
        current_video_name = video_names[st.session_state.current_video_index]

        # Debug information
        if st.session_state.get('debug_mode', False):
            st.write(
                f"🔍 Debug: Getting frame {st.session_state.current_frame_index} from {current_video_name}")

        # Validate frame index first
        video_info = self._get_video_info(current_video_name)
        if video_info is None:
            if st.session_state.get('debug_mode', False):
                st.error(f"Could not get video info for {current_video_name}")
            return None

        if st.session_state.current_frame_index >= video_info.frame_count:
            if st.session_state.get('debug_mode', False):
                st.error(
                    f"Frame index {st.session_state.current_frame_index} exceeds video frame count {video_info.frame_count}")
            # Reset to last valid frame
            st.session_state.current_frame_index = max(
                0, video_info.frame_count - 1)

        # Try analyzed video first
        if current_video_name in st.session_state.analyzer.analysis_results:
            analysis_result = st.session_state.analyzer.analysis_results[current_video_name]
            if analysis_result is not None and hasattr(analysis_result, 'analyzed_path'):
                analyzed_path = analysis_result.analyzed_path
                if analyzed_path and Path(analyzed_path).exists():
                    frame = st.session_state.analyzer.get_frame_at_index(
                        analyzed_path, st.session_state.current_frame_index)
                    if frame is not None:
                        return frame
                    elif st.session_state.get('debug_mode', False):
                        st.warning(
                            f"Could not read frame from analyzed video: {analyzed_path}")
            elif st.session_state.get('debug_mode', False):
                st.warning(
                    f"Analysis result is None or invalid for {current_video_name}")

        # Fallback to original video
        frame = st.session_state.analyzer.get_frame_at_index(
            current_video_path, st.session_state.current_frame_index)

        if frame is None and st.session_state.get('debug_mode', False):
            # Show detailed diagnostic information
            validation = st.session_state.analyzer.validate_frame_access(
                current_video_path, st.session_state.current_frame_index)
            st.error("Frame loading failed - Diagnostic info:")
            st.json(validation)

        return frame

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

    def _run_video_diagnostics(self, video_name):
        """Run comprehensive video diagnostics."""
        # Find video path
        video_path = None
        for path in st.session_state.selected_videos:
            if Path(path).stem == video_name:
                video_path = path
                break

        if not video_path:
            st.error(f"Could not find video path for {video_name}")
            return

        st.write("🔍 Running video diagnostics...")

        # Basic file checks
        file_exists = Path(video_path).exists()
        file_size = Path(video_path).stat().st_size if file_exists else 0

        st.write(f"📁 **File Status:**")
        st.write(f"   - Exists: {'✅' if file_exists else '❌'}")
        st.write(
            f"   - Size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        st.write(f"   - Path: `{video_path}`")

        if not file_exists:
            st.error("Video file does not exist!")
            return

        # Video info
        video_info = st.session_state.analyzer.get_video_info(video_path)
        if video_info:
            st.write(f"🎬 **Video Properties:**")
            st.write(f"   - Frame count: {video_info.frame_count}")
            st.write(f"   - FPS: {video_info.fps:.2f}")
            st.write(f"   - Duration: {video_info.duration:.2f} seconds")
        else:
            st.error("Could not read video properties!")
            return

        # Frame access test
        current_frame = st.session_state.current_frame_index
        validation = st.session_state.analyzer.validate_frame_access(
            video_path, current_frame)

        st.write(f"🎯 **Frame Access Test (Frame {current_frame}):**")
        st.write(
            f"   - Video exists: {'✅' if validation['video_exists'] else '❌'}")
        st.write(
            f"   - Video openable: {'✅' if validation['video_openable'] else '❌'}")
        st.write(
            f"   - Frame index valid: {'✅' if validation['frame_index_valid'] else '❌'}")
        st.write(
            f"   - Frame readable: {'✅' if validation['frame_readable'] else '❌'}")

        if not validation['valid']:
            st.error(f"❌ {validation['error_message']}")
        else:
            st.success("✅ Frame access is working correctly!")

        # Test multiple frames
        st.write("🎲 **Random Frame Test:**")
        test_frames = [0, video_info.frame_count//4, video_info.frame_count//2,
                       3*video_info.frame_count//4, video_info.frame_count-1]

        success_count = 0
        for frame_idx in test_frames:
            if frame_idx < video_info.frame_count:
                test_validation = st.session_state.analyzer.validate_frame_access(
                    video_path, frame_idx)
                status = "✅" if test_validation['valid'] else "❌"
                st.write(f"   - Frame {frame_idx}: {status}")
                if test_validation['valid']:
                    success_count += 1

        st.write(
            f"**Result:** {success_count}/{len(test_frames)} frames accessible")

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

    def _collage_page(self):
        """Display the collage of release point frames."""
        if not st.session_state.collage_data:
            st.error(
                "No collage data available. Please return to analysis and generate the collage.")
            if st.button("Back to Analysis"):
                st.session_state.page = 'analysis'
                st.rerun()
            return

        st.header("Release Point Analysis Collage")

        # Navigation buttons
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("← Back to Analysis"):
                st.session_state.page = 'analysis'
                st.rerun()

        with col3:
            # Download buttons
            col3a, col3b = st.columns(2)
            with col3a:
                if st.button("📥 Download Image"):
                    self._download_collage_image()
            with col3b:
                if st.button("📄 Download CSV"):
                    self._generate_and_download_csv()

        st.markdown("---")

        # Display collage
        collage_data = st.session_state.collage_data
        if collage_data['frames']:
            self._display_collage(
                collage_data['frames'], collage_data['metadata'])
        else:
            st.warning("No frames available for collage")

    def _generate_collage_data(self):
        """Generate data for the collage from release points."""
        collage_frames = []
        metadata = []

        with st.spinner("Generating collage data..."):
            for video_name, frame_index in st.session_state.analyzer.release_frames.items():
                frame_data = self._extract_release_frame(
                    video_name, frame_index)
                if frame_data:
                    collage_frames.append(frame_data['frame'])
                    metadata.append({
                        'video_name': video_name,
                        'frame_index': frame_index,
                        'timestamp': frame_data.get('timestamp')
                    })

        st.session_state.collage_data = {
            'frames': collage_frames,
            'metadata': metadata
        }

    def _extract_release_frame(self, video_name, frame_index):
        """Extract and process a specific release frame."""
        try:
            # First try to load from existing analyzed frame images (bowl directories)
            if video_name in st.session_state.analyzer.analysis_results:
                analysis_result = st.session_state.analyzer.analysis_results[video_name]
                if analysis_result and analysis_result.result_dir:
                    # Look for existing frame images in the Sports2D img directory
                    img_dir = Path(analysis_result.result_dir) / \
                        f"{Path(analysis_result.result_dir).name}_img"

                    if img_dir.exists():
                        # Find the frame image file - Sports2D uses 6-digit zero-padded frame numbers
                        frame_filename = f"{Path(analysis_result.result_dir).name}_{frame_index:06d}.png"
                        frame_path = img_dir / frame_filename

                        if frame_path.exists():
                            # Load the existing frame image (already has pose overlay)
                            frame = cv2.imread(str(frame_path))
                            if frame is not None:
                                processed_frame = self._process_frame_for_collage(
                                    frame)
                                return {
                                    'frame': processed_frame,
                                    'timestamp': self._calculate_timestamp(frame_index, video_name),
                                    'source': 'analyzed_image'
                                }
                        else:
                            print(f"Frame image not found: {frame_path}")

            # Fallback: try to get frame from analyzed video if available
            if video_name in st.session_state.analyzer.analysis_results:
                analysis_result = st.session_state.analyzer.analysis_results[video_name]
                if analysis_result and analysis_result.analyzed_path:
                    frame = st.session_state.analyzer.get_frame_at_index(
                        analysis_result.analyzed_path, frame_index)
                    if frame is not None:
                        processed_frame = self._process_frame_for_collage(
                            frame)
                        return {
                            'frame': processed_frame,
                            'timestamp': self._calculate_timestamp(frame_index, video_name),
                            'source': 'analyzed_video'
                        }

            # Final fallback: try original video
            video_path = None
            for path in st.session_state.selected_videos:
                if Path(path).stem == video_name:
                    video_path = path
                    break

            if video_path:
                frame = st.session_state.analyzer.get_frame_at_index(
                    video_path, frame_index)
                if frame is not None:
                    processed_frame = self._process_frame_for_collage(frame)
                    return {
                        'frame': processed_frame,
                        'timestamp': self._calculate_timestamp(frame_index, video_name),
                        'source': 'original_video'
                    }

        except Exception as e:
            st.error(f"Error extracting frame for {video_name}: {e}")

        return None

    def _process_frame_for_collage(self, frame):
        """Process frame to highlight person in blue square."""
        try:
            # Try to use the release point detector for person detection
            from utils.release_point_detector import ReleasePointDetector

            if hasattr(st.session_state.analyzer, 'release_detector') and st.session_state.analyzer.release_detector.model:
                # Use YOLO to detect person
                detector = st.session_state.analyzer.release_detector

                # Save frame temporarily for YOLO detection
                temp_path = "temp_frame_for_detection.jpg"
                cv2.imwrite(temp_path, frame)

                # Detect objects
                detection_results = detector._detect_objects_standard(
                    temp_path)

                # Clean up temp file
                if Path(temp_path).exists():
                    Path(temp_path).unlink()

                # Process person detections
                if detection_results['persons']:
                    # Use the highest confidence person
                    person = detection_results['persons'][0]
                    bbox = person['bbox']

                    # Draw blue rectangle around person
                    x1, y1, x2, y2 = [int(coord) for coord in bbox]

                    # Create a copy of the frame
                    processed_frame = frame.copy()

                    # Draw blue rectangle
                    cv2.rectangle(processed_frame, (x1, y1),
                                  (x2, y2), (255, 0, 0), 3)  # Blue in BGR

                    # Add confidence text
                    confidence_text = f"Person: {person['confidence']:.2f}"
                    cv2.putText(processed_frame, confidence_text, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                    return processed_frame

        except Exception as e:
            print(f"Error in person detection: {e}")

        # Return original frame if detection fails
        return frame

    def _calculate_timestamp(self, frame_index, video_name):
        """Calculate timestamp for frame."""
        video_info = self._get_video_info(video_name)
        if video_info and video_info.fps > 0:
            return frame_index / video_info.fps
        return None

    def _display_collage(self, frames, metadata):
        """Display frames in a dynamic grid layout."""
        num_frames = len(frames)
        if num_frames == 0:
            return

        # Calculate optimal grid dimensions
        if num_frames <= 2:
            cols = num_frames
            rows = 1
        elif num_frames <= 6:
            cols = 3
            rows = math.ceil(num_frames / 3)
        else:
            cols = 4
            rows = math.ceil(num_frames / 4)

        st.subheader(f"Release Point Frames ({num_frames} frames)")

        # Display frames in grid
        for row in range(rows):
            columns = st.columns(cols)
            for col in range(cols):
                frame_idx = row * cols + col
                if frame_idx < num_frames:
                    with columns[col]:
                        frame = frames[frame_idx]
                        meta = metadata[frame_idx]

                        # Convert frame to RGB for display
                        if len(frame.shape) == 3 and frame.shape[2] == 3:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        else:
                            frame_rgb = frame

                        st.image(
                            frame_rgb,
                            caption=f"{meta['video_name']}\nFrame: {meta['frame_index']}" +
                            (f"\nTime: {meta['timestamp']:.2f}s" if meta['timestamp'] else ""),
                            use_container_width=True
                        )

    def _download_collage_image(self):
        """Generate and download the collage as a single image."""
        if not st.session_state.collage_data or not st.session_state.collage_data['frames']:
            st.error("No collage data available")
            return

        try:
            frames = st.session_state.collage_data['frames']
            metadata = st.session_state.collage_data['metadata']

            # Create collage image
            collage_image = self._create_collage_image(frames, metadata)

            # Convert to PIL Image
            if len(collage_image.shape) == 3 and collage_image.shape[2] == 3:
                collage_pil = Image.fromarray(
                    cv2.cvtColor(collage_image, cv2.COLOR_BGR2RGB))
            else:
                collage_pil = Image.fromarray(collage_image)

            # Save to bytes buffer
            img_buffer = io.BytesIO()
            collage_pil.save(img_buffer, format='PNG')
            img_data = img_buffer.getvalue()

            # Download button
            st.download_button(
                label="Download Collage Image",
                data=img_data,
                file_name="release_point_collage.png",
                mime="image/png",
                use_container_width=True
            )
            st.success("Collage image ready for download!")

        except Exception as e:
            st.error(f"Error creating collage image: {e}")

    def _create_collage_image(self, frames, metadata):
        """Create a single collage image from frames."""
        num_frames = len(frames)
        if num_frames == 0:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        # Calculate grid dimensions
        if num_frames <= 2:
            cols = num_frames
            rows = 1
        elif num_frames <= 6:
            cols = 3
            rows = math.ceil(num_frames / 3)
        else:
            cols = 4
            rows = math.ceil(num_frames / 4)

        # Get frame dimensions (assume all frames are same size)
        frame_height, frame_width = frames[0].shape[:2]

        # Create collage canvas
        collage_height = rows * frame_height
        collage_width = cols * frame_width
        collage = np.zeros((collage_height, collage_width, 3), dtype=np.uint8)

        # Place frames in collage
        for i, frame in enumerate(frames):
            row = i // cols
            col = i % cols

            y_start = row * frame_height
            y_end = y_start + frame_height
            x_start = col * frame_width
            x_end = x_start + frame_width

            # Ensure frame is 3-channel
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            collage[y_start:y_end, x_start:x_end] = frame

        return collage

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
