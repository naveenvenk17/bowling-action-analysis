"""
Configuration management for Cricket Analysis System.
Handles loading and managing analysis configurations.
"""

import toml
from pathlib import Path
from typing import Dict, Any, Optional
from .data_models import AnalysisConfig


class ConfigManager:
    """Manages configuration for cricket analysis."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to configuration file (TOML format)
        """
        self.config_path = config_path
        self._config = None
        self._analysis_config = None

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if self.config_path and Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    self._config = toml.load(f)
                return self._config
            except Exception as e:
                print(
                    f"Warning: Failed to load config from {self.config_path}: {e}")
                print("Using default configuration...")

        # Return default configuration
        self._config = self._get_default_config()
        return self._config

    def get_analysis_config(self) -> AnalysisConfig:
        """Get AnalysisConfig object with current settings."""
        if self._analysis_config is None:
            config_data = self.load_config()
            self._analysis_config = self._create_analysis_config(config_data)
        return self._analysis_config

    def _create_analysis_config(self, config_data: Dict[str, Any]) -> AnalysisConfig:
        """Create AnalysisConfig from configuration data."""
        # Extract relevant settings with defaults
        sports2d_config = config_data.get('sports2d', {})
        yolo_config = config_data.get('yolo', {})
        processing_config = config_data.get('processing', {})

        return AnalysisConfig(
            # Sports2D settings
            nb_persons_to_detect=sports2d_config.get(
                'nb_persons_to_detect', 1),
            person_ordering_method=sports2d_config.get(
                'person_ordering_method', 'highest_likelihood'),
            first_person_height=sports2d_config.get(
                'first_person_height', 1.75),
            pose_model=sports2d_config.get('pose_model', 'HALPE_26'),
            mode=sports2d_config.get('mode', 'balanced'),
            det_frequency=sports2d_config.get('det_frequency', 4),

            # YOLO settings
            yolo_confidence_threshold=yolo_config.get(
                'confidence_threshold', 0.25),
            yolo_model_path=yolo_config.get('model_path'),

            # Processing settings
            save_vid=processing_config.get('save_vid', True),
            save_img=processing_config.get('save_img', True),
            save_pose=processing_config.get('save_pose', True),
            calculate_angles=processing_config.get('calculate_angles', True),
            save_angles=processing_config.get('save_angles', True),
            show_realtime_results=processing_config.get(
                'show_realtime_results', False),
            make_c3d=processing_config.get('make_c3d', True),
        )

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'sports2d': {
                'nb_persons_to_detect': 1,
                'person_ordering_method': 'highest_likelihood',
                'first_person_height': 1.75,
                'pose_model': 'HALPE_26',
                'mode': 'balanced',
                'det_frequency': 4
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
                'make_c3d': True
            },
            'output': {
                'default_output_dir': 'analysis_output',
                'csv_filename': 'release_point_analysis.csv'
            }
        }

    def save_config(self, config_path: Optional[str] = None) -> bool:
        """Save current configuration to file."""
        save_path = config_path or self.config_path
        if not save_path:
            return False

        try:
            with open(save_path, 'w') as f:
                toml.dump(self._config or self._get_default_config(), f)
            return True
        except Exception as e:
            print(f"Error saving config to {save_path}: {e}")
            return False

    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        if self._config is None:
            self.load_config()

        def deep_update(config: Dict[str, Any], updates: Dict[str, Any]) -> None:
            """Recursively update nested dictionary."""
            for key, value in updates.items():
                if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                    deep_update(config[key], value)
                else:
                    config[key] = value

        deep_update(self._config, updates)
        self._analysis_config = None  # Reset cached analysis config
