#!/usr/bin/env python3
"""
Cricket Analysis System - Configuration Management

This module provides comprehensive configuration management for the Cricket Analysis System.
It includes classes and methods for:

- TOML configuration file loading and parsing
- Default configuration generation and management
- Dynamic configuration updates and validation
- Sports2D configuration export and formatting
- Configuration validation and error handling
- Nested configuration dictionary management

Classes:
    ConfigManager: Manages configuration for cricket analysis

Key Features:
    - TOML file format support for human-readable configuration
    - Hierarchical configuration with sections (sports2d, yolo, processing)
    - Default configuration fallback when files are missing
    - Dynamic configuration updates with deep merge support
    - Sports2D-compatible configuration export
    - Configuration validation and error reporting

Dependencies:
    - toml: TOML file format parsing and writing
    - pathlib: Path handling
    - typing: Type annotations

Usage:
    Can be run as a standalone module for testing:
    python utils/config_manager.py --create-config config.toml

    Or imported and used programmatically:
    from utils.config_manager import ConfigManager

Author: Cricket Analysis System
Version: 1.0
"""

import toml
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
import sys
import json

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
            use_gpu=sports2d_config.get('use_gpu', True),

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
            flip_left_right=processing_config.get('flip_left_right', False),
        )

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'sports2d': {
                'nb_persons_to_detect': 1,
                'person_ordering_method': 'highest_likelihood',
                'first_person_height': 1.75,
                'pose_model': 'HALPE_26',
                'mode': 'balanced',  # 'lightweight', 'balanced', 'performance'
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


if __name__ == "__main__":
    """
    Test the configuration manager with command-line arguments.

    Usage examples:
    python utils/config_manager.py --create-config config.toml
    python utils/config_manager.py --validate-config config.toml
    python utils/config_manager.py --help
    """
    parser = argparse.ArgumentParser(
        description="Test Cricket Analysis Configuration Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python utils/config_manager.py --create-config config.toml
  python utils/config_manager.py --validate-config config.toml
  python utils/config_manager.py --show-defaults
  python utils/config_manager.py --update-config config.toml --set sports2d.mode=accurate
        """
    )

    parser.add_argument('--create-config', type=str,
                        help='Create a new configuration file at specified path')
    parser.add_argument('--validate-config', type=str,
                        help='Validate an existing configuration file')
    parser.add_argument('--show-defaults', action='store_true',
                        help='Show default configuration values')
    parser.add_argument('--update-config', type=str,
                        help='Update existing configuration file')
    parser.add_argument('--set', action='append', dest='config_updates',
                        help='Set configuration values (format: section.key=value)')
    parser.add_argument('--output-dir', type=str, default='test_output',
                        help='Output directory for test results (default: test_output)')
    parser.add_argument('--export-sports2d', type=str,
                        help='Export Sports2D configuration for given video path')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("🏏 Testing Cricket Analysis Configuration Manager")
    print("=" * 60)

    try:
        # Test 1: Create Configuration
        if args.create_config:
            print(f"\n📝 Creating configuration file: {args.create_config}")

            config_path = Path(args.create_config)

            # Check if file already exists
            if config_path.exists():
                response = input(
                    f"   ⚠️  File {config_path} already exists. Overwrite? [y/N]: ")
                if response.lower() != 'y':
                    print("   Operation cancelled.")
                    sys.exit(0)

            # Create config manager and save default config
            config_manager = ConfigManager()
            config_manager.config_path = str(config_path)

            # Load default configuration
            default_config = config_manager._get_default_config()
            config_manager._config = default_config

            # Save to file
            success = config_manager.save_config()

            if success:
                print(f"   ✅ Configuration file created successfully!")
                print(f"   📁 Location: {config_path.absolute()}")

                # Show some key settings
                print(f"   📋 Key settings:")
                print(
                    f"      Sports2D mode: {default_config['sports2d']['mode']}")
                print(
                    f"      Pose model: {default_config['sports2d']['pose_model']}")
                print(
                    f"      GPU enabled: {default_config['sports2d']['use_gpu']}")
                print(
                    f"      YOLO confidence: {default_config['yolo']['confidence_threshold']}")
            else:
                print(f"   ❌ Failed to create configuration file")
                sys.exit(1)

        # Test 2: Validate Configuration
        if args.validate_config:
            print(f"\n🔍 Validating configuration file: {args.validate_config}")

            config_path = Path(args.validate_config)
            if not config_path.exists():
                print(
                    f"   ❌ Configuration file not found: {args.validate_config}")
                sys.exit(1)

            # Load and validate configuration
            config_manager = ConfigManager(str(config_path))

            try:
                config_data = config_manager.load_config()
                analysis_config = config_manager.get_analysis_config()

                print(f"   ✅ Configuration file is valid!")
                print(f"   📊 Configuration summary:")
                print(
                    f"      Sports2D settings: {len(config_data.get('sports2d', {}))}")
                print(
                    f"      YOLO settings: {len(config_data.get('yolo', {}))}")
                print(
                    f"      Processing settings: {len(config_data.get('processing', {}))}")

                # Validate specific settings
                validation_results = []

                # Check Sports2D mode
                mode = analysis_config.mode
                if mode in ['accurate', 'balanced', 'fast']:
                    validation_results.append(f"      ✅ Sports2D mode: {mode}")
                else:
                    validation_results.append(
                        f"      ⚠️  Unknown Sports2D mode: {mode}")

                # Check pose model
                pose_model = analysis_config.pose_model
                valid_models = ['HALPE_26', 'COCO_17', 'MPII_16']
                if pose_model in valid_models:
                    validation_results.append(
                        f"      ✅ Pose model: {pose_model}")
                else:
                    validation_results.append(
                        f"      ⚠️  Unknown pose model: {pose_model}")

                # Check YOLO confidence
                yolo_conf = analysis_config.yolo_confidence_threshold
                if 0.0 <= yolo_conf <= 1.0:
                    validation_results.append(
                        f"      ✅ YOLO confidence: {yolo_conf}")
                else:
                    validation_results.append(
                        f"      ❌ Invalid YOLO confidence: {yolo_conf}")

                print(f"   📋 Validation details:")
                for result in validation_results:
                    print(result)

                # Save validation report
                import datetime
                validation_report = {
                    'config_file': str(config_path),
                    'validation_timestamp': datetime.datetime.now().isoformat(),
                    'is_valid': True,
                    'settings_summary': {
                        'sports2d_mode': mode,
                        'pose_model': pose_model,
                        'yolo_confidence': yolo_conf,
                        'gpu_enabled': analysis_config.use_gpu
                    },
                    'validation_results': validation_results
                }

                report_file = output_dir / "config_validation_report.json"
                with open(report_file, 'w') as f:
                    json.dump(validation_report, f, indent=2, default=str)
                print(f"   💾 Validation report saved: {report_file}")

            except Exception as e:
                print(f"   ❌ Configuration validation failed: {e}")
                sys.exit(1)

        # Test 3: Show Defaults
        if args.show_defaults:
            print("\n📋 Default Configuration Values:")

            config_manager = ConfigManager()
            default_config = config_manager._get_default_config()

            # Pretty print the configuration
            def print_config_section(section_name, section_data, indent=0):
                prefix = "  " * indent
                print(f"{prefix}[{section_name}]")
                for key, value in section_data.items():
                    if isinstance(value, dict):
                        print_config_section(key, value, indent + 1)
                    else:
                        print(f"{prefix}  {key} = {value}")
                print()

            for section, data in default_config.items():
                print_config_section(section, data)

            # Save defaults to file
            defaults_file = output_dir / "default_config.toml"
            with open(defaults_file, 'w') as f:
                toml.dump(default_config, f)
            print(f"   💾 Default configuration saved: {defaults_file}")

        # Test 4: Update Configuration
        if args.update_config:
            print(f"\n🔧 Updating configuration file: {args.update_config}")

            config_path = Path(args.update_config)
            if not config_path.exists():
                print(
                    f"   ❌ Configuration file not found: {args.update_config}")
                sys.exit(1)

            if not args.config_updates:
                print(f"   ❌ No updates specified. Use --set section.key=value")
                sys.exit(1)

            # Load existing configuration
            config_manager = ConfigManager(str(config_path))
            config_manager.load_config()

            # Parse and apply updates
            updates = {}
            for update_str in args.config_updates:
                try:
                    key_path, value = update_str.split('=', 1)

                    # Parse nested keys (e.g., "sports2d.mode")
                    keys = key_path.split('.')

                    # Try to parse value as appropriate type
                    if value.lower() in ['true', 'false']:
                        value = value.lower() == 'true'
                    elif value.replace('.', '').replace('-', '').isdigit():
                        value = float(value) if '.' in value else int(value)

                    # Build nested update dictionary
                    current = updates
                    for key in keys[:-1]:
                        if key not in current:
                            current[key] = {}
                        current = current[key]
                    current[keys[-1]] = value

                    print(f"   🔧 Setting {key_path} = {value}")

                except ValueError:
                    print(f"   ❌ Invalid update format: {update_str}")
                    print(f"       Expected format: section.key=value")
                    continue

            # Apply updates
            config_manager.update_config(updates)

            # Save updated configuration
            success = config_manager.save_config()

            if success:
                print(f"   ✅ Configuration updated successfully!")

                # Show updated analysis config
                analysis_config = config_manager.get_analysis_config()
                print(f"   📋 Updated settings:")
                print(f"      Sports2D mode: {analysis_config.mode}")
                print(f"      Pose model: {analysis_config.pose_model}")
                print(f"      GPU enabled: {analysis_config.use_gpu}")
                print(
                    f"      YOLO confidence: {analysis_config.yolo_confidence_threshold}")
            else:
                print(f"   ❌ Failed to save updated configuration")

        # Test 5: Export Sports2D Configuration
        if args.export_sports2d:
            print(
                f"\n🎯 Exporting Sports2D configuration for video: {args.export_sports2d}")

            # Create config manager with default or specified config
            config_file = None
            if args.validate_config and Path(args.validate_config).exists():
                config_file = args.validate_config
            elif args.update_config and Path(args.update_config).exists():
                config_file = args.update_config

            config_manager = ConfigManager(config_file)
            analysis_config = config_manager.get_analysis_config()

            # Create Sports2D configuration
            video_path = args.export_sports2d
            result_dir = output_dir / "sports2d_export"
            result_dir.mkdir(exist_ok=True)

            sports2d_config = analysis_config.to_sports2d_config(
                video_path, str(result_dir))

            # Save Sports2D configuration
            sports2d_file = output_dir / "sports2d_config.json"
            with open(sports2d_file, 'w') as f:
                json.dump(sports2d_config, f, indent=2)

            print(f"   ✅ Sports2D configuration exported!")
            print(f"   📁 Configuration file: {sports2d_file}")
            print(f"   📋 Key Sports2D settings:")
            print(
                f"      Video input: {sports2d_config['base']['video_input']}")
            print(
                f"      Result directory: {sports2d_config['base']['result_dir']}")
            print(f"      Pose model: {sports2d_config['pose']['pose_model']}")
            print(f"      Mode: {sports2d_config['pose']['mode']}")
            print(f"      Device: {sports2d_config['pose']['device']}")

        # Default behavior if no arguments
        if not any([args.create_config, args.validate_config, args.show_defaults,
                   args.update_config, args.export_sports2d]):
            print("\n🎯 Running default tests...")

            # Test basic configuration management
            config_manager = ConfigManager()

            # Test loading default configuration
            print("   🔧 Testing default configuration loading...")
            config_data = config_manager.load_config()
            analysis_config = config_manager.get_analysis_config()

            print(f"   ✅ Default configuration loaded!")
            print(f"      Sports2D mode: {analysis_config.mode}")
            print(f"      Pose model: {analysis_config.pose_model}")
            print(
                f"      YOLO confidence: {analysis_config.yolo_confidence_threshold}")

            # Test configuration updates
            print("   🔧 Testing configuration updates...")
            updates = {
                'sports2d': {'mode': 'accurate'},
                'yolo': {'confidence_threshold': 0.3}
            }
            config_manager.update_config(updates)
            updated_config = config_manager.get_analysis_config()

            print(f"   ✅ Configuration updates applied!")
            print(f"      Updated mode: {updated_config.mode}")
            print(
                f"      Updated YOLO confidence: {updated_config.yolo_confidence_threshold}")

            # Test Sports2D export
            print("   🔧 Testing Sports2D configuration export...")
            sports2d_config = updated_config.to_sports2d_config(
                "test_video.mp4", str(output_dir))

            export_file = output_dir / "default_sports2d_config.json"
            with open(export_file, 'w') as f:
                json.dump(sports2d_config, f, indent=2)

            print(f"   ✅ Sports2D configuration exported: {export_file}")

            # Create a sample configuration file
            sample_config_file = output_dir / "sample_config.toml"
            config_manager.config_path = str(sample_config_file)
            config_manager.save_config()
            print(
                f"   ✅ Sample configuration file created: {sample_config_file}")

        print(f"\n🎉 All tests completed successfully!")
        print(f"📁 Check output directory: {output_dir}")
        print("\n📖 Usage examples:")
        print("   python utils/config_manager.py --create-config config.toml")
        print("   python utils/config_manager.py --validate-config config.toml")
        print("   python utils/config_manager.py --show-defaults")
        print("   python utils/config_manager.py --update-config config.toml --set sports2d.mode=accurate")

    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
