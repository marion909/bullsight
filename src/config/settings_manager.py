"""
Settings manager for persistent application configuration.

Handles loading and saving user settings like sound volume, sound enabled state,
and other preferences across application restarts.

Author: Mario Neuhauser
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SettingsManager:
    """
    Manages persistent application settings.
    
    Settings are stored in config/settings.json and automatically
    loaded at startup and saved when changed.
    
    Default Settings:
    - sound_enabled: True
    - sound_volume: 70
    - fullscreen: True
    - language: "en"
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize settings manager.
        
        Args:
            config_path: Path to settings file (default: config/settings.json)
        """
        if config_path is None:
            config_path = Path("config/settings.json")
        
        self.config_path = config_path
        self.settings = self._default_settings()
        self.load()
    
    def _default_settings(self) -> dict:
        """Get default settings."""
        return {
            "sound_enabled": True,
            "sound_volume": 70,
            "fullscreen": True,
            "language": "en"
        }
    
    def load(self) -> bool:
        """
        Load settings from file.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.config_path.exists():
            logger.info("No settings file found, using defaults")
            return False
        
        try:
            with open(self.config_path, 'r') as f:
                loaded_settings = json.load(f)
            
            # Merge with defaults (in case new settings were added)
            self.settings.update(loaded_settings)
            logger.info(f"Settings loaded from {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
            return False
    
    def save(self) -> bool:
        """
        Save current settings to file.
        
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Ensure config directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                json.dump(self.settings, f, indent=2)
            
            logger.info(f"Settings saved to {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get setting value.
        
        Args:
            key: Setting key
            default: Default value if key not found
            
        Returns:
            Setting value or default
        """
        return self.settings.get(key, default)
    
    def set(self, key: str, value: Any, save_immediately: bool = True) -> None:
        """
        Set setting value.
        
        Args:
            key: Setting key
            value: New value
            save_immediately: Whether to save to file immediately (default: True)
        """
        self.settings[key] = value
        
        if save_immediately:
            self.save()
    
    def reset(self) -> None:
        """Reset all settings to defaults."""
        self.settings = self._default_settings()
        self.save()
        logger.info("Settings reset to defaults")
