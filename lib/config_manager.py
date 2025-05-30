"""
Configuration management for API keys and other settings.
Supports loading from config files and environment variables.
"""

import os
import json
import yaml
import logging
import configparser
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

# Configure logging
logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Manages configuration settings with support for:
    - Default values
    - Config files (JSON, YAML, INI)
    - Environment variables
    - Command-line overrides
    
    Environment variables take precedence over config files.
    """
    
    def __init__(self, 
                 config_paths: Optional[List[str]] = None,
                 env_prefix: str = "",
                 default_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the configuration manager
        
        Args:
            config_paths: List of paths to look for config files
            env_prefix: Prefix for environment variables
            default_config: Default configuration values
        """
        self.config_paths = config_paths or []
        self.env_prefix = env_prefix
        self.config = default_config or {}
        
        # Default API keys that we look for
        self.api_keys = [
            "DEEPGRAM_API_KEY",
            "GROQ_API_KEY",
            "OPENAI_API_KEY", 
            "ELEVENLABS_API_KEY"
        ]
        
        # Add some standard config file locations
        self._add_default_config_paths()
        
        # Load configuration from files and environment
        self.reload()
    
    def _add_default_config_paths(self):
        """Add default config file paths to search"""
        # Current directory
        self.config_paths.append("./config.json")
        self.config_paths.append("./config.yaml")
        self.config_paths.append("./config.yml")
        self.config_paths.append("./config.ini")
        self.config_paths.append("./.env")
        
        self.config_paths.append(os.path.join("../config", "config.json"))
        self.config_paths.append(os.path.join("../config", "config.yaml"))
        self.config_paths.append(os.path.join("../config", "config.yml"))
        self.config_paths.append(os.path.join("../config", "config.ini"))
        self.config_paths.append(os.path.join("../config", ".env"))

        self.config_paths.append(os.path.join("./config", "config.json"))
        self.config_paths.append(os.path.join("./config", "config.yaml"))
        self.config_paths.append(os.path.join("./config", "config.yml"))
        self.config_paths.append(os.path.join("./config", "config.ini"))
        self.config_paths.append(os.path.join("./config", ".env"))

        # User's home directory
        home = str(Path.home())
        self.config_paths.append(os.path.join(home, ".voicebot", "config.json"))
        self.config_paths.append(os.path.join(home, ".voicebot", "config.yaml"))
        self.config_paths.append(os.path.join(home, ".voicebot", "config.yml"))
        self.config_paths.append(os.path.join(home, ".voicebot", "config.ini"))
        self.config_paths.append(os.path.join(home, ".voicebot", ".env"))
        
        # System-wide config
        self.config_paths.append("/etc/voicebot/config.json")
        self.config_paths.append("/etc/voicebot/config.yaml")
        self.config_paths.append("/etc/voicebot/config.yml")
        self.config_paths.append("/etc/voicebot/config.ini")
        self.config_paths.append("/etc/voicebot/.env")
    
    def reload(self):
        """Reload configuration from all sources"""
        # Start with default config
        config = self.config.copy()
        
        # Load from config files
        for path in self.config_paths:
            if os.path.exists(path):
                logger.debug(f"Loading config from: {path}")
                try:
                    file_config = self._load_config_file(path)
                    if file_config:
                        config.update(file_config)
                        logger.info(f"Loaded configuration from {path}")
                except Exception as e:
                    logger.warning(f"Error loading config from {path}: {e}")
        
        # Override with environment variables
        config = self._load_from_env(config)
        
        # Update the config
        self.config = config
        return self.config
    
    def _load_config_file(self, path: str) -> Dict[str, Any]:
        """Load configuration from a file based on its extension"""
        if not os.path.exists(path):
            return {}
            
        ext = os.path.splitext(path)[1].lower()
        
        try:
            if ext == '.json':
                with open(path, 'r') as f:
                    return json.load(f)
                    
            elif ext in ['.yaml', '.yml']:
                with open(path, 'r') as f:
                    return yaml.safe_load(f)
                    
            elif ext == '.ini':
                config = configparser.ConfigParser()
                config.read(path)
                # Convert to dict
                result = {}
                for section in config.sections():
                    for key, value in config[section].items():
                        # For API keys, use uppercase
                        if key.upper() in self.api_keys:
                            result[key.upper()] = value
                        else:
                            # For other settings, use lowercase
                            result[key.lower()] = value
                return result
                
            elif os.path.basename(path) == '.env':
                # Parse .env file
                result = {}
                with open(path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        key, value = line.split('=', 1)
                        # For API keys, use uppercase
                        if key.upper() in self.api_keys:
                            result[key.upper()] = value
                        else:
                            result[key] = value
                return result
                
        except Exception as e:
            logger.warning(f"Error reading config file {path}: {e}")
            
        return {}
    
    def _load_from_env(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        # Look for API keys in environment
        for key in self.api_keys:
            env_key = f"{self.env_prefix}{key}" if self.env_prefix else key
            if env_key in os.environ:
                config[key] = os.environ[env_key]
                logger.debug(f"Loaded {key} from environment variable")
                
        # Look for other voicebot config in environment
        prefix = f"{self.env_prefix}VOICEBOT_" if self.env_prefix else "VOICEBOT_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Strip prefix and convert to lowercase for regular settings
                config_key = key[len(prefix):].lower()
                config[config_key] = value
                logger.debug(f"Loaded {config_key} from environment variable")
                
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value"""
        self.config[key] = value
    
    def get_api_key(self, service: str) -> Optional[str]:
        """
        Get an API key for a specific service
        
        Args:
            service: Service name (deepgram, elevenlabs, openai, groq)
            
        Returns:
            API key or None if not found
        """
        key_name = f"{service.upper()}_API_KEY"
        if key_name not in self.api_keys:
            self.api_keys.append(key_name)
            
        return self.get(key_name)

    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access to configuration"""
        return self.config.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Dictionary-style setting of configuration"""
        self.config[key] = value
    
    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the configuration"""
        return key in self.config
    
    def save(self, path: str) -> bool:
        """Save the current configuration to a file"""
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                
            ext = os.path.splitext(path)[1].lower()
            
            if ext == '.json':
                with open(path, 'w') as f:
                    json.dump(self.config, f, indent=2)
                    
            elif ext in ['.yaml', '.yml']:
                with open(path, 'w') as f:
                    yaml.dump(self.config, f)
                    
            elif ext == '.ini':
                config = configparser.ConfigParser()
                config['DEFAULT'] = {}
                
                # Add API keys to DEFAULT section
                for key in self.api_keys:
                    if key in self.config:
                        config['DEFAULT'][key] = self.config[key]
                
                # Add other settings
                for key, value in self.config.items():
                    if key not in self.api_keys and isinstance(value, (str, int, float, bool)):
                        config['DEFAULT'][key] = str(value)
                
                with open(path, 'w') as f:
                    config.write(f)
                    
            elif os.path.basename(path) == '.env':
                with open(path, 'w') as f:
                    for key, value in self.config.items():
                        if isinstance(value, (str, int, float, bool)):
                            f.write(f"{key}={value}\n")
                            
            else:
                logger.warning(f"Unsupported file extension for {path}")
                return False
                
            logger.info(f"Configuration saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration to {path}: {e}")
            return False

# Global config instance
config = ConfigManager()

def load_config(path: Optional[str] = None, env_prefix: str = "") -> ConfigManager:
    """
    Load configuration from a specific file or use the global instance
    
    Args:
        path: Path to a config file (optional)
        env_prefix: Prefix for environment variables
        
    Returns:
        ConfigManager instance
    """
    global config
    
    if path:
        # Create a new instance with the specified path
        config_paths = [path]
        config = ConfigManager(config_paths=config_paths, env_prefix=env_prefix)
    elif env_prefix:
        # Update the prefix of the existing instance
        config.env_prefix = env_prefix
        config.reload()
        
    return config

def get_api_key(service: str) -> Optional[str]:
    """
    Get an API key for a specific service from the global config
    
    Args:
        service: Service name (deepgram, elevenlabs, openai, groq)
        
    Returns:
        API key or None if not found
    """
    global config
    return config.get_api_key(service)

def init_from_args(args) -> ConfigManager:
    """
    Initialize configuration from command-line arguments
    
    Args:
        args: Command-line arguments with config_file and other options
        
    Returns:
        ConfigManager instance
    """
    global config
    
    # Check for config file in args
    if hasattr(args, 'config_file') and args.config_file:
        config_paths = [args.config_file]
        config = ConfigManager(config_paths=config_paths)
    
    # Override with direct arguments
    for key in config.api_keys:
        arg_key = key.lower()
        if hasattr(args, arg_key) and getattr(args, arg_key):
            config[key] = getattr(args, arg_key)
    
    return config
