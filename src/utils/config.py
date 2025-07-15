"""
Configuration utilities for Domain Name Suggestion LLM

This module provides utilities for loading and managing configuration files
across the project.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in {config_path}: {e}")


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary to save
        config_path: Path where to save the configuration
    """
    # Ensure directory exists
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get a configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to the key (e.g., "model.base_model.name")
        default: Default value if key doesn't exist
        
    Returns:
        Configuration value or default
    """
    keys = key_path.split('.')
    current = config
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    
    return current


def set_config_value(config: Dict[str, Any], key_path: str, value: Any) -> None:
    """
    Set a configuration value using dot notation.
    
    Args:
        config: Configuration dictionary to modify
        key_path: Dot-separated path to the key
        value: Value to set
    """
    keys = key_path.split('.')
    current = config
    
    # Navigate to the parent of the target key
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Set the value
    current[keys[-1]] = value


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configurations, with override_config taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override with
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> None:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                _merge_dicts(base[key], value)
            else:
                base[key] = value
    
    _merge_dicts(merged, override_config)
    return merged


def validate_config(config: Dict[str, Any], required_keys: Optional[list] = None) -> bool:
    """
    Validate configuration structure.
    
    Args:
        config: Configuration to validate
        required_keys: List of required top-level keys
        
    Returns:
        True if valid, False otherwise
    """
    if required_keys is None:
        required_keys = []
    
    for key in required_keys:
        if key not in config:
            return False
    
    return True


def get_model_config(model_version: str, config_path: str = "config/model_config.yaml") -> Dict[str, Any]:
    """
    Get configuration for a specific model version.
    
    Args:
        model_version: Version of the model
        config_path: Path to model configuration file
        
    Returns:
        Model-specific configuration
    """
    config = load_config(config_path)
    
    if model_version not in config.get('versions', {}):
        raise ValueError(f"Unknown model version: {model_version}")
    
    return config['versions'][model_version]


def get_evaluation_config(config_path: str = "config/evaluation_config.yaml") -> Dict[str, Any]:
    """
    Get evaluation configuration.
    
    Args:
        config_path: Path to evaluation configuration file
        
    Returns:
        Evaluation configuration
    """
    return load_config(config_path)


def create_default_configs() -> None:
    """
    Create default configuration files if they don't exist.
    """
    # Default model config
    model_config_path = "config/model_config.yaml"
    if not os.path.exists(model_config_path):
        default_model_config = {
            "base_model": {
                "name": "meta-llama/Llama-2-7b-hf",
                "revision": "main",
                "trust_remote_code": False,
                "use_auth_token": True
            },
            "versions": {
                "baseline": {
                    "training_method": "full_fine_tuning",
                    "learning_rate": 2e-5,
                    "batch_size": 4,
                    "num_epochs": 3
                },
                "v1_lora": {
                    "training_method": "lora",
                    "learning_rate": 3e-4,
                    "batch_size": 8,
                    "num_epochs": 5
                },
                "v2_qlora": {
                    "training_method": "qlora",
                    "learning_rate": 2e-4,
                    "batch_size": 16,
                    "num_epochs": 7
                }
            }
        }
        save_config(default_model_config, model_config_path)
    
    # Default evaluation config
    eval_config_path = "config/evaluation_config.yaml"
    if not os.path.exists(eval_config_path):
        default_eval_config = {
            "llm_judge": {
                "primary": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "temperature": 0.0,
                    "max_tokens": 1000
                }
            },
            "metrics": {
                "weights": {
                    "relevance": 0.3,
                    "memorability": 0.25,
                    "appropriateness": 0.25,
                    "availability_style": 0.2
                }
            }
        }
        save_config(default_eval_config, eval_config_path) 