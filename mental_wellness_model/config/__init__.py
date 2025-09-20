from pathlib import Path
import yaml
from typing import Dict, Any

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file. If None, uses default config.
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent / "default.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

__all__ = ['load_config']