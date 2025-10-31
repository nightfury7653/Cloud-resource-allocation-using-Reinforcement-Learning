import yaml
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigLoader:
    """
    Handles loading and validation of configuration files
    """
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load a configuration file
        
        Args:
            config_name: Name of the config file (without .yaml extension)
            
        Returns:
            dict: Configuration dictionary
        """
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        return config
        
    def load_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all configuration files in the config directory
        
        Returns:
            dict: Dictionary of configuration dictionaries
        """
        configs = {}
        
        for config_file in self.config_dir.glob("*.yaml"):
            config_name = config_file.stem
            configs[config_name] = self.load_config(config_name)
            
        return configs
        
    def save_config(self, config: Dict[str, Any], config_name: str):
        """
        Save a configuration dictionary to file
        
        Args:
            config: Configuration dictionary
            config_name: Name of the config file (without .yaml extension)
        """
        config_path = self.config_dir / f"{config_name}.yaml"
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
    def update_config(
        self,
        config_name: str,
        updates: Dict[str, Any],
        create_if_missing: bool = False
    ):
        """
        Update an existing configuration file
        
        Args:
            config_name: Name of the config file
            updates: Dictionary of updates to apply
            create_if_missing: Whether to create the config if it doesn't exist
        """
        try:
            config = self.load_config(config_name)
        except FileNotFoundError:
            if create_if_missing:
                config = {}
            else:
                raise
                
        # Recursively update the configuration
        def update_dict(d: dict, u: dict):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    update_dict(d[k], v)
                else:
                    d[k] = v
                    
        update_dict(config, updates)
        self.save_config(config, config_name)
        
    def validate_config(
        self,
        config: Dict[str, Any],
        required_fields: Optional[Dict[str, type]] = None
    ) -> bool:
        """
        Validate a configuration dictionary
        
        Args:
            config: Configuration dictionary to validate
            required_fields: Dictionary of required fields and their types
            
        Returns:
            bool: True if configuration is valid
        """
        if required_fields is None:
            return True
            
        for field, field_type in required_fields.items():
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
                
            if not isinstance(config[field], field_type):
                raise TypeError(
                    f"Field {field} has wrong type. "
                    f"Expected {field_type}, got {type(config[field])}"
                )
                
        return True
        
    @staticmethod
    def merge_configs(
        base_config: Dict[str, Any],
        override_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries
        
        Args:
            base_config: Base configuration
            override_config: Configuration to override with
            
        Returns:
            dict: Merged configuration
        """
        merged = base_config.copy()
        
        def merge_dict(d1: dict, d2: dict):
            for k, v in d2.items():
                if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                    merge_dict(d1[k], v)
                else:
                    d1[k] = v
                    
        merge_dict(merged, override_config)
        return merged
