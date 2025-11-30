import yaml
import os
from pathlib import Path
from dotenv import load_dotenv


class Config:
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.root = Path(__file__).resolve().parents[2]
        self.config_path = self.root / config_path

        # Load YAML
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.data = yaml.safe_load(f)

        # Load secrets
        load_dotenv(self.root / "config" / "secrets.env")

    def get(self, path: str, default=None):
        keys = path.split(".")
        value = self.data
        for k in keys:
            if k not in value:
                return default
            value = value[k]
        return value

    def secret(self, key: str):
        return os.getenv(key)


config = Config()
