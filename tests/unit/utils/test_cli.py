from src.xllm.core.config import Config
from src.xllm.utils.cli import setup_cli


def test_setup_cli(config: Config):
    setup_cli(config=config)
