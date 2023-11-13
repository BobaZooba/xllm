from examples.minimal_using_cli.dataset import AntropicDataset
from src.xllm.cli.train import cli_run_train
from src.xllm.core.config import Config
from src.xllm.datasets.registry import datasets_registry

if __name__ == "__main__":
    datasets_registry.add(key="antropic", value=AntropicDataset)
    cli_run_train(config_cls=Config)
