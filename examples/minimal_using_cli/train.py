from xllm.cli.train import cli_run_train
from xllm.core.config import Config
from xllm.datasets.registry import datasets_registry

from examples.minimal_using_cli.dataset import AntropicDataset

if __name__ == "__main__":
    datasets_registry.add(key="antropic", value=AntropicDataset)
    cli_run_train(config_cls=Config)
