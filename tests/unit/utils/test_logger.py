import pytest

from src.xllm import enums
from src.xllm.utils.logger import dist_logger


@pytest.mark.parametrize("message", ["Hello world", "Hey!", "All good"])
def test_info(message: str):
    dist_logger.info(message=message)


@pytest.mark.parametrize("message", ["Hello world", "Hey!", "All good"])
def test_waring(message: str):
    dist_logger.warning(message=message)


@pytest.mark.parametrize("message", ["Hello world", "Hey!", "All good"])
def test_error(message: str):
    dist_logger.error(message=message)


@pytest.mark.parametrize("message", ["Hello world", "Hey!", "All good"])
def test_critical(message: str):
    dist_logger.critical(message=message)


@pytest.mark.parametrize("message", ["Hello world", "Hey!", "All good"])
@pytest.mark.parametrize(
    "level",
    [
        enums.LogLevel.info,
        enums.LogLevel.warning,
        enums.LogLevel.error,
        enums.LogLevel.critical,
        None,
        "some",
    ],
)
def test_log(message: str, level: str):
    dist_logger.log(message=message, level=level)


@pytest.mark.parametrize("message", ["Hello world", "Hey!", "All good"])
@pytest.mark.parametrize(
    "level",
    [
        enums.LogLevel.info,
        enums.LogLevel.warning,
        enums.LogLevel.error,
        enums.LogLevel.critical,
        None,
        "some",
    ],
)
def test_call(message: str, level: str):
    dist_logger(message=message, level=level)
