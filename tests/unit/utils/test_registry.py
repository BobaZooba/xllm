import pytest

from src.xllm.collators.lm import LMCollator
from src.xllm.utils.registry import Registry


def test_registry():
    registry = Registry(name="tmp", default_value=LMCollator)
    registry.add(key="a", value=1)
    a_value = registry.get(key="a")
    assert a_value == 1
    default_value = registry.get(key="b")
    default_value_from_none = registry.get(key=None)
    assert default_value == default_value_from_none


def test_registry_without_default():
    registry = Registry(name="tmp")
    with pytest.raises(ValueError):
        registry.get(key="a")
