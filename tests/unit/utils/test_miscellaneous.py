from typing import Any, Dict, List

import pytest

from src.xllm.utils.miscellaneous import have_missing_keys


@pytest.mark.parametrize(
    argnames=["data", "must_have_keys"],
    argvalues=[
        ({"a": 1}, ["a"]),
        ({"a": 1, "b": 2}, ["a", "b"]),
        ({"a": 1, "b": 2, "c": 3}, ["a", "b", "c"]),
    ],
)
def test_no_have_missing_keys(data: Dict[str, Any], must_have_keys: List[str]) -> None:
    flag, difference = have_missing_keys(data=data, must_have_keys=must_have_keys)
    assert not flag
    assert len(difference) == 0


@pytest.mark.parametrize("data", [{"a": 1}, {"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 3}])
@pytest.mark.parametrize("must_have_keys", [["c", "d", "e"], ["1", "2", "3"]])
def test_have_missing_keys(data: Dict[str, Any], must_have_keys: List[str]) -> None:
    flag, difference = have_missing_keys(data=data, must_have_keys=must_have_keys)
    assert flag
    assert len(difference) > 0
