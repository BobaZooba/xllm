# Copyright 2023 Boris Zubarev. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, Optional

from loguru import logger


class Registry:
    """
    The `Registry` class provides a generic structure to register and access objects using string-based keys. It is a
    useful component in various software design patterns, such as for creating factories, or when there's a need to
    maintain a centralized registry of classes, functions, or values.

    Attributes:
        DEFAULT_KEY (`str`): A class constant used to store the default value with a key in the registry.
        name (`str`): The name given to the registry instance, used for identification and logging purposes.
        mapper (`Dict[str, Any]`): A dictionary to store the mapping of string keys to values.

    Methods:
        - `__init__`: Constructs a new `Registry` instance, initializing the mapper and optionally setting a default
            value.
        - `add`: Registers a new key-value pair in the registry with an optional ability to override existing entries.
        - `get`: Retrieves a value from the registry based on a provided key, or the default value if the key
            is not found.

    The `Registry` class enables the management of a collection of items that can be registered and retrieved
    dynamically at runtime. It is versatile in its usage and can adapt to various use cases where a key-based retrieval
    system is needed.

    Example usage:
        # Creating an instance of Registry without a default value.
        my_registry = Registry(name="MyRegistry")

        # Registering an object with a unique key in the registry.
        my_registry.add(key="example_object", value=SomeObject)

        # Retrieving the registered object using its key.
        retrieved_object = my_registry.get(key="example_object")

    Note: When attempting to add an object to the registry with an existing key without overriding, a `ValueError`
    will be raised to prevent accidental overwrites. Similarly, when retrieving a non-existing key without a
    default value, a `ValueError` will be raised.
    """

    DEFAULT_KEY = "default"

    def __init__(self, name: str, default_value: Optional[Any] = None):
        """
        Initializes a new instance of the Registry class.

        The Registry class serves as a store for mapping string keys to various values, typically used for
        registering classes, functions, or objects in a central place. It provides methods to add new key-value
        pairs and to retrieve values by their keys. An optional default value can be set at initialization,
        which is returned when a requested key is not found and a default is available.

        Args:
            name (`str`):
                The name of the registry. This can be used to identifier the specific instance of the registry.
            default_value (`Optional[Any]`, defaults to `None`):
                The default value to return if a key is not found in the registry.
                If `None`, no default value will be set.

        The `__init__` method sets up the `name` of the registry and initializes an internal dictionary `mapper`.
        If the `default_value` is provided, it is added to the `mapper` with the key `DEFAULT_KEY`, which is a
        constant defined in the class. The `mapper` dictionary is then used to store the key-value pairs added
        to the registry through the `add` method, and to retrieve them using the `get` method.

        Raises:
            ValueError: if the `add` method is called with `override` set to `False` and the key already exists in
                        the `mapper` dictionary.

        Example:
            # Creating a registry with no default value
            function_registry = Registry("FunctionRegistry")

            # Creating a registry with a default value
            class_registry = Registry("ClassRegistry", default_value=DefaultClass)
        """
        self.name = name

        self.mapper: Dict[str, Any] = dict()

        if default_value is not None:
            self.mapper[self.DEFAULT_KEY] = default_value

    def add(self, key: str, value: Any, override: bool = False) -> None:
        """
        Adds a new key-value pair to the registry. If the key already exists in the registry, the behavior depends
        on the `override` flag. By default, the method raises a ValueError if an attempt is made to add a key that
        already exists without setting the `override` flag to `True`.

        Args:
            key (`str`):
                The key associated with the value to be added to the registry. The key is used to retrieve
                the value later using the `get` method.
            value (`Any`):
                The value to be associated with the provided key. This could be any object, such as a class,
                function, or instance.
            override (`bool`, defaults to `False`):
                Determines whether to override the existing value in the registry if the key already exists.
                If `True`, the existing value will be replaced with the new value. If `False` and the key
                already exists, a ValueError is raised.

        Raises:
            ValueError: If `override` is `False` and the key already exists in the registry.

        Example:
            # Adding a key-value pair without override. If the key already exists, it will raise a ValueError.
            my_registry.add("my_key", my_value)

            # Adding a key-value pair with override. If the key exists, its value will be replaced.
            my_registry.add("my_key", new_value, override=True)
        """
        if not override and key in self.mapper:
            raise ValueError(f"Key exist in {self.name} registry")
        self.mapper[key] = value
        return None

    def get(self, key: Optional[str] = None) -> Any:
        """
        Retrieves a value from the registry by its associated key. When a key is provided, the method looks up
        the value in the registry's mapper dictionary. If the key is not found, the method attempts to retrieve
        the default value set during initialization.

        If a key is not provided or found, and a default value is available, a warning is logged and the default
        value is returned. If neither the key nor a default value is present, a ValueError is raised.

        Args:
            key (`Optional[str]`, defaults to `None`):
                The key for which to retrieve the associated value from the registry.
                If `None`, the default value is retrieved.

        Returns:
            Any: The value associated with the provided key or the default value if the key is not found.

        Raises:
            ValueError: If the key is not found in the registry and no default value is available.

        Example:
            # Retrieving a value using an existing key
            value = my_registry.get("my_key")

            # Retrieving the default value when key is not provided or not found
            default_value = my_registry.get(None)  # or my_registry.get("non_existent_key")

        Note:
            When the default value is returned due to a missing key, a warning is logged using the `loguru` logger to
            indicate that the default item from the registry was chosen.
        """
        if key is not None:
            value = self.mapper.get(key, None)
        else:
            value = self.mapper.get(self.DEFAULT_KEY, None)

        if value is None:
            value = self.mapper.get(self.DEFAULT_KEY, None)
            if value is not None:
                logger.warning(f"Default item {value.__name__} chosen from {self.name} registry")

        if value is None:
            raise ValueError(f"Item with key {key} not found in {self.name} registry")

        return value
