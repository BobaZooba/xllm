from setuptools import find_packages, setup

# Extras
extras = dict()

extras["quality"] = [
    "black[jupyter]",
    "mypy",
    "mypy-extensions",
    "pre-commit",
    "ruff",
]

extras["docs"] = [
    "sphinx",
    "sphinx-rtd-theme",
]

extras["tests"] = [
    "pytest",
    "coverage",
    "pytest-html",
    "pytest-cov",
]

extras["build"] = [
    "wheel",
    "twine",
]

extras["dev"] = extras["quality"] + extras["tests"] + extras["build"] + extras["docs"]

extras["deepspeed"] = ["deepspeed"]

extras["auto-gptq"] = ["auto-gptq"]

extras["flash-attn"] = ["flash-attn>=2.2.1"]

extras["train"] = extras["deepspeed"] + extras["flash-attn"] + extras["auto-gptq"]

extras["all"] = extras["train"] + extras["dev"]

# Install deps
install_requires = [
    "numpy>=1.17",
    "packaging>=20.0",
    "psutil",
    "torch>=2.0.1",
    "loguru",
    "peft>=0.5.0",
    "wandb",
    "python-dotenv",
    "requests",
    "optimum>=1.12.0",
    "bitsandbytes>=0.41.1",
    "scipy",
    "transformers",
    "tqdm",
    "safetensors",
]

# Setup
setup(
    name="xllm",
    version="0.1.6",
    description="Simple & Cutting Edge LLM Finetuning",
    license_files=["LICENSE"],
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="ai nlp llm text deep-learning",
    license="Apache",
    author="BobaZooba",
    author_email="bobazooba@gmail.com",
    url="https://github.com/bobazooba/xllm",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"xllm": ["py.typed"]},
    entry_points={},
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Legal Industry",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
        "Environment :: GPU :: NVIDIA CUDA :: 11.8",
        "Framework :: Jupyter",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)

# Release checklist
# 1. Change the version in __init__.py and setup.py
# 2. Run "make test-pypi-release VERSION=VERSION"
# 3. Make sure test pypi works correct
# 4. Run "make pypi-upload"
# 5. Add release notes to the tag in github once everything is looking hunky-dory
# 6. Change the version to dev "-dev" in __init__.py and setup.py
# 7. Add new dev "make push-dev-version VERSION=VERSION"

# Legacy Release checklist
# 1. Change the version in __init__.py and setup.py.
# 2. Commit these changes with the message: "Release: VERSION"
# 3. Add a tag in git to mark the release: "git tag VERSION -m 'Adds tag VERSION for pypi' "
#    Push the tag to git: git push --tags origin main
# 4. Run the following commands in the top-level directory:
#      [Optional] rm -rf dist/
#      python setup.py bdist_wheel
#      python setup.py sdist
# 5. Upload the package to the pypi test server first:
#      twine upload dist/* -r testpypi
# 6. Check that you can install it in a virtualenv by running:
#      pip install --upgrade -i https://testpypi.python.org/pypi xllm
# 7. Upload the final version to actual pypi:
#      twine upload dist/* -r pypi
# 8. Add release notes to the tag in github once everything is looking hunky-dory.
# 9. Update the version in __init__.py, setup.py to the new version "-dev" and push to master
