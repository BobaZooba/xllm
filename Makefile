.DEFAULT_GOAL:=help

.EXPORT_ALL_VARIABLES:

ifndef VERBOSE
.SILENT:
endif

#* Variables
PYTHON := python3
PYTHON_RUN := $(PYTHON) -m
VERSION := 0.0.0

#* Docker variables
IMAGE := xllm
DOCKER_VERSION := latest

help:  ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z0-9_-]+:.*?##/ { printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

#* Installation
.PHONY: install
install:  ## Installation
	$(PYTHON_RUN) pip install -U .

.PHONY: all-install
all-install:  ## All deps installation
	$(PYTHON_RUN) pip install -U .[all]

.PHONY: dev-install
dev-install:  ## Developer installation
	$(PYTHON_RUN) pip install -U .[dev]

.PHONY: pre-commit-install
pre-commit-install:  ## Install pre-commit hooks
	pre-commit install

#* Docker
# Example: make docker-build DOCKER_VERSION=latest
# Example: make docker-build IMAGE=some_name DOCKER_VERSION=0.0.1
.PHONY: docker-build
docker-build:  ## Build docker image
	@echo Building Docker $(IMAGE):$(DOCKER_VERSION) ...
	docker build \
		-t $(IMAGE):$(DOCKER_VERSION) . \
		-f ./docker/production/Dockerfile --no-cache

# Example: make docker-remove DOCKER_VERSION=latest
# Example: make docker-remove IMAGE=some_name DOCKER_VERSION=0.0.1
.PHONY: docker-remove
docker-remove:  ## Remove docker image
	@echo Removing Docker $(IMAGE):$(DOCKER_VERSION) ...
	docker rmi -f $(IMAGE):$(DOCKER_VERSION)

#* Formatters
.PHONY: codestyle
codestyle:  ## Apply codestyle (black, ruff)
	$(PYTHON_RUN) black --config pyproject.toml .
	$(PYTHON_RUN) ruff check . --fix --preview
	
.PHONY: check-black
check-black:  ## Check black
	$(PYTHON_RUN) black --diff --check --config pyproject.toml src/xllm

.PHONY: check-ruff
check-ruff:  ## Check ruff
	$(PYTHON_RUN) ruff check src/xllm --preview

.PHONY: check-codestyle
check-codestyle:  ## Check codestyle
	make check-black
	make check-ruff

#* Tests
.PHONY: unit-tests
unit-tests:  ## Run unit tests
	$(PYTHON_RUN) pytest -c pyproject.toml --disable-warnings --cov-report=html --cov=src/xllm tests/unit

# .PHONY: integration-tests
# integration-tests:  ## Run integration tests
# 	$(PYTHON_RUN) pytest -c pyproject.toml --disable-warnings --cov-report=html --cov=src/xllm tests/integration

PHONY: ci-unit-tests
ci-unit-tests:  ## Run unit tests for CI
	$(PYTHON_RUN) pytest -c pyproject.toml --cache-clear --disable-warnings --cov-report=xml:coverage-unit.xml --cov=src/xllm tests/unit

# .PHONY: ci-integration-tests
# ci-integration-tests:  ## Run integration tests for CI
# 	$(PYTHON_RUN) pytest -c pyproject.toml --cache-clear --disable-warnings --cov-report=xml:coverage-unit.xml --cov=src/xllm tests/integration

#* Linting
.PHONY: mypy
mypy:  ## Run static code analyzer
	$(PYTHON_RUN) mypy --config-file pyproject.toml ./src/xllm

# .PHONY: check-safety
# check-safety:  ## Run safety, bandit checks
# 	# poetry run safety check --full-report
# 	poetry run bandit -ll --recursive xllm tests

.PHONY: lint
# lint: unit-test integration-test check-codestyle mypy check-safety  ## Run all checks
lint: check-codestyle mypy unit-test  ## Run all checks

#* Develop
.PHONY: push-new-version
push-new-version:  ## Push new version to the GitHub
	git add .
	git commit -m "Release: $(VERSION)"
	git tag $(VERSION) -m 'Adds tag $(VERSION) for pypi'
	git push --tags origin main

.PHONY: delete-dist
delete-dist:  ## Delete all dist builds
	rm -rf ./dist

.PHONY: build-dist
build-dist:  ## Build dist
	$(PYTHON) setup.py bdist_wheel
	$(PYTHON) setup.py sdist

.PHONY: test-pypi-upload
test-pypi-upload:  ## Upload package to the test pypi
	twine upload dist/* -r testpypi

.PHONY: pypi-upload
pypi-upload:  ## Upload package to the pypi
	twine upload dist/* -r pypi

# make test-pypi-release VERSION=0.1.0
.PHONY: test-pypi-release
test-pypi-release:  ## Release test pypi package
	@if [ "$(VERSION)" = "0.0.0" ]; then \
		echo "VERSION is equal to default 0.0.0"; \
		echo "Please specify correct version"; \
		echo "For example:"; \
		echo "make test-pypi-release VERSION=1.2.3"; \
		exit 1; \
	else \
		exit 0; \
	fi
	make codestyle
	make check-codestyle
	make mypy
	make unit-tests
	make push-new-version
	make delete-dist
	make build-dist
	make test-pypi-upload
