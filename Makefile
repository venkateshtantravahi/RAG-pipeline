#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = rag-pipeline
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	pip install -e .
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) ragpipeline/dataset.py


## Ingest Data into local vector db
.PHONY: ingest
ingest: requirements
	$(PYTHON_INTERPRETER) ragpipeline/ingestion.py


## Retriever search for a querying top_k results
.PHONY: retrieve
retrieve: requirements
	$(PYTHON_INTERPRETER) ragpipeline/retrieval.py


## Generator Module Testing for RAG Pipeline
.PHONY: generate
generate: requirements
	$(PYTHON_INTERPRETER) ragpipeline/generation.py

#################################################################
# Single Shot Setup & Run
################################################################

## Single setup command to install all dependencies required
.PHONY: setup
setup: requirements
	@echo "Pulling the Specific LLM model (llama3.2:1b)..."
	ollama pull llama3.2:1b
	@echo "Ingesting data into Vector Database..."
	python -m ragpipeline.ingestion

## Run will start the api server for rag pipeline
.PHONY: run
run:
	@echo "Starting FASTAPI Server..."
	uvicorn ragpipeline.api:app --reload --host 0.0.0.0 --port 8000

## Run streamlit server
.PHONY: run-streamlit
run-streamlit:
	@echo "Starting Streamlit Server..."
	streamlit run frontend/app.py

## start-all will spin up the database load data and spin up the server for requests.
.PHONY: start-all
start-all: setup run

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
