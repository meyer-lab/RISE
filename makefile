SHELL := /bin/bash

.PHONY: clean test

flist = $(wildcard gmm/figures/figure*.py)

all: $(patsubst gmm/figures/figure%.py, output/figure%.svg, $(flist))

output/figure%.svg: gmm/figures/figure%.py
	@ mkdir -p ./output
	XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 poetry run fbuild $*

test:
	XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 poetry run pytest -s -x -v

coverage.xml:
	XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 poetry run pytest --cov=gmm --cov-report=xml

clean:
	rm -rf output

testprofile:
	XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 poetry run python3 -m cProfile -o profile -m pytest -s -v -x

mypy:
	poetry run mypy --install-types --non-interactive --ignore-missing-imports gmm
