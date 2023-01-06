SHELL := /bin/bash

.PHONY: clean test

flist = $(wildcard sccp/figures/figure*.py)

all: $(patsubst sccp/figures/figure%.py, output/figure%.svg, $(flist))

output/figure%.svg: sccp/figures/figure%.py
	@ mkdir -p ./output
	poetry run fbuild $*

test:
	poetry run pytest -s -x -v

coverage.xml:
	poetry run pytest --cov=sccp --cov-report=xml

clean:
	rm -rf output

testprofile:
	poetry run python3 -m cProfile -o profile -m pytest -s -v -x

mypy:
	poetry run mypy --install-types --non-interactive --ignore-missing-imports sccp
