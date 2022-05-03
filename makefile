SHELL := /bin/bash

.PHONY: clean test

flist = $(wildcard gmm/figures/figure*.py)

all: $(patsubst gmm/figures/figure%.py, output/figure%.svg, $(flist))

output/figure%.svg: gmm/figures/figure%.py
	@ mkdir -p ./output
	poetry run fbuild $*

test:
	poetry run pytest -s -x -v

clean:
	rm -rf output

testprofile:
	poetry run python3 -m cProfile -o profile -m pytest -s -v -x

mypy:
	poetry run mypy --install-types --non-interactive --ignore-missing-imports gmm
