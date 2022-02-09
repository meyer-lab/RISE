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
