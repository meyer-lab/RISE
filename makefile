SHELL := /bin/bash

.PHONY: clean test

flist = $(wildcard sccp/figures/figure*.py)

all: $(patsubst sccp/figures/figure%.py, output/figure%.svg, $(flist))

output/figure%.svg: sccp/figures/figure%.py
	@ mkdir -p ./output
	poetry run fbuild $*

test:
	poetry run pytest -s -x -v --full-trace

coverage.xml:
	poetry run pytest --cov=sccp --cov-report=xml

clean:
	rm -rf output profile profile.svg

testprofile:
	poetry run python3 -m cProfile -o profile -m pytest -s -v -x
	gprof2dot -f pstats --node-thres=5.0 profile | dot -Tsvg -o profile.svg

mypy:
	poetry run mypy --install-types --non-interactive --ignore-missing-imports sccp
