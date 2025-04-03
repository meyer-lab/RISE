.PHONY: clean test pyright

flist = $(wildcard RISE/figures/figure*.py)
allOutput = $(patsubst RISE/figures/figure%.py, output/figure%.svg, $(flist))

all: $(allOutput)

allThomson: $(filter output/figureThomson%, $(allOutput))

allLupus: $(filter output/figureLupus%, $(allOutput))

output/figure%.svg: RISE/figures/figure%.py
	@ mkdir -p ./output
	rye run fbuild $*

test: .venv
	rye run pytest -s -v -x

.venv: pyproject.toml
	rye sync

coverage.xml: .venv
	rye run pytest --junitxml=junit.xml --cov=RISE --cov-report xml:coverage.xml

pyright: .venv
	rye run pyright RISE

clean:
	rm -rf output profile profile.svg
	rm -rf factor_cache
