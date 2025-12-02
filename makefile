.PHONY: clean test pyright

flist = $(wildcard RISE/figures/figure*.py)
allOutput = $(patsubst RISE/figures/figure%.py, output/figure%.svg, $(flist))

all: $(allOutput)

allThomson: $(filter output/figureThomson%, $(allOutput))

allLupus: $(filter output/figureLupus%, $(allOutput))

output/figure%.svg: RISE/figures/figure%.py
	@ mkdir -p ./output
	uv run fbuild $*

test: .venv
	uv run pytest -s -v -x

.venv: pyproject.toml
	uv sync

coverage.xml: .venv
	uv run pytest --junitxml=junit.xml --cov=RISE --cov-report xml:coverage.xml

pyright: .venv
	uv run pyright RISE

clean:
	rm -rf output profile profile.svg
	rm -rf factor_cache
