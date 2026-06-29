.PHONY: clean test ty

flist = $(wildcard analysis/figures/figure*.py)
allOutput = $(patsubst analysis/figures/figure%.py, output/figure%.svg, $(flist))
benchOutput = output/figureS2.svg output/figureS9a_d.svg
regularOutput = $(filter-out $(benchOutput), $(allOutput))

all: $(allOutput)

$(regularOutput): output/figure%.svg: analysis/figures/figure%.py
	@ mkdir -p ./output
	uv run fbuild $*

$(benchOutput): output/figure%.svg: analysis/figures/figure%.py
	@ mkdir -p ./output
	uv run --group benchmarking fbuild $*

test: .venv
	uv run pytest -s -v

.venv: pyproject.toml
	uv sync

coverage.xml: .venv
	uv run pytest --junitxml=junit.xml --cov=RISE --cov-report xml:coverage.xml

lint: .venv
	uv run ruff format .
	uv run ruff check . --fix
	uv run ty check RISE analysis

clean:
	rm -rf output profile profile.svg factor_cache
