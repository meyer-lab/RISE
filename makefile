.PHONY: clean test ty

flist = $(wildcard analysis/figures/figure*.py)
allOutput = $(patsubst analysis/figures/figure%.py, output/figure%.svg, $(flist))

all: $(allOutput)

output/figure%.svg: analysis/figures/figure%.py
	@ mkdir -p ./output
	uv run fbuild $*

test: .venv
	uv run pytest -s -v

.venv: pyproject.toml
	uv sync

coverage.xml: .venv
	uv run pytest --junitxml=junit.xml --cov=RISE --cov-report xml:coverage.xml

ty: .venv
	uv run ty check RISE analysis

clean:
	rm -rf output profile profile.svg factor_cache
