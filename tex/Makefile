PYTHON ?= python3
TECTONIC ?= tectonic

.PHONY: all figures check review
all: main.pdf review.pdf

figures:
	$(PYTHON) ../source/analyze.py

main.pdf: main.tex references.bib generated/numbers.tex generated/throughput.tex generated/delay.tex generated/timing.tex generated/scenarios.tex generated/gains.pdf generated/latency.pdf
	@if command -v latexmk >/dev/null 2>&1; then latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex; else $(TECTONIC) --keep-logs main.tex; fi

review.pdf: review.tex main.tex references.bib main.pdf
	@if command -v latexmk >/dev/null 2>&1; then latexmk -pdf -interaction=nonstopmode -halt-on-error review.tex; else $(TECTONIC) --keep-logs review.tex; fi

check: all
	$(PYTHON) ../source/check_paper.py
