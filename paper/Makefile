PAPER := main
BUILD_DIR := build
LATEX := pdflatex
BIBTEX := bibtex
LATEX_FLAGS := -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=$(BUILD_DIR)

TEX_SOURCES := main.tex preamble.tex macros.tex $(wildcard sections/*.tex)
FIGURES := $(wildcard figures/*)

.PHONY: all clean

all: $(BUILD_DIR)/$(PAPER).pdf

$(BUILD_DIR)/$(PAPER).pdf: $(TEX_SOURCES) $(FIGURES) refer.bib usenix.sty acm.bst | $(BUILD_DIR)
	$(LATEX) $(LATEX_FLAGS) $(PAPER).tex
	cd $(BUILD_DIR) && BIBINPUTS=..: BSTINPUTS=..: $(BIBTEX) $(PAPER)
	$(LATEX) $(LATEX_FLAGS) $(PAPER).tex
	$(LATEX) $(LATEX_FLAGS) $(PAPER).tex

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf -- "$(BUILD_DIR)"
