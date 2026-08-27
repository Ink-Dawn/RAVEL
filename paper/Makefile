PAPER := main
BUILD_DIR := build
LATEX_FLAGS := -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=$(BUILD_DIR)

.PHONY: all clean

all: $(BUILD_DIR)/$(PAPER).pdf

$(BUILD_DIR)/$(PAPER).pdf: main.tex preamble.tex sections/*.tex figures/* sample.bib usenix.sty acm.bst | $(BUILD_DIR)
	pdflatex $(LATEX_FLAGS) $(PAPER).tex
	cd $(BUILD_DIR) && bibtex $(PAPER)
	pdflatex $(LATEX_FLAGS) $(PAPER).tex
	pdflatex $(LATEX_FLAGS) $(PAPER).tex

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf -- "$(BUILD_DIR)"
