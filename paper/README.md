# RAVEL paper

`main.tex` is the modular paper entry point. The latest source was imported
from the supplied `USENIX_2023.zip` and reorganized without changing its
section content. `usenix.tex` is retained as a compilable monolithic version.

## Layout

- `preamble.tex`: packages and document-wide settings
- `sections/00_abstract.tex` through `sections/07_conclusion.tex`: main paper
- `sections/08_appendix_design.tex`: detailed operational semantics
- `sections/09_appendix_mechanism_ablation.tex`: mechanism-ablation analysis
- `sections/10_appendix_robustness.tex`: repeated-run robustness
- `figures/`: all image assets referenced by the paper
- `sample.bib`: bibliography database

## Build

From the `paper` directory, run `make`. The generated PDF is
`paper/build/main.pdf`.
