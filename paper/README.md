# RAVEL paper

`main.tex` is the paper entry point. Each top-level paper section lives in one
file under `sections/`; publication-ready visual assets live directly under
`figures/`, and standalone table sources belong under `tables/`.

## Build

From the repository root, use the repository's default Conda environment:

```bash
conda run -n rwluenv make -C paper
```

The generated paper and all LaTeX intermediates are written to
`paper/build/`. The final PDF is `paper/build/main.pdf`.

To remove generated files:

```bash
conda run -n rwluenv make -C paper clean
```

## Layout

- `main.tex`: title, section order, and bibliography wiring
- `preamble.tex`: packages and document-wide layout settings
- `macros.tex`: RAVEL-specific commands and algorithm styling
- `sections/`: one `.tex` file per top-level section
- `figures/`: flat directory of paper figures
- `tables/`: standalone table sources
- `refer.bib`: bibliography database
