# White Paper

This folder contains the LaTeX source for the traffic forecasting white paper.

Source files:

- `traffic_forecasting.tex`
- `references.bib`

Build outputs such as `.aux`, `.bbl`, `.blg`, `.fdb_latexmk`, `.fls`, `.out`, and `.synctex.gz` are ignored.

To compile locally:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error traffic_forecasting.tex
```

If `latexmk` is not available, install a TeX distribution such as TeX Live or MiKTeX.

The paper still needs to be synchronized with the latest code state, especially the seed-time sampler, shape-loss experiments, real-time dense evaluation, and model traceability limitations.
