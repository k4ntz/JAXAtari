# Report

`report.tex` plus `figures/`. Nothing else is needed to build it.
`report.pdf` is the compiled output, 14 pages, committed so it can be read
without a LaTeX toolchain.

## Building it

**Locally, no sudo.** [Tectonic](https://tectonic-typesetting.github.io/) is a
single binary that fetches the packages it needs on first run:

```bash
brew install tectonic && tectonic -X compile report.tex
```

This is how `report.pdf` was produced. It compiles clean, no warnings.

**Overleaf, no install.** Create a new project, upload `report.tex` and the whole
`figures/` folder, press Recompile.

**A full TeX Live.** `brew install --cask basictex`, open a new shell, then
`pdflatex report.tex` twice, so that the table and figure references resolve.

## Related documents

The group report for the lab, which covers both halves of Topic 30 in the short
structured format the group agreed on, lives outside this repository. This
document is the technical companion to it: same results, full derivations and
justifications.
