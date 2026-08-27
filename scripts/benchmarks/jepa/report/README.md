# Report

`report.tex` plus `figures/`. Nothing else is needed to build it.

## Building it

There is no LaTeX toolchain on the machine this was written on, so the document
has been checked structurally (environments, braces, table columns, figure paths)
but not compiled. Two ways to build it:

**Overleaf, no install.** Create a new project, upload `report.tex` and the whole
`figures/` folder, press Recompile. This is the quickest route.

**Locally.** `brew install --cask basictex`, open a new shell, then
`pdflatex report.tex` twice. Two passes so that the table and figure references
resolve.

## Before submitting

One table is deliberately incomplete. Table `tab:published` compares against
reported PPO scores from the original ALE implementation, which the lab guidelines
ask for. The published column is marked `\todofill{}` and prints in red. Fill it
from the source linked in the guidelines rather than from memory, then delete the
`\todofill` macro definition at the top of the file.
