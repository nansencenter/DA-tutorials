#!/usr/bin/env python3


from pathlib import Path
import re
import sys

import nbformat

macros=r'''$
% START OF MACRO DEF
% DO NOT EDIT IN INDIVIDUAL NOTEBOOKS, BUT IN '''+__file__+r'''
%
\newcommand{\Reals}{\mathbb{R}}
\newcommand{\Expect}[0]{\mathbb{E}}
\newcommand{\NormDist}{\mathcal{N}}
%
\newcommand{\mat}[1]{{\mathbf{{#1}}}} % ALWAYS
%\newcommand{\mat}[1]{{\pmb{\mathsf{#1}}}}
\newcommand{\bvec}[1]{{\mathbf{#1}}} % ALWAYS
%
\newcommand{\trsign}{{\mathsf{T}}} % ALWAYS
\newcommand{\tr}{^{\trsign}} % ALWAYS
\newcommand{\tn}[1]{#1} % ALWAYS
%
\newcommand{\I}[0]{\mat{I}}
\newcommand{\K}[0]{\mat{K}}
\newcommand{\bP}[0]{\mat{P}}
\newcommand{\bH}[0]{\mat{H}}
\newcommand{\bF}[0]{\mat{F}}
\newcommand{\R}[0]{\mat{R}}
\newcommand{\Q}[0]{\mat{Q}}
\newcommand{\B}[0]{\mat{B}}
\newcommand{\Ri}[0]{\R^{-1}}
\newcommand{\Bi}[0]{\B^{-1}}
\newcommand{\X}[0]{\mat{X}}
\newcommand{\A}[0]{\mat{A}}
\newcommand{\Y}[0]{\mat{Y}}
\newcommand{\E}[0]{\mat{E}}
\newcommand{\U}[0]{\mat{U}}
\newcommand{\V}[0]{\mat{V}}
%
\newcommand{\x}[0]{\bvec{x}}
\newcommand{\y}[0]{\bvec{y}}
\newcommand{\q}[0]{\bvec{q}}
\newcommand{\br}[0]{\bvec{r}}
\newcommand{\bb}[0]{\bvec{b}}
%
\newcommand{\bx}[0]{\bvec{\bar{x}}}
\newcommand{\barB}[0]{\mat{\bar{B}}}
\newcommand{\barP}[0]{\mat{\bar{P}}}
\newcommand{\barK}[0]{\mat{\bar{K}}}
%
\newcommand{\D}[0]{\mat{D}}
\newcommand{\Dobs}[0]{\mat{D}_{\text{obs}}}
\newcommand{\Dmod}[0]{\mat{D}_{\text{obs}}}
%
\newcommand{\ones}[0]{\bvec{1}}
\newcommand{\AN}[0]{\big( \I_N - \ones \ones\tr / N \big)}
%
% END OF MACRO DEF
$'''

_macros = macros.split("\n")

# Convert to {macro_name: macro_lineno}
declaration = re.compile(r'''^\\newcommand{(.+?)}''')
lineno_by_name = {}
for i, ln in enumerate(_macros):
    match = declaration.match(ln)
    if match: lineno_by_name[match.group(1)] = i

# Regex for macro, for ex. \mat, including \mat_, but not \mathbf:
no_escape = lambda s: s.replace("\\",r"\\")
delimit = lambda m: re.compile( no_escape(m) + r'(_|\b)' )


def include_macros(content):
    """Include those macros that are used in content (used for answers.)"""
    # Find macros present in content
    ii = [i for macro, i in lineno_by_name.items() if delimit(macro).search(content)]
    # Include in content
    if ii:
        mm = [_macros[i] for i in ii]
        # PRE-pend those that should always be there
        mm = [m for m in _macros if ("ALWAYS" in m) and (m not in mm)] + mm
        # Escape underscore coz md2html sometimes interprets it as <em>.
        # mm = [m.replace("_","\\_") for m in mm]
        # Include surrounding dollar signs
        mm = _macros[:1] + mm + _macros[-1:]
        # Insert space if needed
        space = " " if content.startswith("$") else ""
        # Collect
        content = "\n".join(mm) + space + content
    return content


def broadcast_macros():
    """Insert macros in 1st markdown cell of all notebooks."""

    def find_notebooks():
        ff = [str(f) for f in Path().glob("notebooks/T*.ipynb")]
        ff = [f for f in ff if "_checkpoint" not in f]
        assert len(ff), "Must have notebooks dir in PWD."
        return ff

    HEADER = _macros[1]
    FOOTER = _macros[-2]

    def update(nb):
        for cell in nb["cells"]:
            if cell["cell_type"] == "markdown":
                lines = cell["source"].split("\n")

                # Find line indices of macros section.
                # +/-1 is used to include surrounding dollar signs.
                try:
                    L1 = lines.index(HEADER)-1
                    L2 = lines.index(FOOTER)+1
                    assert lines[L1]=="$"
                    assert lines[L2]=="$"
                except (ValueError, AssertionError) as e:
                    return

                if lines[L1:L2+1] != _macros:
                    lines = lines[:L1] + _macros + lines[L2+1:]
                    cell["source"] = "\n".join(lines)
                    return True # indicate that changes were made

    for f in sorted(find_notebooks()):

        try:
            nb = nbformat.read(f, as_version=4)

            if update(nb):
                print("Updating", f)
                nbformat.write(nb, f)

        except nbformat.reader.NotJSONError as e:
            print("Could not read file", f)


if __name__ == "__main__" and any("update" in arg for arg in sys.argv):
    broadcast_macros()
