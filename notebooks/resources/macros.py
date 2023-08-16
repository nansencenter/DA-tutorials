#!/usr/bin/env python3

# Macros must be included in
# - every notebook.
# - every answer that uses them.


from pathlib import Path
import re
import sys

import nbformat


HEADER = r'''% ######################################## Loading TeX (MathJax)... Please wait ########################################'''
macros=r'''
\newcommand{\Reals}{\mathbb{R}}
\newcommand{\Expect}[0]{\mathbb{E}}
\newcommand{\NormDist}{\mathcal{N}}

\newcommand{\DynMod}[0]{\mathscr{M}}
\newcommand{\ObsMod}[0]{\mathscr{H}}

\newcommand{\mat}[1]{{\mathbf{{#1}}}} % ALWAYS
%\newcommand{\mat}[1]{{\pmb{\mathsf{#1}}}}
\newcommand{\bvec}[1]{{\mathbf{#1}}} % ALWAYS

\newcommand{\trsign}{{\mathsf{T}}} % ALWAYS
\newcommand{\tr}{^{\trsign}} % ALWAYS
\newcommand{\ceq}[0]{\mathrel{≔}}
\newcommand{\xDim}[0]{D}
\newcommand{\supa}[0]{^\text{a}}
\newcommand{\supf}[0]{^\text{f}}

\newcommand{\I}[0]{\mat{I}} % ALWAYS
\newcommand{\K}[0]{\mat{K}}
\newcommand{\bP}[0]{\mat{P}}
\newcommand{\bH}[0]{\mat{H}}
\newcommand{\bF}[0]{\mat{F}}
\newcommand{\R}[0]{\mat{R}}
\newcommand{\Q}[0]{\mat{Q}}
\newcommand{\B}[0]{\mat{B}}
\newcommand{\C}[0]{\mat{C}}
\newcommand{\Ri}[0]{\R^{-1}}
\newcommand{\Bi}[0]{\B^{-1}}
\newcommand{\X}[0]{\mat{X}}
\newcommand{\A}[0]{\mat{A}}
\newcommand{\Y}[0]{\mat{Y}}
\newcommand{\E}[0]{\mat{E}}
\newcommand{\U}[0]{\mat{U}}
\newcommand{\V}[0]{\mat{V}}

\newcommand{\x}[0]{\bvec{x}}
\newcommand{\y}[0]{\bvec{y}}
\newcommand{\z}[0]{\bvec{z}}
\newcommand{\q}[0]{\bvec{q}}
\newcommand{\br}[0]{\bvec{r}}
\newcommand{\bb}[0]{\bvec{b}}

\newcommand{\bx}[0]{\bvec{\bar{x}}}
\newcommand{\by}[0]{\bvec{\bar{y}}}
\newcommand{\barB}[0]{\mat{\bar{B}}}
\newcommand{\barP}[0]{\mat{\bar{P}}}
\newcommand{\barC}[0]{\mat{\bar{C}}}
\newcommand{\barK}[0]{\mat{\bar{K}}}

\newcommand{\D}[0]{\mat{D}}
\newcommand{\Dobs}[0]{\mat{D}_{\text{obs}}}
\newcommand{\Dmod}[0]{\mat{D}_{\text{obs}}}

\newcommand{\ones}[0]{\bvec{1}} % ALWAYS
\newcommand{\AN}[0]{\big( \I_N - \ones \ones\tr / N \big)}
'''
macros = [ln for ln in macros.splitlines() if ln and not ln.startswith('%')]
always = [i for i, ln in enumerate(macros) if "ALWAYS" in ln]
macros = [m.replace("% ALWAYS","").rstrip() for m in macros]

# Convert to {macro_name: macro_lineno}
declaration = re.compile(r'''^\\newcommand{(.+?)}''')
lineno_by_name = {}
for i, ln in enumerate(macros):
    match = declaration.match(ln)
    if match: lineno_by_name[match.group(1)] = i

# Regex for macro, for ex. \mat, including \mat_, but not \mathbf:
no_escape = lambda s: s.replace("\\",r"\\")
delimit = lambda m: re.compile( no_escape(m) + r'(_|\b)' )


def include_macros(content):
    """Include macros in answers. Only those that are required."""
    # Find macros present in content
    necessary = [i for macro, i in lineno_by_name.items() if delimit(macro).search(content)]
    # Include in content
    if necessary:
        mm = [macros[i] for i in necessary]
        # PRE-pend those that should always be there
        mm = [macros[i] for i in always if (macros[i] not in mm)] + mm
        # Escape underscore coz md2html sometimes interprets it as <em>.
        mm = [m.replace("_","\\_") for m in mm]
        # Include surrounding dollar signs
        mm = ["$"] + mm + ["$"]
        # Avoid accidental $$
        space = " " if content.startswith("$") else ""
        # Collect
        content = "\n".join(mm) + space + content
    return content


def update_1nbscript(f: Path):
    """Update the macros of a notebook script (synced with `jupytext`)."""
    print(f.name.ljust(40), end=": ")
    lines = f.read_text().splitlines()
    mLine = "# " + " ".join(macros)

    try:
        iHeader = lines.index("# " + HEADER)
    except (ValueError, AssertionError):
        print("Could not locate pre-existing macros")
        return

    if not (lines[iHeader-1] == "# $" and
            lines[iHeader+2] == "# $"):
        print("Could not parse macros")

    # elif lines[iHeader+1] == mLine:
    #     print("Macros already up to date.")

    else:
        # lines[iHeader] = "# % ##### NEW HEADER ######"
        lines[iHeader+1] = mLine
        f.write_text("\n".join(lines))
        print("Macros updated!")


if __name__ == "__main__" and any("update" in arg for arg in sys.argv):
    for f in sorted((Path(__file__).parents[1] / "scripts").glob("T*.py")):
        update_1nbscript(f)
