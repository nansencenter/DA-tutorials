
import re
import contextlib


@contextlib.contextmanager
def rewrite(fname):
    """File-editor contextmanager.

    Example:

    >>> with rewrite("myfile.txt") as lines:
    >>>     for i, line in enumerate(lines):
    >>>         lines[i] = line.replace("old","new")
    """
    with open(fname, 'r') as f:
        lines = [line for line in f]

    yield lines

    with open(fname, 'w') as f:
        f.write("".join(lines))

def list_used_newcommands(lines):
    """
    Lists the \newcommand macros actually in use in a given Markdown file.
    """
    defined_macros = {}
    used_macros = [r'\mat', r'\bvec']

    # Regex to find \newcommand{\macro_name}
    newcommand_pattern = re.compile(r'^\\newcommand\{(\\[A-Za-z0-9]+)\}')

    # First pass: find all defined \newcommand macros
    for i, line in enumerate(lines):
        if "\\newcommand" in line: # Optimize to only check lines that likely contain newcommands
            matches = newcommand_pattern.findall(line)
            for macro in matches:
                defined_macros[macro] = i

    macro_by_line = {v: k for k, v in defined_macros.items()}

    # Second pass: find where these defined macros are used in the rest of the document
    for i, line in enumerate(lines):
        for macro in defined_macros:
            if macro_by_line.get(i, None) == macro:
                continue  # Skip if this is defining line
            if macro in line: # for speed
                if re.search(re.escape(macro)+r'\b', line) or re.search(re.escape(macro)+'_', line):
                    if macro not in used_macros:
                        used_macros.append(macro)

    # Print
    print("Used \\newcommand macros:")
    for macro in used_macros:
        print(macro)

    if REPLACE:
        for macro, i in defined_macros.items():
            if macro not in used_macros:
                print(f"Removing unused macro: {macro} at line {i+1}")
                lines[i] = ""

    # Enable recursion
    return lines

# Specify the path to the Markdown file
# file_path = "notebooks/scripts/T1 - DA & EnKF.md"
# file_path = "notebooks/scripts/T2 - Gaussian distribution.md" 
# file_path = "notebooks/scripts/T3 - Bayesian inference.md" 
# file_path = "notebooks/scripts/T4 - Time series filtering.md" 
# file_path = "notebooks/scripts/T5 - Multivariate Kalman filter.md" 
# file_path = "notebooks/scripts/T6 - Geostats & Kriging (optional).md" 
# file_path = "notebooks/scripts/T7 - Chaos & Lorenz (optional).md" 
# file_path = "notebooks/scripts/T8 - Monte-Carlo & ensembles.md" 
file_path = "notebooks/scripts/T9 - Writing your own EnKF.md" 

REPLACE = True

with rewrite(file_path) as lines:
    # Call multiple times to eliminate macros used by other macros
    list_used_newcommands(list_used_newcommands(list_used_newcommands(lines)))
