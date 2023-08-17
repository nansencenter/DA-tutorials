#!/usr/bin/env python
"""Test that all notebooks run without error.

These tests are not for use with pytest (does not use asserts, orchestrates itself).
Simply run the script as any regular Python script.
Why: Mainly because it did not seem necessary. Also I find debugging with pytest somewhat hard.
"""

from pathlib import Path
import subprocess
import sys


## Convert: notebooks/T*.ipynb --> tests/T*.py
text = dict(capture_output=True, text=True)
THIS = Path(__file__)
ROOT = THIS.parents[1]

converted = []
Ts = sorted((ROOT / "notebooks").glob("T*.ipynb"))
for f in Ts:
    script = (THIS.parent / f.name).with_suffix('.py')
    converted.append(script)
    cmd = ["jupytext", "--output", str(script), str(f)]
    print(subprocess.run(cmd, **text, check=True).stdout)


## Modify scripts. Asserts
def assert_show_answer_in_comments(lines, fname):
    for i, ln in enumerate(lines):
        if "show_answer" in ln:
            assert ln.startswith("# "), f"`show_answer` uncommented in '{fname}':{i}"

def uncomment_show_answer(lines):
    for i, ln in enumerate(lines):
        OLD = "# ws.show_answer"
        NEW = "ws.show_answer"
        if ln.startswith(OLD):
            lines[i] = ln.replace(OLD, NEW)
    return lines

def insert_sys_path(lines):
    return ['import sys',
            f"""sys.path.insert(0, '{ROOT / "notebooks"}')""",
            ] + lines

for script in converted:
    lines = script.read_text().splitlines()

    assert_show_answer_in_comments(lines, script)
    lines = uncomment_show_answer(lines)
    lines = insert_sys_path(lines)

    script.write_text("\n".join(lines))


## Run scripts
erred = []
for script in converted:
    print(script.name, end="\n" + "="*len(script.name) + "\n")
    run = subprocess.run(["python", str(script)], **text, check=False)
    print(run.stdout)
    if run.returncode:
        erred.append(script)
        print(run.stderr, file=sys.stderr)

# Provide return code
if erred:
    print("========================================")
    print("Found issues with:\n" + "\n- ".join([str(f) for f in erred]), file=sys.stderr)
    print("See above for individual tracebacks.")
    sys.exit(1)
