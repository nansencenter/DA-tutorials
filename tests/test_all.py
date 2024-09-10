#!/usr/bin/env python
"""Test that all notebooks run without error.

Also test stuff relating to `show_answer`.

These tests are not for use with pytest (does not use asserts, orchestrates itself).
Simply run the script as any regular Python script.
Why: Mainly because it did not seem necessary. Also I find debugging with pytest somewhat hard.
"""

from pathlib import Path
import subprocess
import sys
import requests
from urllib.parse import unquote

from markdown import markdown as md2html


UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
ROOT = Path(__file__).parents[1]


def _report_error(msg):
    # raise AssertionError(msg)  # for post-portem debugging
    print(msg)
    return True


def _find_anchor(fname, anchor):
    lines = fname.read_text().splitlines()
    headings = [x for x in lines if x.startswith("# #")]  # filter for "### Example heading"
    headings = [x[2:].lstrip("#").lstrip() for x in headings]
    headings = [x.replace(" ", "-") for x in headings]
    return anchor in headings


def assert_all_links_work(lines, fname):
    failed = False
    for i, line in enumerate(lines):

        # Skip
        if not line.startswith("#"):
            continue
        if any(x in line for x in [
                "www.google.com/search",  # because md2html fails to parse
                "www.example.com"]):
            continue

        # First do a *quick* scan for links.
        if "](" in line or "http" in line:
            # Extract link
            html = md2html(line) # since extracting url from md w/ regex is hard
            # PS: linebreaks in links ⇒ failure (as desired)
            link = html.split('href="')[1].split('">')[0]
            # fix parsing error for links ending in ')'
            if "))" in link:
                link = link.split("))")[0] + ")"

            # Common error message
            def errm(issue):
                return f"Issue on line {i} with {issue} link\n    {link}"

            # Internet links
            if "http" in link:
                try:
                    response = requests.head(link, headers={'User-Agent': UA})
                    # https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
                    assert response.status_code < 400
                except Exception:
                    # Stackoverflow does not like GitHub CI IPs?
                    # https://meta.stackexchange.com/questions/443
                    skip = "stack" in link and response.status_code == 403
                    if not skip:
                        failed |= True
                        _report_error(errm("**requesting**") +
                            f"\nStatus code: {response.status_code}")

            # Local links
            else:
                link = unquote(link)
                link_fname, *link_anchor = link.split("#")

                # Validate filename
                if link_fname:
                    if not (ROOT / "notebooks" / link_fname).is_file():
                        failed |= _report_error(errm("**filename** of"))

                # Validate anchor
                if link_anchor:
                    if not link_fname:
                        # Anchor only ⇒ same file
                        link_fname = fname
                    else:
                        # Change "T4...ipynb" --> "tests/T4...py"
                        link_fname = (ROOT / "tests" / link_fname).with_suffix(".py")

                    if not _find_anchor(link_fname, link_anchor[0]):
                        failed |= _report_error(errm("**anchor tag** of"))
    return failed


def assert_show_answer(lines, _fname):
    """Misc checks on `show_answer`"""
    failed = False
    found_import = False
    for i, line in enumerate(lines):
        found_import |= ("show_answer" in line and "import" in line)
        if line.lstrip().startswith("show_answer"):
                print(f"`show_answer` uncommented on line {i}")
                failed |= True
    if not found_import:
        print("`import show_answer` not found.")
        failed = True
    return failed


def uncomment_show_answer(lines):
    """Causes checking existance of answer when script gets run."""
    for i, line in enumerate(lines):
        OLD = "# show_answer"
        NEW = "show_answer"
        if line.startswith(OLD):
            lines[i] = line.replace(OLD, NEW)
    return lines


def make_script_runnable_by_fixing_sys_path(lines):
    """Makes it seem like CWD is `notebooks`."""
    return ['import sys',
            f"""sys.path.insert(0, '{ROOT / "notebooks"}')""",
            ] + lines


## Convert: notebooks/T*.ipynb --> tests/T*.py
print("\nConverting from notebooks/...ipynb to tests/...py")
print("========================================")
text = dict(capture_output=True, text=True)
converted = []
ipynbs = sorted((ROOT / "notebooks").glob("T*.ipynb"))
for f in ipynbs:
    script = (ROOT / "tests" / f.name).with_suffix('.py')
    converted.append(script)
    cmd = ["jupytext", "--output", str(script), str(f)]
    print(subprocess.run(cmd, **text, check=True).stdout)


## Static checks. Also: modify scripts
erred = []
for script in converted:
    print("\nStatic analysis for", script.stem)
    print("========================================")
    lines = script.read_text().splitlines()
    failed = False

    # Validatation checks
    failed |= assert_all_links_work(lines, script)
    failed |= assert_show_answer(lines, script)

    # Modify script in preparation of running it
    lines = uncomment_show_answer(lines)
    lines = make_script_runnable_by_fixing_sys_path(lines)

    if failed:
        erred.append(script)
    script.write_text("\n".join(lines))


print("\nStatic analysis for", "answers.py")
print("========================================")
sys.path.insert(0, f"{ROOT / 'notebooks'}")
import resources.answers  # type: ignore # noqa
for key, answer in resources.answers.answers.items():
    lines = ["# " + line for line in answer[1].splitlines()]
    fname = Path(resources.answers.__file__ + ":" + key)
    if assert_all_links_work(lines, fname):
        erred.append(fname)


## Run ipynbs as python scripts
for script in converted:
    print("\nRunning", script.name)
    print("========================================")
    run = subprocess.run(["python", str(script)], **text, check=False)
    # print(run.stdout)
    if run.returncode:
        erred.append(script)
        print(run.stderr, file=sys.stderr)

# Provide return code
if erred:
    print("========================================")
    print("FOUND ISSUES")
    print("========================================")
    print(*["- " + str(f) for f in erred], file=sys.stderr)
    print("See above for individual tracebacks.")
    sys.exit(1)
