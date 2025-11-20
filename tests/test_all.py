#!/usr/bin/env python
"""Test that all notebooks run without error.

Also test stuff relating to `show_answer`.

These tests are not for use with pytest (does not use asserts, orchestrates itself).
Simply run the script as any regular Python script.
Why: Mainly because it did not seem necessary. Also I find debugging with pytest somewhat hard.
"""

from pathlib import Path
import os
import subprocess
import sys
import re
import requests
from urllib.parse import unquote

from markdown import markdown as md2html


UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
ROOT = Path(__file__).parents[1]


def _find_headings(fname: Path):
    """Find headings"""
    lines = fname.read_text().splitlines()
    # filter for "# ### Example heading" or "# - ### Heading in bullet point"
    headings = [x for x in lines if x.startswith("# #") or x.startswith("# - #")]
    headings = [x.lstrip("# -") for x in headings]
    headings = [x.replace(" ", "-") for x in headings]
    return headings

def _find_anchors(fname: Path):
    """Find anchors"""
    lines = fname.read_text().splitlines()
    # filter for "<a name="example-heading"></a>"
    anchors = [x for x in lines if x.lstrip().startswith("# <a name=")]
    anchors = [x.split('name=')[1][1:].split('>')[0][:-1] for x in anchors]
    return anchors

cache_headings = {}
cache_anchors = {}
def assert_all_links_work(lines, fname):
    any_failed = False
    for i, line in enumerate(lines):
        # Skip misc
        if not line.startswith("#"):
            continue
        if "url={" in line.lower():
            # Dont check bibtex URLs
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
            def print_err(msg):
                # raise AssertionError(msg)  # for post-portem debugging
                print(f"Issue on line {i} for link\n    {link}\n    {msg}")

            # Internet links
            if "http" in link:
                if any(x in line for x in [
                        "www.google.com/search",  # because md2html fails to parse
                        "www.example.com"]):
                    continue
                response = None
                try:
                    response = requests.head(link, headers={'User-Agent': UA}, allow_redirects=True, timeout=10)
                    if response.status_code in (403, 405):
                        # Fallback to GET if HEAD is not allowed or forbidden
                        response = requests.get(link, headers={'User-Agent': UA}, allow_redirects=True, timeout=10)
                    # Ignore status code 429 (Too Many Requests)
                    if response.status_code == 429:
                        continue
                    assert response.status_code < 400
                except Exception as e:
                    # Known problematic domains
                    skip_domains = ["stack", "wiley.com", "springer.com", "elsevier.com"]
                    status = response.status_code if response is not None else "N/A"
                    skip = os.getenv("GITHUB_ACTIONS") and any(domain in link for domain in skip_domains) or status == 429
                    if not skip:
                        any_failed |= True
                        print_err(f"Status code: {status} when **requesting**. Error: {e}")

            # Local links
            else:
                link = unquote(link)
                link_fname, *link_anchor = link.split("#")

                # Validate filename
                if link_fname:
                    if not (ROOT / "notebooks" / link_fname).is_file():
                        any_failed |= True
                        print_err("Filename not found.")

                # Validate anchor
                if link_anchor:
                    # Find headings in `link_fname`
                    if "answers.py" not in str(fname):
                        if not link_fname:
                            # Anchor only ⇒ same file
                            link_fname = fname
                        else:
                            # Change "T4...ipynb" --> "tests/T4...py"
                            link_fname = (ROOT / "tests" / link_fname).with_suffix(".py")
                        # With caching
                        if link_fname not in cache_anchors:
                            cache_headings[link_fname] = _find_headings(link_fname)
                            cache_anchors[link_fname] = _find_anchors(link_fname)
                        anchors = cache_anchors[link_fname]
                        headings = cache_headings[link_fname]
                    else:
                        # For answers.py, use union of anchors in all notebooks
                        anchors = [h for hh in cache_anchors.values() for h in hh]
                        headings = [h for hh in cache_headings.values() for h in hh]

                    # Gheck if anchor present in anchors
                    if link_anchor[0] not in anchors:
                        any_failed |= True
                        if link_anchor[0] in headings:
                            print_err("Anchor (necessary on Colab) missing or incorrect")
                        else:
                            closest = []
                            for h in anchors:
                                for word in re.split(r'[-–:_]', link_anchor[0]):
                                    if len(word) > 2 and (word.lower() not in ["exc", "optional"]) and word.lower() in h.lower():
                                        closest.append(h)
                                        break
                            closest = "\n      * ".join(["Change to one of the following?"] + closest)
                            print_err("Anchor tag not found. " + closest)
    return any_failed


def assert_show_answer(lines, _fname):
    """Misc checks on `show_answer`"""
    any_failed = False
    found_import = False
    for i, line in enumerate(lines):
        found_import |= ("show_answer" in line and "import" in line)
        if line.lstrip().startswith("show_answer"):
                print(f"`show_answer` uncommented on line {i}")
                any_failed |= True
    if not found_import:
        print("`import show_answer` not found.")
        any_failed = True
    return any_failed


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
    # script = (ROOT / "notebooks" / "nb_mirrors" / f.name).with_suffix('.py')
    converted.append(script)
    cmd = ["jupytext", "--output", str(script), str(f)]
    print(subprocess.run(cmd, **text, check=True).stdout)


## Static checks. Also: modify scripts
erred = []
for script in converted:
    print("\nStatic analysis for", script.stem)
    print("========================================")
    lines = script.read_text().splitlines()
    any_failed = False

    # Validatation checks
    any_failed |= assert_all_links_work(lines, script)
    any_failed |= assert_show_answer(lines, script)

    # Modify script in preparation of running it
    lines = uncomment_show_answer(lines)
    lines = make_script_runnable_by_fixing_sys_path(lines)

    if any_failed:
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


# ## Run ipynbs as python scripts
# for script in converted:
#     print("\nRunning", script.name)
#     print("========================================")
#     run = subprocess.run(["python", str(script)], **text, check=False)
#     # print(run.stdout)
#     if run.returncode:
#         erred.append(script)
#         print(run.stderr, file=sys.stderr)

# Provide return code
if erred:
    print("========================================")
    print("FOUND ISSUES IN")
    print("========================================")
    print(*["- " + str(f) for f in erred], file=sys.stderr, sep="\n")
    print("See above for individual tracebacks.")
    sys.exit(1)
