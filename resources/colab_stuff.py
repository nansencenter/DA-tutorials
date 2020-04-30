"""Tools to make notebooks work on Google Colab.

"""

from IPython.display import HTML, display

# Notes:
# - md2html rendering sometimes breaks
#   because it has failed to parse the eqn properly.
#   For ex: _ in math sometimes gets replaced by <em>.
#   Can be fixed by escaping, i.e. writing \_

try:
    import google.colab
    is_colab = True
except ImportError:
    is_colab = False

def setup_typeset():
    """MathJax initialization for the current cell.

    This installs and configures MathJax for the current output.

    Necessary in Google Colab. Ref:
    https://github.com/googlecolab/colabtools/issues/322
    """
    if not is_colab: return
    display(HTML('''
            <script src="https://www.gstatic.com/external_hosted/mathjax/latest/MathJax.js?config=TeX-AMS_HTML-full,Safe&delayStartupUntil=configured"></script>
            <script>
                (() => {
                    const mathjax = window.MathJax;
                    mathjax.Hub.Config({
                    'tex2jax': {
                        'inlineMath': [['$', '$'], ['\\(', '\\)']],
                        'displayMath': [['$$', '$$'], ['\\[', '\\]']],
                        'processEscapes': true,
                        'processEnvironments': true,
                        'skipTags': ['script', 'noscript', 'style', 'textarea', 'code'],
                        'displayAlign': 'center',
                    },
                    'HTML-CSS': {
                        'styles': {'.MathJax_Display': {'margin': 0}},
                        'linebreaks': {'automatic': true},
                        // Disable to prevent OTF font loading, which aren't part of our
                        // distribution.
                        'imageFont': null,
                    },
                    'messageStyle': 'none'
                });
                mathjax.Hub.Configured();
            })();
            </script>
            '''))


import subprocess
import sys

# https://stackoverflow.com/a/50255019/38281
def pip_install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
