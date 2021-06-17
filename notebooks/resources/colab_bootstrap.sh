#!/usr/bin/env bash

# Colab doesn't provide functionality for
# working with full packages/repos, including
# - defining python environments (e.g. requirements.txt)
# - pre-loading data/scripts other than what's in the notebook.
# We therefore make this script for bootstrapping the notebooks
# by cloning the full repo.


# Install requirements
main () {
    set -e

    # Clear cache
    rm -rf /root/.cache

    # Download repo
    URL=https://github.com/nansencenter/DA-tutorials.git
    if [[ ! -d REPO ]]; then git clone --depth=1 $URL REPO; fi

    # Install requirements
    pip install -r REPO/requirements.txt

    # Put repo contents in PWD
    cp -r REPO/notebooks/resources ./
    cp REPO/notebooks/dpr_config.ini ./
}

# Only run if we're on colab
if python -c "import colab"; then
    if echo $@ | grep -- '--debug' > /dev/null ; then
        # Verbose (regular output)
        main
    else
        # Quiet execution
        main > /dev/null 2>&1
    fi
    echo "Initialization for Colab done."
fi
