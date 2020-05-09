#!/usr/bin/env bash

# Colab doesn't provide functionality for working with full packages/repos, including
# - defining python environments (e.g. requirements.txt)
# - pre-loading data/scripts other than what's in the notebook.
# We therefore make this script for bootstrapping the notebooks by cloning the full repo.
# It should be run in a Colab notebook using:
# URL="https://raw.githubusercontent.com/nansencenter/DA-tutorials/master/notebooks/resources/colab_bootstrap.sh"
#!wget -qO- $URL | bash -s -- --debug


setup () {
    set -e
    URL=https://github.com/nansencenter/DA-tutorials.git
    if [[ ! -d REPO ]]; then git clone --depth=1 $URL REPO; fi
    pip install -r REPO/requirements.txt
    cp -r REPO/notebooks/resources ./
    cp REPO/notebooks/dpr_config.ini ./
}

# Only run if we're on colab
if python -c "import colab"; then
    # Quiet execution
    if echo $@ | grep -- '--debug' > /dev/null ; then
        setup
    else
        setup > /dev/null 2>&1
    fi
    echo "Initialization for Colab done."
fi
