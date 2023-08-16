#!/usr/bin/env bash

# Colab doesn't provide
# - Auto-installing requirements.txt
# - Pre-loading data/modules (aside from the notebook itself)
# This script takes care of the above by cloning the full (shallow) repo.

# Install requirements
main () {
    set -e

    # Clear any existing REPO for a fresh git clone
    rm -rf REPO

    # Download repo
    URL=https://github.com/nansencenter/DA-tutorials.git
    if [[ ! -d REPO ]]; then git clone --depth=1 $URL REPO; fi

    # https://pythonspeed.com/articles/upgrade-pip/
    pip install --upgrade pip

    # Install requirements
    pip install -r REPO/requirements.txt

    # Put repo contents in PWD
    cp -r REPO/notebooks/resources ./
    cp REPO/notebooks/dpr_config.ini ./
}

# Only run if we're on colab
if python -c "import google.colab" 2>/dev/null; then

    # Use `bash -s -- --debug` to get verbose output
    if echo $@ | grep -E -- '(--debug|-v)' > /dev/null ; then
        main
    else
        # Quiet
        main > /dev/null 2>&1
    fi

    echo "Initialization for Colab done."
else
    echo "Not running on Colab => Didn't do anything."
fi
