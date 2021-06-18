#!/usr/bin/env bash

# Colab doesn't provide functionality for
# working with full packages/repos, including
# - defining python environments (e.g. requirements.txt)
# - pre-loading data/scripts other than what's in the notebook.
# We therefore make this script for bootstrapping the notebooks
# by cloning the full repo.

#############
#  WARNING  #
#############
# This will ALWAYS give you the current master version of the repo,
# something which can be a source of confusion, e.g.:
# - When bug hunting by checking out previous commits.
# - When working on Colab branch, and changing something other than
#   the notebook or script. This change won't be used on Colab
#   until it's merged into master. Of course, we could point the
#   below download to the Colab branch, but that won't fix the previous item,
#   and also assumes that we make the choice of maintaining a separate branch
#   for Colab (instead of using the check `import google.colab`, or telling
#   user to insert the appropriate code when running on colab).


# Install requirements
main () {
    set -e

    # Clear any existing REPO for a fresh git clone
    rm -rf REPO

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

    # Use `bash -s -- --debug` to get verbose output
    if echo $@ | grep -E -- '(--debug|-v)' > /dev/null ; then
        main
    else
        # Quiet
        main > /dev/null 2>&1
    fi

    echo "Initialization for Colab done."
fi
