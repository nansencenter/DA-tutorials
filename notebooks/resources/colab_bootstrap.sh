#!/usr/bin/env bash

# This script provides functionality for working with full repos on Colab
# - Installing packages/dependencies from requirements.txt
# - Loading data/scripts other than what's in the notebook.

# Warning: This will ALWAYS give you the **'master'** branch of the repo.
# Reasons:
# - Script gets run from a notebook which doesn't know its branch.
# - If script always checks out 'Colab' branch,
#   that might be surprising if on a 'Colab-dev' branch.
# - Maybe in future there won't be a 'Colab' branch (we can use instead an
#   `import google.colab` guard, or telling user to manually run script).


#############
#  WARNING  #
#############
# - When bug hunting by checking out previous commits.
# - When working on Colab branch, and changing something other than
#   the notebook or script. This change won't be used on Colab
#   until it's merged into master. Of course, we could point the
#   below download to the Colab branch, but that won't fix the previous item,


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
if python -c "import google.colab"; then

    # Use `bash -s -- --debug` to get verbose output
    if echo $@ | grep -E -- '(--debug|-v)' > /dev/null ; then
        main
    else
        # Quiet
        main > /dev/null 2>&1
    fi

    echo "Initialization for Colab done."
fi
