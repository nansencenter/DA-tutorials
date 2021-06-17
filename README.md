# Introduction to data assimilation and the EnKF

Jump right in using one of these cloud computing providers:

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/nansencenter/DA-tutorials/blob/Colab)
  (requires Google login)
- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nansencenter/DA-tutorials/master)
  (no login but can be slow to start)

### Overview

- Interactive (Jupyter notebook)
- Contains theory, code (Python), and exercises.
- Recommendation: work in pairs.
- Each tutorial takes ≈75 min.
- The tutor will circulate to assist with the exercises,  
  and summarize each section after you have worked on them.

### Instructions for working locally

You can also run these notebooks on your own (Linux/Windows/Mac) computer.
This is a bit snappier than running them online.

1. **Prerequisite**: Python 3.7.  
   If you're an expert, setup a python environment however you like.
   Otherwise:
   Install [Anaconda](https://www.anaconda.com/download), then
   open the [Anaconda terminal](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#starting-conda)
   and run the following commands:

   ```bash
   conda create --yes --name my-env python=3.7
   conda activate my-env
   python -c 'import sys; print("Version:", sys.version.split()[0])'
   ```

   Ensure the output at the end gives version 3.7.  
   *Keep using the same terminal for the commands below.*

2. **Install**:  
   Run these commands in the terminal:

   ```sh
   git clone https://github.com/nansencenter/DA-tutorials.git
   pip install -r DA-tutorials/requirements.txt
   ```

3. **Launch the Jupyter notebooks**:  
   Run `jupyter-notebook`.  
   This will open up a page in your web browser that is a file navigator.  
   Enter the folder `DA-tutorials/notebooks`, and click on a tutorial (`T1... .ipynb`).

<!-- markdownlint-disable-file heading-increment -->
