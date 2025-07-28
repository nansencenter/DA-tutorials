[Feedback form](https://docs.google.com/forms/d/e/1FAIpQLSdIFxsnPq6i0Swrxujj7AUrFcajw8GDGDbtbByDACh9g50eRA/viewform?usp=header)

# Intro to data assimilation (DA) and the EnKF

An interactive (Jupyter notebook) tutorial.
Jump right in (no installation!) by clicking
the button of one of these cloud computing providers:

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/nansencenter/DA-tutorials)
  (requires Google login)
- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nansencenter/DA-tutorials/master)
  (no login but can be slow to start)

*Prerequisites*: basics of calculus, matrices (e.g. inverses),
random variables, Python (numpy).

![ToC](./intro-toc.png)

### Instructions for working locally

If you prefer, you can also run these notebooks on your own (Linux/Windows/Mac) computer.
This is a bit snappier than running them online.

1. **Prerequisite**: Python 3.9.  
   If you're an expert, setup a python environment however you like.
   Otherwise:
   Install [Anaconda](https://www.anaconda.com/download), then
   open the [Anaconda terminal](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#starting-conda)
   and run the following commands:

   ```bash
   conda create --yes --name my-env python=3.9
   conda activate my-env
   python --version
   ```

   Ensure the printed version is 3.9.  
   *Keep using the same terminal for the commands below.*

2. **Install**:

    - Download and unzip (or `git clone`)
      this repository (see the green button up top)
    - Move the resulting folder wherever you like
    - `cd` into the folder
    - Install requirements:  
      `pip install -r path/to/requirements.txt`

3. **Launch the Jupyter notebooks**:

    - Launch the "notebook server" by executing:  
      `jupyter-notebook`  
      This will open up a page in your web browser that is a file navigator.  
    - Enter the folder `DA-tutorials/notebooks`, and click on a tutorial (`T1... .ipynb`).

<!-- markdownlint-disable-file heading-increment no-inline-html -->


## Developer notes

*Please don't hesitate to submit issues or pull requests!*

[![GitHub CI](https://github.com/nansencenter/DA-tutorials/actions/workflows/tests.yml/badge.svg)](https://github.com/nansencenter/DA-tutorials/actions)

#### Why `scripts/` dir?

- Easier to read git diffs
- Enable importing from notebook (script mirrors)
