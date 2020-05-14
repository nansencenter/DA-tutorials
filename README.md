# Introduction to data assimilation and the EnKF

Jump right in by chosing one of these cloud computing providers:

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/nansencenter/DA-tutorials/blob/Colab) (requires Google login)
- [![Azure Notebooks](https://notebooks.azure.com/launch.png)](https://notebooks.azure.com/import/gh/nansencenter/DA-tutorials) (requires MS/Azure login and
[Step1](./notebooks/resources/instruction_images/azure1.png)
[Step2](./notebooks/resources/instruction_images/azure2.png)
[Step3](./notebooks/resources/instruction_images/azure3.png))
- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nansencenter/DA-tutorials/master)
  (no login, but sometimes slow to start)


### Overview
<!--
! 
! Previews notebooks/resources/getting_started/*.svg
! 
-->

<!---![Getting started 1](./notebooks/resources/getting_started/intro1.svg)-->
<!---![Getting started 2](./notebooks/resources/getting_started/intro2.svg)-->
<!---![Getting started 4](./notebooks/resources/getting_started/intro4.svg)-->

* Interactive (Jupyter notebook)
* Contains theory, code (Python), and exercises.
* Recommendation: work in pairs.
* Each tutorial takes ≈75 min.
* The tutor will circulate to assist with the exercises,  
  and summarize each section after you have worked on them.

### Instructions for working locally
You can also run these notebooks on your own (Linux/Windows/Mac) computer.
This is a bit snappier than running them online.

1. **Prerequisite**: Python>=3.6.  
   If you're not a python expert:  
   1a. Install Python via [Anaconda](https://www.anaconda.com/download).  
   1b. Use the [Anaconda terminal](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#starting-conda) to run the commands below.  
   1c. (Optional) [Create & activate a new Python environment](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#managing-environments).  
   If the installation (below) fails, try doing step 1c first.

2. **Install**:  
   Run these commands in the terminal (excluding the `$` sign):  
   `$ git clone https://github.com/nansencenter/DA-tutorials.git`  
   `$ pip install -r DA-tutorials/requirements.txt`  

3. **Launch the Jupyter notebooks**:  
   `$ jupyter-notebook`  
   This will open up a page in your web browser that is a file navigator.  
   Enter the folder `DA-tutorials/notebooks`, and click on a tutorial (`T1... .ipynb`).
