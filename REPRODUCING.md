# Reproducing our results

You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

```
git clone https://github.com/thiagokalid/acoustic-lens-ray-simulator.git
```

or downloading a zip archive from the provided DOI in the `README.md`.

All source code used to generate the results and figures in the paper is in
the `scripts` folder. There, you can find the `.py` scripts that perform the calculations and generate the
figures and results presented in the paper. 

All the generated figures are stored in `figures` folder in `.pdf` or `.png` format.

The data required to generate the results are available at the Zenodo [repository](https://doi.org/10.5281/zenodo.15115255).

## Setting up your environment

You'll need a working Python 3 environment with all packages described in `requirements.txt` or `pyproject.toml`.

Instead of manually installing them, they can all be automatically installed
using a virtual environment (venv):

1. Inside the cloned repository (or an unzipped version), create a new virtual by
   ```
   python3 -m venv /path/to/environment
   ```
   
1. Activate the new environment by running:
   ```
   source /path/to/environment/bin/activate
   ```

1. Finally, install all packages from the `requirements.txt` or `pyproject.toml`, respectively:
   ```
   python3 -m pip install -r requirements.txt
   ```
   or
   ```
   python3 -m pip install .
   ```


## Generating the results from the paper

All results and their associated figures are created by the notebooks in the
`scripts` folder. To generate results from scratch, you must run all scripts. 
Each script is assigned a number before its name; the number dictates the order
that you must follow when running them. An alternative is running `/scripts/main.py`
that executes all scripts in the proper order. 

We also made the pre-trained models publicly available on the Zenodo repository. 
Under these circumstances, you can start from the `/5_test.py` (included) to generate all
figures presented in the paper.

After running the scripts, new figures will be generated in the `figures` folder. Since all 
figures presented in the paper were previously generated and updated to the repo, it might
appear that there wasn't a new file. Some scripts might take a while to run and
require a good amount of RAM, so be aware.

With the environment activated, run each script by:

```
python3 scripts/script_name.py
