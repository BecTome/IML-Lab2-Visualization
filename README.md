# IML-Lab1-Clustering

## Modules
- analysis
    - `data_selection.py`: Run to generate pandas profiles fot all the datasets. The output folder is not sync-ed with git.

## Environment setup
1. Developed on PyCharm using Conda environment + Python 3.7
2. Working Directory: `IML-Lab1-Clustering`
3. Export conda env `conda env export --no-builds > environment.yml`
4. Restore conda env `conda env create -f environment.yml`

## How to run the code

**Important**: In the editor, set the working directory to `IML-Lab1-Clustering` before running the code.

```
python main.py
```

**Note**: PAM algorithm takes longer than the rest.

## Configuration

In `config.py`, you can configure the following parameters:

- `Hyperparameters grid`
- `Feature selection`
- `Clustering algorithms`
- `Datasets`
- `Output folder`

## More details

A highly automatized process has been designed to perform fast and efficient clustering on multiple datasets. The process is as follows:

1. Load the datasets
2. Perform feature selection
3. Perform clustering
4. Evaluate the clustering
5. Save the results

Everything is inside the `src.trainflow` library