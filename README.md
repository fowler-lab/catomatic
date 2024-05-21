[![codecov](https://codecov.io/gh/fowler-lab/catomatic/branch/main/graph/badge.svg?token=8fnOy6rHCd)](https://codecov.io/gh/fowler-lab/catomatic)

# catomatic

Python code that algorithmically builds antimicrobial resistance catalogues of mutations.

## Introduction

This method relies on the logic that mutations that do not cause resistance can co-occur with those that do, and if a mutation in isolation (solo) does not cause resistance, then it will also not contribute to the phenotype when not in isolation.

Mutations that occur in isolation across specified genes are traversed in sequence, and if their proportion of drug-susceptibility (vs resistance) passes the specified statistical test, they are characterized as benign and removed from the dataset. This step repeats while there are susceptible mutations in isolation. Once the dataset has been 'cleaned' of benign mutations, resistant mutations are classified via their proportions by the specified test, failing which they are added to the catalogue as 'Unclassified'.

Construction can either rely on homogenous susceptibility for the particular mutation (and no explicit phenotyping is carried out, other than to unlock susceptible variants), use a Binomial test where the proportion of resistance is tested against a specified background rate, or a Fisher's test where the proportion of resistance is tested against a calculated background rate.

Although the method is entirely algorithmic, there are 2 entry points for intervention. Firstly, one is able to 'seed' the method with neutral mutations (such as those gathered in a literature search - often helpful if a gene contains phylogenetic mutations with high prevalence that add noise), and secondly one can add or overwrite classifications and entries to the catalogue, although not recommended unless aggregating.

Because the method uses and understands GARC1 grammar, one can supply 'rules' to the catalogue post-hoc - such as `{rpoB@*_fs:R}` for frameshifts in rpoB, which can either simply be added (and would have lower prediction priority to finer grain mutations, such as `rpoB@44_ins`) or can replace any mutations that fall under that rule, effectively aggregating relevant variants.

The generated catalogue can be returned either as a dictionary, or as a Pandas dataframe which can be exported in a Piezo compatible format for rapid parsing and resistance predictions.

Contingency tables, proportions, p_values, and Wilson's Confidence Intervals are logged under the 'EVIDENCE' column of the catalogue.

## Installation

### Using Conda

It is recommended to manage the Python environment and dependencies through Conda. You can install Catomatic within a Conda environment by following these steps:

#### Create and Activate Environment

First, ensure that you have Conda installed. Then, create and activate a new environment:

````bash
conda env create -f environment.yml
conda activate catomatic

## Running catomatic

### CLI

After installation, the simplest way to run the catomatic catalogue builder is via the command line interface. --to_piezo or --to_json flags will need to specified to save the catalogue (with additional arguments if using --to_piezo)

`BuildCatalogue --samples path/to/samples.csv --mutations path/to/mutations.csv  --to_json --outfile path/to/out/catalogue.json`

or

`BuildCatalogue --samples path/to/samples.csv --mutations path/to/mutations.csv  --to_piezo --outfile path/to/out/catalogue.csv --genbank_ref '...' --catalogue_name '...' --version '...' --drug '...' --wildcards path/to/wildcards.json`

### Python/Jupyter notebook

Should you which to run catomatic in a notebook, for example, you can do so simply by calling BuildCatalogue after import.

```python
import catomatic
#instantiate a catalogue object - this will build the catalogue
catalogue = catomatic.BuildCatalogue(samples = samples_df, mutations = mutations_df)

#return the catalogue as a dictionary in order of variant addition
catalogue.return_catalogue()

#return the catalogue as a piezo-structured dataframe
catalogue.return_piezo(genbank_ref='...', catalogue_name='...', version='...', drug='...', wildcards='path/to/wildcards.json')
````

### CLI Parameters

| Parameter          | Type    | Description                                                                                       |
| ------------------ | ------- | ------------------------------------------------------------------------------------------------- |
| `--samples`        | `str`   | Path to the samples file. Required.                                                               |
| `--mutations`      | `str`   | Path to the mutations file. Required.                                                             |
| `--FRS`            | `float` | Fraction Read Support threshold. Optional.                                                        |
| `--seed`           | `list`  | List of seed mutations. Optional.                                                                 |
| `--test`           | `str`   | Type of statistical test to run: `None`, `Binomial`, or `Fisher`. Optional.                       |
| `--background`     | `float` | Background mutation rate for the binomial test. Optional.                                         |
| `--p`              | `float` | Significance level for statistical testing. Optional. Defaults to `0.95`.                         |
| `--strict_unlock`  | `bool`  | Enforce strict unlocking for classifications. Optional.                                           |
| `--to_json`        | `bool`  | Export the catalogue to JSON format. Optional.                                                    |
| `--outfile`        | `str`   | Path to output file for exporting the catalogue. Used with `--to_json` or `--to_piezo`. Optional. |
| `--to_piezo`       | `bool`  | Export catalogue to Piezo format. Optional.                                                       |
| `--genbank_ref`    | `str`   | GenBank reference for the catalogue. Optional.                                                    |
| `--catalogue_name` | `str`   | Name of the catalogue. Optional.                                                                  |
| `--version`        | `str`   | Version of the catalogue. Optional.                                                               |
| `--drug`           | `str`   | Drug associated with the mutations. Optional.                                                     |
| `--wildcards`      | `str`   | JSON file with wildcard rules. Optional.                                                          |
| `--grammar`        | `str`   | Grammar used in the catalogue. Optional. Defaults to `GARC1`.                                     |
| `--values`         | `str`   | Values used for predictions in the catalogue. Optional. Defaults to `RUS`.                        |
