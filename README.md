# catomatic

Python code that algorithmically builds antimicrobrial resistance catalogues of mutations.

## Introduction

This method relies on the logic that mutations that do not cause resistance can co-occur with those that do, and if a mutation in isolation (solo) does not cause resistance, then it will also not contribute to the phenotype when not in isolation.d

Mutations that occur in isolation across specified genes are traversed in sequence, and if their proportion of drug-susceptiblity (vs resistance) passes the specified statistical test, they are characertised as benign and removed from the data set. This step repeats while there are susceptible mutations in isolation. Once the dataset has been 'cleaned' of benign mutations, resistant mutations are classified via their proportions by the specifed test, failing which they are added to the catalogue as 'Unclassified".

The generated catalogue can be returned either as a dictionary, or as a Pandas dataframe which can be exported in a Piezo compatible format for rapid parsing and resistance predictions.

Contingency tables, proportions, p_values and Wilson's Confidence Intervals are logged under the 'EVIDENCE' column of the catalogue.

## Installation

### Using Conda

It is recommended to manage the Python environment and dependencies through Conda. You can install Catomatic within a Conda environment by following these steps:

#### Create and Activate Environment

First, ensure that you have Conda installed. Then, create and activate a new environment:

```bash
conda env create -f environment.yml
conda activate catomatic
```

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
```
