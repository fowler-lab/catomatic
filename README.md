[![codecov](https://codecov.io/gh/fowler-lab/catomatic/branch/ecoff/graph/badge.svg?token=8fnOy6rHCd)](https://codecov.io/gh/fowler-lab/catomatic) [![DOI](https://zenodo.org/badge/801462003.svg)](https://doi.org/10.5281/zenodo.14917920)


# catomatic

Python code that algorithmically builds antimicrobial resistance catalogues of mutations.

## Introduction

This repo contains 2 approaches to build resistance catalogues:

1. **Definite defectives (solo-based approach)**
2. **Interval regression**

The first is used in [https://doi.org/10.1101/2025.01.30.635633](https://doi.org/10.1101/2025.01.30.635633), and the second is a Python translation of the method used in [https://doi.org/10.1038/s41467-023-44325-5](https://doi.org/10.1038/s41467-023-44325-5), but is still under development.

---

## Binary Builder

This method relies on the logic that mutations that do not cause resistance can co-occur with those that do. If a mutation in isolation (solo) does not cause resistance, then it is not contributing to the phenotype when present in a mixture either.

Mutations that occur in isolation across specified genes are traversed in sequence, and if their proportion of drug-susceptibility (vs. resistance) passes the specified statistical test, they are characterized as benign (susceptible) and removed from the dataset. This step repeats iteratively until no more solo benign mutations are found.

Remaining mutations are classified based on their resistance rates and statistical test results. Those that don’t meet thresholds are labeled as `U`.

The classification approach supports:

- **No test**: assumes homogeneous susceptibility is sufficient for S
- **Binomial test**: against a specified background resistance rate
- **Fisher's test**: using a contingency background

### Optional Interventions

1. **Seeding**: You can pre-seed the catalogue with known neutral mutations.
2. **Overrides**: You can override or supplement the final catalogue with manual or rule-based entries.

Because the method uses GARC1 grammar, rules like `{rpoB@*_fs:R}` can be supplied post-hoc. These rules can:

- Be additive (lower priority than specific entries)
- Replace matching mutations (requires `replace=True` and `wildcards` supplied)

The catalogue can be returned as a dictionary or a Piezo-compatible `pandas.DataFrame`.

Contingency tables, proportions, p-values, and Wilson confidence intervals are stored in the `EVIDENCE` field.

### Example Workflow using Fisher's:

![Catalogue Diagram](docs/workflow.png)

---

## Regression Builder

This method is under development and will be released soon with accompanying documentation.

---

## Installation

### Using Conda

We recommend using Conda for environment and dependency management.

```bash
conda env create -f env.yml
conda activate catomatic
pip install .
```

## Running catomatic's Binary Builder

You need two input DataFrames:

- **Samples**: one row per sample, with 'R' or 'S' phenotypes (`UNIQUEID`, `PHENOTYPE`)
- **Mutations**: one row per mutation per sample (`UNIQUEID`, `MUTATION`)

If exporting to Piezo format:

- The `MUTATION` column must follow GARC1 grammar (`gene@mutation`)
- A path to a `wildcards.json` file (containing mutation rules) must be provided

### Python/Jupyter Example

```python
from catomatic.BinaryCatalogue import BinaryBuilder

# Build catalogue
catalogue = BinaryBuilder(samples=samples_df, mutations=mutations_df).build()

# View dictionary version
cat_dict = catalogue.return_catalogue()

# Convert to Piezo-compatible format
catalogue_df = catalogue.build_piezo(
    genbank_ref='...',
    catalogue_name='...',
    version='...',
    drug='...',
    wildcards='path/to/wildcards.json'
)

# Optionally export to CSV
catalogue.to_piezo(
    genbank_ref='...',
    catalogue_name='...',
    version='...',
    drug='...',
    wildcards='path/to/wildcards.json',
    outfile='path/to/output.csv'
)
```

### CLI

After installation, the simplest way to run the catomatic catalogue builder is via the command line interface using the `binary` subcommand. You must use either `--to_piezo` or `--to_json` to specify the output format. Additional metadata is required when using `--to_piezo`.

#### Export to JSON

```bash
python -m catomatic binary \
  --samples path/to/samples.csv \
  --mutations path/to/mutations.csv \
  --to_json \
  --outfile path/to/output/catalogue.json
```

#### Export to Piezo format

```bash
python -m catomatic binary \
  --samples path/to/samples.csv \
  --mutations path/to/mutations.csv \
  --to_piezo \
  --outfile path/to/output/catalogue.csv \
  --genbank_ref '...' \
  --catalogue_name '...' \
  --version '...' \
  --drug '...' \
  --wildcards path/to/wildcards.json
```

### CLI Parameters

| Parameter          | Type    | Description                                                                                    |
| ------------------ | ------- | ---------------------------------------------------------------------------------------------- |
| `--samples`        | `str`   | Path to the samples (phenotypes) file. Required.                                               |
| `--mutations`      | `str`   | Path to the mutations file. Required.                                                          |
| `--outfile`        | `str`   | Output file path for saving the catalogue. Required with `--to_json` or `--to_piezo`.          |
| `--to_json`        | `flag`  | Export the resulting catalogue in JSON format. Optional.                                       |
| `--to_piezo`       | `flag`  | Export the resulting catalogue in Piezo-compatible CSV format. Optional.                       |
| `--genbank_ref`    | `str`   | GenBank reference string for Piezo export. Required with `--to_piezo`.                         |
| `--catalogue_name` | `str`   | Name of the catalogue. Required with `--to_piezo`.                                             |
| `--version`        | `str`   | Catalogue version. Required with `--to_piezo`.                                                 |
| `--drug`           | `str`   | Name of the drug. Required with `--to_piezo`.                                                  |
| `--wildcards`      | `str`   | Path to JSON file containing wildcard mutation definitions. Required with `--to_piezo`.        |
| `--test`           | `str`   | Type of statistical test to apply. One of: `None`, `Binomial`, or `Fisher`. Optional.          |
| `--background`     | `float` | Background mutation rate (0–1). Required if `--test Binomial` is used.                         |
| `--p`              | `float` | P-value threshold for statistical significance. Optional. Defaults to `0.95`.                  |
| `--tails`          | `str`   | Tail type for statistical test. One of: `one`, `two`. Optional. Defaults to `two`.             |
| `--strict_unlock`  | `flag`  | If set, disables classification of susceptible (`S`) mutations unless statistically confident. |

### Notes

- When using post-hoc rule updates via .update(), you must provide wildcards and set replace=True if you intend to override existing entries.
- For Piezo export, placeholder entries are inserted automatically if needed to satisfy parser requirements (R, S, and U must be represented).
- The EVIDENCE column includes contingency tables, proportions, confidence intervals, and p-values, and may optionally include sample IDs if `record_ids=True`.

## Citation

If you use catomatic in your research, please cite:

- https://doi.org/10.1101/2025.01.30.635633
