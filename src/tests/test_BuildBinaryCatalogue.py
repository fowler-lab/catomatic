import sys
import io
import pytest
import json
import subprocess
from pathlib import Path
import pandas as pd
from catomatic.BinaryCatalogue import BinaryBuilder
from scipy.stats import binomtest, fisher_exact
from catomatic.cli import parse_binary_builder, main_binary_builder



# a left join of phenotypes and mutations will give:
"""   
UNIQUEID PHENOTYPE MUTATION
0         1         R     gene@A1S
1         2         S     gene@A1S
2         3         S      NaN
3         4         S     gene@A2S
4         5         R      NaN
5         6         S      NaN
6         7         S     gene@A3S
7         8         S     gene@A3S
8         9         R     gene@A3S
9        10         R      NaN
"""
# this should allow for pretty robust statistical condition testing
# have pushed background very R to test fisher S classifications
# the only function that uses the backgrounds is test_fisher_build_S (R/U uses explicit tables)


@pytest.fixture
def mutation_data():
    return pd.DataFrame(
        {
            "UNIQUEID": [1, 2, 4, 7, 8, 9],
            "MUTATION": [
                "gene@A1S",
                "gene@A1S",
                "gene@A2S",
                "gene@A3S",
                "gene@A3S",
                "gene@A3S",
            ],
        }
    )


@pytest.fixture
def phenotype_data():
    return pd.DataFrame(
        {
            "UNIQUEID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "PHENOTYPE": ["R", "S", "R", "S", "R", "S", "S", "S", "R", "R"],
        }
    )


@pytest.fixture
def phenotypes_file(phenotype_data, tmp_path):
    file = tmp_path / "phenotypes.csv"
    phenotype_data.to_csv(file, index=False)
    return str(file)


@pytest.fixture
def mutations_file(mutation_data, tmp_path):
    file = tmp_path / "mutations.csv"
    mutation_data.to_csv(file, index=False)
    return str(file)


@pytest.fixture
def output_file(tmp_path):
    return str(tmp_path / "output.json")


@pytest.fixture
def solo_data(phenotype_data, mutation_data):
    return pd.merge(phenotype_data, mutation_data, on=["UNIQUEID"], how="left")


@pytest.fixture
def wildcards():
    return {"gene@-*?": {"pred": "U", "evid": {}}, "gene@*=": {"pred": "S", "evid": {}}}


@pytest.fixture
def builder(request, phenotype_data, mutation_data):
    # default parameters
    test_type = None
    background = 0.1
    p = 0.95
    strict_unlock = False

    # check if the test that's calling this fixture is parametrized
    if hasattr(request, "param"):
        # override defaults
        test_type = request.param.get("test", test_type)
        background = request.param.get("background", background)
        p = request.param.get("p", p)
        strict_unlock = request.param.get("strict_unlock", strict_unlock)

    return BinaryBuilder(
        phenotype_data,
        mutation_data,
        
    ).build(
        background=background,
        strict_unlock=strict_unlock,
        test=test_type,
        p=p,
    )


def test_add_mutation(builder):
    # using default parameterisation

    mutation = "mut4"
    prediction = "R"
    evidence = {"detail": "test"}

    builder.add_mutation(mutation, prediction, evidence)
    assert mutation in builder.catalogue
    assert builder.catalogue[mutation]["pred"] == prediction
    assert builder.catalogue[mutation]["evid"] == evidence


def test_build_contingency(solo_data):
    # for R + S
    x_mut1 = BinaryBuilder.build_contingency(solo_data, "gene@A1S")[0]
    assert x_mut1 == [[1, 1], [3, 1]]
    # for R
    x_mut2 = BinaryBuilder.build_contingency(solo_data, "gene@A2S")[0]
    assert x_mut2 == [[0, 1], [3, 1]]
    # For non-existent mutation - although should be filtered out beforehand
    x_mut4 = BinaryBuilder.build_contingency(solo_data, "gene@A4S")[0]
    assert x_mut4 == [[0, 0], [3, 1]]


def test_calc_proportion():
    x_tests = [
        ([[10, 5], [3, 7]], 10 / (10 + 5)),
        ([[20, 0], [5, 5]], 1),
        ([[0, 30], [5, 5]], 0),
    ]

    for x, expected in x_tests:
        assert (
            BinaryBuilder.calc_proportion(x) == expected
        ), f"Failed for contingency {x}"


def test_calc_odds_ratio():
    x_tests = [
        ([[10, 5], [3, 7]], 45 / 11),
        ([[20, 0], [5, 5]], 41),
        ([[0, 30], [5, 5]], 1 / 61),
    ]

    for x, expected in x_tests:
        assert (
            BinaryBuilder.calc_odds_ratio(x) == expected
        ), f"Failed for contingency {x}"


@pytest.mark.parametrize("builder", [{"p": 0.95}], indirect=True)
def test_calc_confidence_interval(builder):

    x_tests = [
        ([[10, 5], [3, 7]], [0.4171, 0.8482]),
        ([[20, 0], [5, 5]], [0.8389, 1]),
        ([[0, 30], [5, 5]], [0, 0.1135]),
    ]

    for x, expected in x_tests:
        ci = builder.calc_confidence_interval(x)
        ci = [round(ci[0], 4), round(ci[1], 4)]

        assert ci == expected, f"Failed for contingency {x}"


@pytest.mark.parametrize("builder", [{"p": 0.95}], indirect=True)
def test_skeleton_build(builder):
    # uses default parameters (ie test=None), which should fire the skeleton build

    # now test addition when run_iter = False
    mutation = "mut4"
    x = [[27, 21], [3, 22]]

    builder.skeleton_build(mutation, x)
    # check its actually been added
    assert mutation in builder.catalogue
    # check if the correct phenotype has been added
    assert builder.catalogue[mutation]["pred"] == "U"

    expected_e = {
        "proportion": builder.calc_proportion(x),
        "confidence": builder.calc_confidence_interval(x),
        "contingency": x,
    }

    assert builder.catalogue[mutation]["evid"] == expected_e


@pytest.mark.parametrize(
    "builder",
    [{"test": "Binomial", "background": 0.1, "strict_unlock": False, "p": 0.95}],
    indirect=True,
)
def test_binomial_build_RU(builder):
    """tests when run_iter = False (ie, classiyfing R and U variants)"""

    # now test additions when run_iter = False
    mutations = ["mut4", "mut5"]
    data_tests = [
        ([[27, 21], [3, 22]], "R"),
        ([[5, 21], [3, 22]], "U"),
    ]

    for mut in range(len(mutations)):
        mutation = mutations[mut]
        phenotype = data_tests[mut][1]
        x = data_tests[mut][0]

        p_expected = binomtest(
            x[0][0], (x[0][0] + x[0][1]), 0.1, alternative="two-sided"
        ).pvalue

        expected_e = {
            "proportion": builder.calc_proportion(x),
            "confidence": builder.calc_confidence_interval(x),
            "p_value": p_expected,
            "contingency": x,
        }

        builder.binomial_build(mutation, x)
        # make sure mutaiton was added
        assert mutation in builder.catalogue, f"Failed for contingency {x}"
        # make sure phenotype was determined correclty
        assert (
            builder.catalogue[mutation]["pred"] == phenotype
        ), f"Failed for contingency {x}"
        # make sure evidence was added correclty
        assert (
            builder.catalogue[mutation]["evid"] == expected_e
        ), f"Failed for contingency {x}"

        # fyi haven't tested entire decision tree for run_iter=True, but they're are almost identical


@pytest.mark.parametrize(
    "builder",
    [{"test": "Fisher", "strict_unlock": False, "p": 0.95}],
    indirect=True,
)
def test_fisher_build_RU(builder):
    """tests when run_iter = False (ie classifying R and U variants)"""

    # now test additions when run_iter = False
    mutations = ["mut4", "mut5"]
    data_tests = [
        ([[27, 21], [3, 22]], "R"),
        ([[5, 21], [3, 22]], "U"),
    ]

    for mut in range(len(mutations)):
        mutation = mutations[mut]
        phenotype = data_tests[mut][1]
        x = data_tests[mut][0]

        _, p_expected = fisher_exact(x)

        expected_e = {
            "proportion": builder.calc_proportion(x),
            "confidence": builder.calc_confidence_interval(x),
            "p_value": p_expected,
            "contingency": x,
        }

        builder.fishers_build(mutation, x)
        # make sure mutaiton was added
        assert mutation in builder.catalogue, f"Failed for contingency {x}"
        # make sure phenotype was determined correclty
        assert (
            builder.catalogue[mutation]["pred"] == phenotype
        ), f"Failed for contingency {x}"
        # make sure evidence was added correclty
        assert (
            builder.catalogue[mutation]["evid"] == expected_e
        ), f"Failed for contingency {x}"


# the S tests below are conceptually more complex than the RU tests, as these use the
# instantiation data, and therefore implicitly also test the classify function


@pytest.mark.parametrize(
    "builder",
    [{"test": "Binomial", "background": 0.9, "strict_unlock": False, "p": 0.95}],
    indirect=True,
)
def test_binomial_build_S(builder):
    # test proportion 0 (not strict) + p_calc<self.p (added and not added)
    # using background = 0.9 to force it without needing tonnes of data
    # havent tested propotion 0 (strict) but i think its fine

    s_entries = {
        key: val for key, val in builder.catalogue.items() if val["pred"] == "S"
    }
    # check only mut2 and mut3 were added
    assert "gene@A1S" not in s_entries
    assert "gene@A2S" in s_entries
    assert "gene@A3S" in s_entries


@pytest.mark.parametrize(
    "builder",
    [{"test": "Fisher", "strict_unlock": False, "p": 0.5}],
    indirect=True,
)
def test_fisher_build_S(builder):
    # test proportion 0 (not strict) + p_calc<self.p (added and not added)
    # using bp value 0.5 and very high R background to force S classifications
    # havent tested propotion 0 (strict) but i think its fine

    s_entries = {
        key: val for key, val in builder.catalogue.items() if val["pred"] == "S"
    }
    # check only mut2 and mut3 were added
    assert "gene@A1S" not in s_entries
    assert "gene@A2S" in s_entries
    assert "gene@A3S" in s_entries


@pytest.mark.parametrize("builder", [{"p": 0.95}], indirect=True)
def test_classify(builder):
    # this a bit redundant as the test_S functions above inplicity test the classify function

    # check that once there is 1 susceptible entry in the catalogue (gene@A1S), run_iter switches to false
    s_entries = {
        key: val for key, val in builder.catalogue.items() if val["pred"] == "S"
    }
    assert len(s_entries) == 1
    assert "gene@A2S" in s_entries.keys()
    assert builder.run_iter == False


@pytest.mark.parametrize("builder", [{"p": 0.95}], indirect=True)
def test_update_catalogue(builder, wildcards):

    # check addition to the catalogue with replacement
    assert builder.catalogue["gene@A1S"]["pred"] == "U"
    builder.update_catalogue({"gene@A1S": "R"})
    assert builder.catalogue["gene@A1S"]["pred"] == "R"
    # check addition to the catalogue with wildcard and replacement
    builder.update_catalogue({"gene@*?": "S"}, wildcards, replace=True)
    assert builder.catalogue["gene@*?"]["pred"] == "S"
    assert "gene@A2S" not in builder.catalogue.keys()
    # check addition to the catalogue without replacement
    builder.update_catalogue({"gene@A5S": "R"}, wildcards, replace=False)
    assert builder.catalogue["gene@A5S"]["pred"] == "R"
    assert builder.catalogue["gene@*?"]["pred"] == "S"


@pytest.mark.parametrize("builder", [{"p": 0.95}], indirect=True)
def test_build_piezo(builder, wildcards):
    # build a piezo compitable catalogue df
    catalogue = builder.build_piezo(
        "genbank", "test", "1", "drug", wildcards, grammar="GARC1", values="RUS"
    )
    # test basic argument additions to catalogue df
    assert catalogue.GENBANK_REFERENCE[0] == "genbank"
    assert catalogue.CATALOGUE_NAME[0] == "test"
    assert catalogue.CATALOGUE_VERSION[0] == "1"
    assert catalogue.DRUG[0] == "drug"
    # check mutations are in the catalogue, , with correct classification
    assert catalogue[catalogue.MUTATION == "gene@A2S"].PREDICTION.values[0] == "S"
    # check wildcards are in the catalogue, with correct classification
    assert catalogue[catalogue.MUTATION == "gene@*="].PREDICTION.values[0] == "S"


def test_cli_help():
    parser = parse_binary_builder()
    buf = io.StringIO()
    parser.print_help(file=buf)
    help_text = buf.getvalue()
    assert "usage" in help_text.lower()
    # optional sanity checks for a couple arguments
    assert "--samples" in help_text
    assert "--mutations" in help_text


def test_cli_execution(phenotypes_file, mutations_file, output_file):
    parser = parse_binary_builder()

    args_list = [
        "--samples",
        phenotypes_file,
        "--mutations",
        mutations_file,
        "--to_json",
        "--outfile",
        output_file,
        "--test",
        "Binomial",
        "--background",
        "0.1",
        "--p",
        "0.95",
        "--strict_unlock",
    ]

    args = parser.parse_args(args_list)

    # call the main handler directly; if it raises, pytest will show the traceback
    result = main_binary_builder(args)

    # If main_binary_builder returns an int status, assert it's zero, else assert output exists.
    if isinstance(result, int):
        assert result == 0
    else:
        # ensure outfile was created (some implementations may write file and return None)
        assert Path(output_file).exists(), "Output file not created"



def test_to_json_output(phenotypes_file, mutations_file, output_file):
    parser = parse_binary_builder()
    args_list = [
        "--samples",
        phenotypes_file,
        "--mutations",
        mutations_file,
        "--to_json",
        "--outfile",
        output_file,
    ]
    args = parser.parse_args(args_list)
    result = main_binary_builder(args)

    if isinstance(result, int):
        assert result == 0
    else:
        assert Path(output_file).exists(), "JSON outfile not created"

    with open(output_file, "r") as f:
        data = json.load(f)

    assert isinstance(data, dict)
    assert "gene@A1S" in data
    assert "gene@A2S" in data
    assert "gene@A3S" in data



def test_missing_piezo(phenotypes_file, mutations_file, output_file):
    """
    Expect the CLI to fail (non-zero exit) when required --to_piezo arguments
    (e.g. --drug, --wildcards) are missing.
    """
    parser = parse_binary_builder()
    args_list = [
        "--samples",
        phenotypes_file,
        "--mutations",
        mutations_file,
        "--to_piezo",
        "--outfile",
        output_file,
        "--genbank_ref",
        "genbank",
        "--catalogue_name",
        "test",
        "--version",
        "1",
    ]  # missing --drug and --wildcards

    args = parser.parse_args(args_list)

    # Accept SystemExit with non-zero code, any exception, or a non-zero int return.
    try:
        result = main_binary_builder(args)
    except SystemExit as e:
        # argparse or code may call sys.exit(1). Ensure exit code is non-zero.
        code = e.code
        # SystemExit.code might be None, an int, or a string - normalize
        try:
            code_int = int(code) if code is not None else 1
        except Exception:
            code_int = 1
        assert code_int != 0, "Expected non-zero exit code for missing piezo args"
        return
    except Exception:
        # Any other exception is acceptable for this negative test.
        return

    # If we get here there was no exception. Expect a non-zero int return to indicate failure.
    if isinstance(result, int):
        assert result != 0, "Expected non-zero return code for missing piezo args"
    else:
        pytest.fail("Expected CLI to fail for missing piezo args, but it succeeded")



def test_to_piezo_output(phenotypes_file, mutations_file, output_file, tmp_path):
    wildcards_file = tmp_path / "wildcards.json"
    wildcards_file.write_text(
        json.dumps(
            {
                "gene@-*?": {"pred": "U", "evid": {}},
                "gene@*=": {"pred": "S", "evid": {}},
            }
        )
    )

    parser = parse_binary_builder()
    args_list = [
        "--samples",
        phenotypes_file,
        "--mutations",
        mutations_file,
        "--to_piezo",
        "--outfile",
        output_file,
        "--genbank_ref",
        "genbank",
        "--catalogue_name",
        "test",
        "--version",
        "1",
        "--drug",
        "drug",
        "--wildcards",
        str(wildcards_file),
    ]

    args = parser.parse_args(args_list)
    result = main_binary_builder(args)

    if isinstance(result, int):
        assert result == 0
    else:
        assert Path(output_file).exists(), "Piezo CSV not created"

    piezo_df = pd.read_csv(output_file)
    assert "GENBANK_REFERENCE" in piezo_df.columns
    assert piezo_df.loc[0, "GENBANK_REFERENCE"] == "genbank"
    assert piezo_df.loc[0, "CATALOGUE_NAME"] == "test"
    # some implementations write version as string (ensure equals numerically or as string)
    v = piezo_df.loc[0, "CATALOGUE_VERSION"]
    assert int(v) == 1
    assert piezo_df.loc[0, "DRUG"] == "drug"
    assert "gene@A2S" in piezo_df["MUTATION"].values
    assert "gene@*=" in piezo_df["MUTATION"].values

