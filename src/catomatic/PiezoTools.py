import json
import pandas as pd
from .defence_module import validate_build_piezo_inputs

class PiezoExporter:

    def __init__(self, catalogue=None, entry=None):
        """
        Initialize the PiezoExporter with optional catalogue and entry.

        Parameters:
            catalogue (dict, optional): A dictionary representing the mutation catalogue.
            entry (list, optional): A list representing the order of mutations in the catalogue.
        """
        self.catalogue = catalogue if catalogue is not None else {}
        self.entry = entry if entry is not None else []

    def to_piezo(
        self,
        genbank_ref,
        catalogue_name,
        version,
        drug,
        wildcards,
        outfile,
        grammar="GARC1",
        values="RUS",
        public=True,
        for_piezo=True,
        json_dumps=True,
        include_U=True,
    ):
        """
        Exports a pizeo-compatible dataframe as a csv file.

        Parameters:
            genbank_ref (str): GenBank reference identifier.
            catalogue_name (str): Name of the catalogue.
            version (str): Version of the catalogue.
            drug (str): Target drug associated with the mutations.
            wildcards (dict): Piezo wildcard (default rules) mutations with phenotypes.
            outfile: The path to the output csv file where the catalogue will be saved.
            grammar (str, optional): Grammar used in the catalogue, default "GARC1" (no other grammar currently supported).
            values (str, optional): Prediction values, default "RUS" representing each phenotype (no other values currently supported).
            public (bool, optional): private or public call
            for_piezo (bool, optional): Whether to include the missing phenotype placeholders (only piezo requires them)

        """

        piezo_df = self.build_piezo(
            genbank_ref,
            catalogue_name,
            version,
            drug,
            wildcards,
            grammar,
            values,
            public,
            for_piezo,
            json_dumps,
            include_U,
        )
        
        piezo_df.to_csv(outfile)

    def build_piezo(
        self,
        genbank_ref,
        catalogue_name,
        version,
        drug,
        wildcards,
        grammar="GARC1",
        values="RUS",
        public=True,
        for_piezo=True,
        json_dumps=False,
        include_U=True,
    ):
        """
        Builds a piezo-format catalogue df from the catalogue object.

        Parameters:
            genbank_ref (str): GenBank reference identifier.
            catalogue_name (str): Name of the catalogue.
            version (str): Version of the catalogue.
            drug (str): Target drug associated with the mutations.
            wildcards (dict or path): Piezo wildcard (default rules) mutations with phenotypes.
            grammar (str, optional): Grammar used in the catalogue, default "GARC1" (no other grammar currently supported).
            values (str, optional): Prediction values, default "RUS" representing each phenotype (no other values currently supported).
            public (bool, optional): private or public call
            for_piezo (bool, optional): Whether to include the missing phenotype placeholders (only piezo requires them)
            json_dumps (bool, optional): Whether to dump evidence column into json object for piezo (e.g if in notebook, unnecessary)
            include_U (bool, optional): Whether to add unclassified mutations to catalogue

        Returns:
            self: instance with piezo_catalogue set
        """

        validate_build_piezo_inputs(
            genbank_ref,
            catalogue_name,
            version,
            drug,
            wildcards,
            grammar,
            values,
            public,
            for_piezo,
            json_dumps,
            include_U,
        )

        # if user-called
        if public:
            # add piezo wildcards to the catalogue
            if isinstance(wildcards, str):
                # if a path is supplied, read from the file
                with open(wildcards) as f:
                    wildcards = json.load(f)
            [self.add_mutation(k, v["pred"], v["evid"]) for k, v in wildcards.items()]
            # inlcude a placeholder for each phenotype if don't exist - piezo requires all R, U, S to parse
            if for_piezo:
                if not any(v["pred"] == "R" for v in self.catalogue.values()):
                    self.add_mutation("placeholder@R1R", "R", {})
                if not any(v["pred"] == "S" for v in self.catalogue.values()):
                    self.add_mutation("placeholder@S1S", "S", {})
                if (
                    not any(v["pred"] == "U" for v in self.catalogue.values())
                    or not include_U
                ):
                    self.add_mutation("placeholder@U1U", "U", {})
            data = self.catalogue
            if include_U == False:
                data = {
                    k: v
                    for k, v in data.items()
                    if (v["pred"] != "U")
                    or (k == "placeholder@U1U")
                    or ("*" in k)
                    or ("del_0.0" in k)
                }
        else:
            # if internal:
            data = wildcards

        columns = [
            "GENBANK_REFERENCE",
            "CATALOGUE_NAME",
            "CATALOGUE_VERSION",
            "CATALOGUE_GRAMMAR",
            "PREDICTION_VALUES",
            "DRUG",
            "MUTATION",
            "PREDICTION",
            "SOURCE",
            "EVIDENCE",
            "OTHER",
        ]
        # construct the catalogue dataframe in piezo-standardised format
        piezo_catalogue = (
            pd.DataFrame.from_dict(data, orient="index")
            .reset_index()
            .rename(
                columns={
                    "index": "MUTATION",
                    "pred": "PREDICTION",
                    "evid": "EVIDENCE",
                }
            )
            .assign(
                GENBANK_REFERENCE=genbank_ref,
                CATALOGUE_NAME=catalogue_name,
                CATALOGUE_VERSION=version,
                CATALOGUE_GRAMMAR=grammar,
                PREDICTION_VALUES=values,
                DRUG=drug,
                SOURCE=json.dumps({}) if json_dumps else {},
                EVIDENCE=lambda df: df["EVIDENCE"].apply(
                    json.dumps if json_dumps else lambda x: x
                ),
                OTHER=json.dumps({}) if json_dumps else {},
            )[columns]
        )

        if public:
            # Create a temporary column for the order in self.entry
            piezo_catalogue["order"] = piezo_catalogue["MUTATION"].apply(
                lambda x: self.entry.index(x)
            )

            # Sort by PREDICTION and the temporary order column
            piezo_catalogue["PREDICTION"] = pd.Categorical(
                piezo_catalogue["PREDICTION"], categories=["S", "R", "U"], ordered=True
            )
            piezo_catalogue = piezo_catalogue.sort_values(by=["PREDICTION", "order"])

            # Drop the temporary order column
            piezo_catalogue = piezo_catalogue.drop(columns=["order"])
            piezo_catalogue = piezo_catalogue[columns]

        return piezo_catalogue
