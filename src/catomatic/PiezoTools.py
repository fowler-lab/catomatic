from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Callable

import pandas as pd

from .defence_module import validate_build_piezo_inputs


class PiezoExporter(ABC):
    """
    Base class providing Piezo export utilities.

    Subclasses must implement `add_mutation()` to define how mutations are inserted
    and how insertion order is tracked.

    Notes:
        This ABC assumes the subclass exposes:
            - self.catalogue: dict-like mapping mutation -> {'pred': ..., 'evid': ...}
            - self.entry: list tracking insertion order
    """

    catalogue: MutableMapping[str, dict[str, Any]]
    entry: list[str]

    def __init__(
        self,
        catalogue: Optional[MutableMapping[str, dict[str, Any]]] = None,
        entry: Optional[list[str]] = None,
    ) -> None:
        """
        Initialize exporter state.

        Args:
            catalogue: Optional catalogue mapping to use.
            entry: Optional insertion-order list to use.

        Returns:
            None
        """
        self.catalogue = catalogue if catalogue is not None else {}
        self.entry = entry if entry is not None else []

    @abstractmethod
    def add_mutation(self, mutation: str, prediction: str, evidence: dict[str, Any]) -> None:
        """
        Add a mutation to the catalogue and update insertion order.

        Args:
            mutation: Mutation identifier.
            prediction: Phenotype label (e.g. 'R', 'S', 'U').
            evidence: Evidence metadata to associate with the entry.

        Returns:
            None
        """
        raise NotImplementedError

    def to_piezo(
        self,
        genbank_ref: str,
        catalogue_name: str,
        version: str,
        drug: str,
        wildcards: Mapping[str, dict[str, Any]] | str,
        outfile: str | Path,
        grammar: str = "GARC1",
        values: str = "RUS",
        public: bool = True,
        for_piezo: bool = True,
        json_dumps: bool = True,
        include_U: bool = True,
    ) -> None:
        """
        Export a Piezo-compatible catalogue to CSV.

        Args:
            genbank_ref: GenBank reference identifier.
            catalogue_name: Catalogue name.
            version: Catalogue version.
            drug: Drug name.
            wildcards: Wildcard rules dict or path to JSON file.
            outfile: Path to output CSV.
            grammar: Catalogue grammar (default 'GARC1').
            values: Prediction values string (default 'RUS').
            public: If True, uses and augments this instance's catalogue.
            for_piezo: If True, adds phenotype placeholders for Piezo parsing.
            json_dumps: If True, JSON-encode evidence/source/other.
            include_U: If False, exclude non-placeholder 'U' entries.

        Returns:
            None
        """
        piezo_df = self.build_piezo(
            genbank_ref=genbank_ref,
            catalogue_name=catalogue_name,
            version=version,
            drug=drug,
            wildcards=wildcards,
            grammar=grammar,
            values=values,
            public=public,
            for_piezo=for_piezo,
            json_dumps=json_dumps,
            include_U=include_U,
        )
        piezo_df.to_csv(outfile, index=False)

    def build_piezo(
        self,
        genbank_ref: str,
        catalogue_name: str,
        version: str,
        drug: str,
        wildcards: Mapping[str, dict[str, Any]] | str,
        grammar: str = "GARC1",
        values: str = "RUS",
        public: bool = True,
        for_piezo: bool = True,
        json_dumps: bool = False,
        include_U: bool = True,
    ) -> pd.DataFrame:
        """
        Build a Piezo-format catalogue DataFrame from the instance catalogue.

        Args:
            genbank_ref: GenBank reference identifier.
            catalogue_name: Catalogue name.
            version: Catalogue version.
            drug: Drug name.
            wildcards: Wildcard rules dict or path to JSON file.
            grammar: Catalogue grammar (default 'GARC1').
            values: Prediction values string (default 'RUS').
            public: If True, merges wildcards into this instance and sorts by insertion order.
            for_piezo: If True, ensures placeholders for R/S/U exist.
            json_dumps: If True, JSON-encode evidence/source/other.
            include_U: If False, exclude non-placeholder 'U' entries.

        Returns:
            Piezo-format DataFrame.
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

        # Load wildcards from file if required.
        if isinstance(wildcards, str):
            with open(wildcards) as f:
                wildcards_dict: Mapping[str, dict[str, Any]] = json.load(f)
        else:
            wildcards_dict = wildcards

        if public:
            # Merge wildcards into this instance's catalogue.
            for k, v in wildcards_dict.items():
                self.add_mutation(k, str(v["pred"]), dict(v.get("evid", {})))

            if for_piezo:
                if not any(v["pred"] == "R" for v in self.catalogue.values()):
                    self.add_mutation("placeholder@R1R", "R", {})
                if not any(v["pred"] == "S" for v in self.catalogue.values()):
                    self.add_mutation("placeholder@S1S", "S", {})
                if (not any(v["pred"] == "U" for v in self.catalogue.values())) or (not include_U):
                    self.add_mutation("placeholder@U1U", "U", {})

            data: dict[str, dict[str, Any]] = dict(self.catalogue)

            if include_U is False:
                data = {
                    k: v
                    for k, v in data.items()
                    if (v["pred"] != "U")
                    or (k == "placeholder@U1U")
                    or ("*" in k)
                    or ("del_0.0" in k)
                }
        else:
            # Internal: build from provided wildcards only (no mutation of self).
            data = {k: {"pred": v["pred"], "evid": v.get("evid", {})} for k, v in wildcards_dict.items()}

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

        # typed transformer so mypy can reason about .apply(...)
        _transformer: Callable[[Any], Any] = (lambda x: json.dumps(x)) if json_dumps else (lambda x: x)

        # build initial df from dict and rename index
        piezo_catalogue = (
            pd.DataFrame.from_dict(data, orient="index")
            .reset_index()
            .rename(columns={"index": "MUTATION", "pred": "PREDICTION", "evid": "EVIDENCE"})
        )

        piezo_catalogue["GENBANK_REFERENCE"] = genbank_ref
        piezo_catalogue["CATALOGUE_NAME"] = catalogue_name
        piezo_catalogue["CATALOGUE_VERSION"] = version
        piezo_catalogue["CATALOGUE_GRAMMAR"] = grammar
        piezo_catalogue["PREDICTION_VALUES"] = values
        piezo_catalogue["DRUG"] = drug
        piezo_catalogue["SOURCE"] = json.dumps({}) if json_dumps else ""
        piezo_catalogue["OTHER"] = json.dumps({}) if json_dumps else ""
        piezo_catalogue["EVIDENCE"] = piezo_catalogue["EVIDENCE"].apply(_transformer)

        piezo_catalogue = piezo_catalogue[columns]

        if public:
            piezo_catalogue["order"] = piezo_catalogue["MUTATION"].apply(self.entry.index)
            piezo_catalogue["PREDICTION"] = pd.Categorical(
                piezo_catalogue["PREDICTION"], categories=["S", "R", "U"], ordered=True
            )
            piezo_catalogue = (
                piezo_catalogue.sort_values(by=["PREDICTION", "order"])
                .drop(columns=["order"])
                .reindex(columns=columns)
            )

        return piezo_catalogue
