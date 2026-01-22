import argparse


def parse_binary_builder():
    parser = argparse.ArgumentParser(
        description="Build a catalogue using the binary frequentist approach"
    )
    parser.add_argument(
        "--samples", required=True, type=str, help="Path to the samples file."
    )
    parser.add_argument(
        "--mutations", required=True, type=str, help="Path to the mutations file."
    )
    parser.add_argument(
        "--frs",
        type=float,
        default=None,
        help="Optional: Fraction Read Support threshold.",
    )
    parser.add_argument("--seed", nargs="+", help="Optional: List of seed mutations.")
    parser.add_argument(
        "--test",
        choices=["Binomial", "Fisher"],
        default=None,
        help="Optional: Type of statistical test to run.",
    )
    parser.add_argument(
        "--background",
        type=float,
        default=None,
        help="Optional: Background mutation rate for the binomial test.",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.95,
        help="Significance level for statistical testing.",
    )
    parser.add_argument(
        "--strict_unlock",
        action="store_true",
        help="Enforce strict unlocking for classifications.",
    )
    parser.add_argument(
        "--record_ids",
        action="store_true",
        help="Whether to record UNIQUEIDS in the catalogue.",
    )
    parser.add_argument(
        "--to_json",
        action="store_true",
        help="Flag to trigger exporting the catalogue to JSON format.",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        help="Path to output file for exporting the catalogue. Used with --to_json or --to_piezo.",
    )
    parser.add_argument(
        "--to_piezo",
        action="store_true",
        help="Flag to export catalogue to Piezo format.",
    )
    parser.add_argument(
        "--genbank_ref", type=str, help="GenBank reference for the catalogue."
    )
    parser.add_argument("--catalogue_name", type=str, help="Name of the catalogue.")
    parser.add_argument("--version", type=str, help="Version of the catalogue.")
    parser.add_argument("--drug", type=str, help="Drug associated with the mutations.")
    parser.add_argument("--wildcards", type=str, help="JSON file with wildcard rules.")
    parser.add_argument(
        "--grammar", type=str, default="GARC1", help="Grammar used in the catalogue."
    )
    parser.add_argument(
        "--values",
        type=str,
        default="RUS",
        help="Values used for predictions in the catalogue.",
    )
    parser.add_argument(
        "--for_piezo",
        action="store_true",
        help="If not planning to use piezo, set to False to avoid placeholder rows being added",
    )
    return parser


def main_binary_builder(args):
    from catomatic.BinaryCatalogue import BinaryBuilder

    # No re-parsing, use `args` passed from __main__.py
    builder = BinaryBuilder(
        samples=args.samples,
        mutations=args.mutations,
        frs=args.frs,
        seed=args.seed,
    )

    builder.build(
        test=args.test,
        background=args.background,
        p=args.p,
        strict_unlock=args.strict_unlock,
        record_ids=args.record_ids,
    )

    if args.to_json:
        main_json_exporter(builder, args)

    if args.to_piezo:
        main_piezo_exporter(builder, args)


def parse_regression_builder():
    """
    Parse command-line options for the RegressionBuilder class.

    Returns:
        argparse.Namespace: Parsed arguments from the command line.
    """
    parser = argparse.ArgumentParser(
        description="Build a regression-based mutation catalogue."
    )
    parser.add_argument(
        "--samples", required=True, type=str, help="Path to the samples file (CSV)."
    )
    parser.add_argument(
        "--mutations", required=True, type=str, help="Path to the mutations file (CSV)."
    )
    parser.add_argument(
        "--genes",
        nargs="+",
        default=[],
        help="A list of RAV genes. A list must be supplied if non-RAV genes are in the mutations table (ie if clustering snp distances)",
    )
    parser.add_argument(
        "--dilution_factor", type=int, default=2, help="Dilution factor (default: 2)."
    )
    parser.add_argument(
        "--censored",
        action="store_true",
        help="Indicates if the data is censored (default: False).",
    )
    parser.add_argument(
        "--tail_dilutions",
        type=int,
        default=1,
        help="Tail dilutions for uncensored data (default: 1).",
    )
    parser.add_argument(
        "--frs",
        type=float,
        default=None,
        help="Fraction Read Support threshold (default: None).",
    )
    parser.add_argument(
        "--ecoff",
        type=float,
        default=None,
        help="Epidemiological cutoff value for classification. If None, it will be calculated",
    )
    parser.add_argument(
        "--b_bounds",
        nargs=2,
        type=float,
        default=(None, None),
        help="Bounds for beta coefficients.",
    )
    parser.add_argument(
        "--u_bounds",
        nargs=2,
        type=float,
        default=(None, None),
        help="Bounds for random effects.",
    )
    parser.add_argument(
        "--s_bounds",
        nargs=2,
        type=float,
        default=(None, None),
        help="Bounds for sigma.",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.95,
        help="Significance level for statistical testing (default: 0.95).",
    )
    parser.add_argument(
        "--fixed_effects",
        nargs="+",
        help="List of fixed effect column names (default: None).",
    )
    parser.add_argument(
        "--random_effects",
        action="store_true",
        help="Whether to perform SNP clustering and include as a random effect (default True)",
    )
    parser.add_argument(
        "--cluster_distance",
        type=float,
        default=1,
        help="Clustering distance threshold (default: 1).",
    )
    parser.add_argument(
        "--outfile", type=str, required=False, help="Path to save the output JSON file."
    )
    parser.add_argument(
        "--options",
        type=dict,
        default=None,
        help="Scipy minimise's ptimization options for the regression fitting.",
    )
    parser.add_argument(
        "--L2_penalties",
        type=dict,
        default=None,
        help="Regularization penalties for fixed and random effects",
    )
    parser.add_argument(
        "--to_json",
        action="store_true",
        help="Flag to trigger exporting the catalogue to JSON format.",
    )
    parser.add_argument(
        "--to_piezo",
        action="store_true",
        help="Flag to export catalogue to Piezo format.",
    )
    parser.add_argument(
        "--genbank_ref", type=str, help="GenBank reference for the catalogue."
    )
    parser.add_argument("--catalogue_name", type=str, help="Name of the catalogue.")
    parser.add_argument("--version", type=str, help="Version of the catalogue.")
    parser.add_argument("--drug", type=str, help="Drug associated with the mutations.")
    parser.add_argument("--wildcards", type=str, help="JSON file with wildcard rules.")
    parser.add_argument(
        "--grammar", type=str, default="GARC1", help="Grammar used in the catalogue."
    )
    parser.add_argument(
        "--values",
        type=str,
        default="RUS",
        help="Values used for predictions in the catalogue.",
    )
    parser.add_argument(
        "--for_piezo",
        action="store_true",
        help="If not planning to use piezo, set to False to avoid placeholder rows being added",
    )

    return parser


def main_regression_builder(args):
    """
    Main function to build the regression-based mutation catalogue and handle CLI options.
    """
    from catomatic.RegressionCatalogue import RegressionBuilder

    # Instantiate RegressionBuilder and build the catalogue
    builder = RegressionBuilder(
        samples=args.samples,
        mutations=args.mutations,
        genes=args.genes,
        dilution_factor=args.dilution_factor,
        censored=args.censored,
        tail_dilutions=args.tail_dilutions,
        frs=args.frs,
    )

    builder.build(
        b_bounds=args.b_bounds,
        u_bounds=args.u_bounds,
        s_bounds=args.s_bounds,
        options=args.options,
        L2_penalties=args.L2_penalties,
        ecoff=args.ecoff,
        p=args.p,
        fixed_effects=args.fixed_effects,
        random_effects=args.random_effects,
        cluster_distance=args.cluster_distance,
    )

    # Handle JSON export
    if args.to_json:
        main_json_exporter(builder, args)

    # Handle Piezo export if enabled
    if args.to_piezo:
        main_piezo_exporter(builder, args)


def main_piezo_exporter(builder, args):
    if not all(
        [
            args.genbank_ref,
            args.catalogue_name,
            args.version,
            args.drug,
            args.wildcards,
            args.outfile,
        ]
    ):
        print("Missing required arguments for Piezo export.")
        exit(1)
    builder.to_piezo(
        genbank_ref=args.genbank_ref,
        catalogue_name=args.catalogue_name,
        version=args.version,
        drug=args.drug,
        wildcards=args.wildcards,
        outfile=args.outfile,
        grammar=args.grammar,
        values=args.values,
        for_piezo=args.for_piezo,
    )
    print("Catalogue exported to Piezo format.")


def main_json_exporter(builder, args):
    if not args.outfile:
        print("Please specify an output file with --outfile when using --to_json")
        exit(1)
    builder.to_json(args.outfile)
    print(f"Catalogue exported to {args.outfile}")
