import argparse
import sys
from catomatic.cli_module import (
    main_binary_builder,
    main_ecoff_generator,
    main_regression_builder,
    parse_opt_binary_builder,
    parse_opt_ecoff_generator,
    parse_opt_regression_builder,
)

def main():
    parser = argparse.ArgumentParser(description="Catomatic CLI")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available subcommands")

    # Binary Builder Command
    binary_parser = subparsers.add_parser("binary", help="Run binary builder")
    binary_args = parse_opt_binary_builder()
    for action in binary_args._actions:
        if action.option_strings:
            binary_parser.add_argument(*action.option_strings, **vars(action))

    # Ecoff Generator Command
    ecoff_parser = subparsers.add_parser("ecoff", help="Run ecoff generator")
    ecoff_args = parse_opt_ecoff_generator()
    for action in ecoff_args._actions:
        if action.option_strings:
            ecoff_parser.add_argument(*action.option_strings, **vars(action))

    # Regression Builder Command
    regression_parser = subparsers.add_parser("regression", help="Run regression builder")
    regression_args = parse_opt_regression_builder()
    for action in regression_args._actions:
        if action.option_strings:
            regression_parser.add_argument(*action.option_strings, **vars(action))

    args = parser.parse_args()

    # Dispatch
    if args.command == "binary":
        main_binary_builder()
    elif args.command == "ecoff":
        main_ecoff_generator()
    elif args.command == "regression":
        main_regression_builder()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
