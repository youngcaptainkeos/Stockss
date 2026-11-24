import argparse
from .common import ROOT


def main():
    parser = argparse.ArgumentParser(
        description="NSE Research Lab — historical paper simulation only"
    )
    parser.add_argument(
        "command",
        choices=["ingest", "develop", "final-test", "report", "serve", "verify"],
    )
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.command == "ingest":
        from .data import ingest

        ingest()
    elif args.command == "develop":
        from .experiment import develop

        develop()
    elif args.command == "final-test":
        from .experiment import final_test

        final_test()
    elif args.command == "report":
        from .report import generate_report

        generate_report()
    elif args.command == "serve":
        from .server import serve

        serve(args.port)
    elif args.command == "verify":
        from .verification import verify

        verify()


if __name__ == "__main__":
    main()
