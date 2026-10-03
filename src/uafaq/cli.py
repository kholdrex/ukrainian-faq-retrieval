import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="uafaq")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("data", "run", "latency", "report"):
        sub.add_parser(name)
    command = parser.parse_args().command
    if command == "data":
        from .data import build
        build()
    elif command == "run":
        from .experiment import run
        run()
    elif command == "latency":
        from .latency import measure
        measure()
    else:
        from .report import report
        report()
