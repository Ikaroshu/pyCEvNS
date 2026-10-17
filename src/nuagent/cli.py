import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="nuagent",
        description="Compute CEvNS event spectra for new neutrino interactions and compare them with the Standard Model.",
    )
    parser.parse_args()
    parser.print_help()
