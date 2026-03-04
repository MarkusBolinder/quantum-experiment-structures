"""Collection of helpful functions and classes."""

import argparse


class ArgparseFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawTextHelpFormatter,
):
    """Amalgamation of argparse formatting classes."""
