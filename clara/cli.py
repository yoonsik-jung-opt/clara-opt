"""CLARA command-line interface."""

import click


@click.group()
@click.version_option()
def main():
    """CLARA — Classical LP Analysis for Reoptimization and Attribution."""
    pass
