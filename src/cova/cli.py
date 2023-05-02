# -*- coding: utf-8 -*-

"""Command line interface for :mod:`cova`.

Why does this file exist, and why not put this in ``__main__``? You might be tempted to import things from ``__main__``
later, but that will cause problems--the code will get executed twice:

- When you run ``python3 -m cova`` python will execute``__main__.py`` as a script.
  That means there won't be any ``cova.__main__`` in ``sys.modules``.
- When you import __main__ it will get executed again (as a module) because
  there's no ``cova.__main__`` in ``sys.modules``.

.. seealso:: https://click.palletsprojects.com/en/7.x/setuptools/#setuptools-integration
"""

import argparse
import logging

from cova.cli_helper import _run

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(
        description="This program runs a COVA pipeline defined in a json-like config file."
    )
    parser.add_argument("config", type=str, help="Path to the configuration file.")

    return parser


def main():
    parser = get_args()
    args = parser.parse_args()
    _run(args.config)
