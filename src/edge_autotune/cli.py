# -*- coding: utf-8 -*-

"""Command line interface for :mod:`edge_autotune`.

Why does this file exist, and why not put this in ``__main__``? You might be tempted to import things from ``__main__``
later, but that will cause problems--the code will get executed twice:

- When you run ``python3 -m edge_autotune`` python will execute``__main__.py`` as a script.
  That means there won't be any ``edge_autotune.__main__`` in ``sys.modules``.
- When you import __main__ it will get executed again (as a module) because
  there's no ``edge_autotune.__main__`` in ``sys.modules``.

.. seealso:: https://click.palletsprojects.com/en/7.x/setuptools/#setuptools-integration
"""

import argparse
import logging
from typing import Tuple

from edge_autotune.cli_helper import _run


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="This program runs a COVA pipeline defined in a json-like config file."
    )
    parser.add_argument(
        "config", type=str, help="Path to a video or a sequence of image."
    )

    args = parser.parse_args()
    _run(args.config)
