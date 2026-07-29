# SPDX-FileCopyrightText: 2024-present Thomas Hopf <thomas.hopf@gmail.com>
#
# SPDX-License-Identifier: MIT

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("evedesign")
except PackageNotFoundError:
    # package is not installed, e.g. running from a source checkout
    # without `uv sync` / `pip install -e .` having been run
    __version__ = "unknown"

__all__ = ["__version__"]
