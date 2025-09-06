"""Teaching demonstration script.

This is a placeholder implementation for the teaching demonstration experiment described
in the `virtual-earth-language` project. The original repository references a
`teaching_demo.py` script that has not yet been provided. This stub preserves
the command‑line entry point and documents its intended purpose.

Usage: Run this script with Python to see a message explaining that the
implementation is missing. You can extend this file to implement your own
teaching protocol, leveraging the models and utilities defined in the
`src/` package.
"""

import sys
from typing import Optional


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for the teaching demonstration.

    Parameters
    ----------
    argv : list[str] | None
        Optional list of command‑line arguments. If None, ``sys.argv`` will be
        used instead.

    Returns
    -------
    int
        Exit status code. 0 indicates success.
    """
    if argv is None:
        argv = sys.argv[1:]
    # Placeholder implementation
    print(
        "[teaching_demo] This script is a stub. The teaching demonstration \n"
        "experiment described in the README has not yet been implemented.\n"
        "Please refer to the project's documentation or implement your own \n"
        "teaching protocol using the modules in src/."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
