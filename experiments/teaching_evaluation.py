"""Teaching evaluation script.

This file serves as a stub for the teaching evaluation experiment referenced in
`virtual-earth-language`. At present, the full implementation has not been
released publicly. The script prints a message explaining that it is a
placeholder and invites contributors to implement the evaluation of teaching
protocols using the project's utilities.
"""

import sys
from typing import Optional


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for the teaching evaluation.

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
    print(
        "[teaching_evaluation] This script is a stub. The teaching evaluation \n"
        "experiment described in the README has not yet been implemented.\n"
        "To develop your own evaluation, import the models and evaluation\n"
        "utilities from the src/ package and implement the appropriate logic."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
