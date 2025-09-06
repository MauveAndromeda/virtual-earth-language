"""
Entry point for the standalone interpretable earth visualizer.

This script provides a placeholder for launching the interactive Digital Earth visualization outside of the Python package. For now it simply instructs the user to run the Streamlit app located under src/visualization/interactive_earth.py.
"""

import sys


def main() -> None:
    """Provide instructions to launch the interactive earth visualization."""
    print(
        "This is a placeholder for the interpretable earth visualizer. "
        "To explore the Digital Earth interface, run the following command:\n"
        "    streamlit run src/visualization/interactive_earth.py"
    )
    raise SystemExit(0)


if __name__ == "__main__":
    main()
