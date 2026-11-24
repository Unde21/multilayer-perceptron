"""Terminal colors shared by the programs."""

import sys

GREEN = "\033[32m"
RED = "\033[31m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def color(text: str, *codes: str) -> str:
    """Wrap text in color codes, only if the output is a terminal."""

    if not sys.stdout.isatty():
        return text
    return f"{''.join(codes)}{text}{RESET}"
