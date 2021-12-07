"""Utilities for agents."""
import sys


def debug(*args, **kwargs):
    """Same as print but goes to stderr."""
    print(*args, **kwargs, file=sys.stderr)
