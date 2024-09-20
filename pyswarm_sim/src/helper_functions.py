"""
This module contains helper functions that are used in the main script.
"""

from contextlib import contextmanager
from timeit import default_timer

@contextmanager
def elapsed_timer():
    """
    Context manager to measure the elapsed time.

    Yields:
        callable: time ellapsed since the start of the context manager.
    """
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start