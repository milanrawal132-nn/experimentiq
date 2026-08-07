"""Data loading, validation and persistence.

Import directly from the submodule, e.g.::

    from src.data.load import load_processed, make_comparison_frame

The package deliberately re-exports nothing: eagerly importing `load` here
makes `python -m src.data.load` emit a double-import RuntimeWarning.
"""
