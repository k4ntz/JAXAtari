"""Retired import name — use ``jaxtari``.

This module exists only so ``import jaxatari`` raises a clear ``ImportError``
instead of a bare ``ModuleNotFoundError``. It does **not** load the library.

Canonical usage::

    pip install jaxtari
    import jaxtari

A temporary PyPI distribution also named ``jaxatari`` still re-exports the
library with a deprecation warning; that alias will be removed soon.
"""

raise ImportError(
    "The 'jaxatari' import was renamed to 'jaxtari'. "
    "Use `import jaxtari` after `pip install jaxtari` "
    "(or `pip install -e .` from this repository). "
    "If you still need the old import temporarily, `pip install jaxatari` "
    "installs a short-lived alias package — it will be removed soon."
)
