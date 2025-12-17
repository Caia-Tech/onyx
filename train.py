"""
Compatibility wrapper for historical imports.

The training CLI/script lives in `onyx_train.py`. Some external snippets and
tests refer to `train.py`, so we re-export the public surface here.
"""

from onyx_train import *  # noqa: F403

