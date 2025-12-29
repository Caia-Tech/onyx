#!/usr/bin/env python3
"""
Training CLI wrapper.
"""

try:
    from _bootstrap import add_repo_root
except ImportError:
    from scripts._bootstrap import add_repo_root

add_repo_root()

from onyx.train import main


if __name__ == "__main__":
    main()
