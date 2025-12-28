#!/usr/bin/env python3
"""
Inference CLI wrapper.
"""

from _bootstrap import add_repo_root

add_repo_root()

from onyx.inference import main


if __name__ == "__main__":
    main()
