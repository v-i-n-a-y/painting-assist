# Copyright 2026 Vinay Williams

"""PyInstaller entry script.

Kept separate from ``painting_assist/app.py`` so the frozen bundle has a stable
top-level script that simply delegates to the package's ``main()``.
"""

import sys

from painting_assist.app import main

if __name__ == "__main__":
    sys.exit(main())
