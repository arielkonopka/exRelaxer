"""python -m exrelaxer.nntest: the Python experiment runner (see exrelaxer.harness)."""
import sys

from .harness import main

if __name__ == "__main__":
    sys.exit(main())
