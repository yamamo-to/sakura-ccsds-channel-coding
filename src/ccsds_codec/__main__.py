"""CLI entry point for the ccsds_codec package (``python -m ccsds_codec``).

All argument parsing and dispatch logic lives in :mod:`ccsds_codec.cli`.
"""

from .cli import main

if __name__ == "__main__":
    main()
