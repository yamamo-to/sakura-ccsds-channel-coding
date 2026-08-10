"""Algorithm core for ccsds_codec.

Pure, framework-free implementations of the CCSDS channel-coding primitives:

* :mod:`bits` – bit/byte conversions.
* :mod:`galois` – GF(2^8) arithmetic (Reed‑Solomon).
* :mod:`interleaver` – CCSDS §6.3g turbo interleaver.
* :mod:`convolutional` – convolutional encoder + Viterbi decoder.
* :mod:`reed_solomon` – RS(255,223) encoder/decoder.
* :mod:`turbo` – Turbo encoder + iterative Log-MAP decoder.
* :mod:`randomizer` – CCSDS pseudo-randomizer.

These modules operate only on plain Python / NumPy data (no I/O); the
high-level API (:mod:`ccsds_codec.api`) and the CLI (:mod:`ccsds_codec.cli`)
build on top of them.
"""
