# CCSDS Codec Python Implementation

This repository provides lightweight, pipe‑compatible reference implementations of the
CCSDS forward error correction chain:

* **Randomizer** – CCSDS scrambler/descrambler (polynomial *x⁷ + x⁶ + 1*).
* **Convolutional coder** – rate‑1/2, constraint length 7 (generators 0x79 and 0x5B).
* **Reed‑Solomon** – RS(255,223) over GF(2⁸) (compatible with the CCSDS standard).
* **Turbo coder** – simplified version that concatenates two convolutional encoders
  with a deterministic interleaver.

All tools read binary data from **stdin** and write the result to **stdout**, so they
can be chained with ordinary Unix pipes.

## Installation

```bash
pip install reedsolo  # required for Reed‑Solomon
```

## Usage examples

```bash
# Scramble → Convolutional encode → Reed‑Solomon encode
cat payload.bin |
python -m ccsds_codec.randomizer |
python -m ccsds_codec.conv encode |
python -m ccsds_codec.rs encode > encoded.bin

# Reverse direction (decode)
cat encoded.bin |
python -m ccsds_codec.rs decode |
python -m ccsds_codec.conv decode |
python -m ccsds_codec.randomizer > recovered.bin
```

Turbo (simplified) example:

```bash
cat payload.bin |
python -m ccsds_codec.turbo encode > turbo.bin

cat turbo.bin |
python -m ccsds_codec.turbo decode > recovered.bin
```

The decoder for Turbo currently extracts only the systematic bits; a full MAP
decoder can be added later.

---

Each module can also be invoked directly with a ``mode`` argument:

```bash
python -m ccsds_codec.randomizer           # scrambles stdin
python -m ccsds_codec.conv encode          # convolutional encode
python -m ccsds_codec.rs decode            # Reed‑Solomon decode
python -m ccsds_codec.turbo encode         # Turbo encode
```

The code is deliberately minimal and meant as a teaching/starting point rather
than a production‑grade library.
