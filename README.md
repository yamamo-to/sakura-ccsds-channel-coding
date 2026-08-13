# CCSDS Codec Python Implementation

This repository provides lightweight, pipe‑compatible implementations of the
CCSDS forward error correction chain (CCSDS 131.0‑B‑4):

* **Randomizer** – CCSDS pseudo‑randomizer (polynomial *x⁸ + x⁷ + x⁵ + x³ + 1*).
* **Convolutional coder** – rate‑1/2, constraint length 7 (generators
  171₈ / 133₈, with the second output inverted on the channel), plus
  punctured rates
  2/3, 3/4, 5/6 and 7/8.  Decoding uses the Viterbi algorithm.
* **Reed–Solomon** – RS(255,223) over GF(2⁸), interleaving depth 1…5.
* **Turbo coder** – RSC constituent codes, QPP interleaver, rates 1/2, 1/3, 1/4, 1/6,
  decoded iteratively with the Log‑MAP algorithm.

All tools read binary data from **stdin** and write the result to **stdout**, so
they can be chained with ordinary Unix pipes.

## Installation

```bash
pip install numpy numba   # runtime dependencies
pip install reedsolo      # optional: native Reed–Solomon backend
```

## Usage examples

The unified CLI entry point is `python -m ccsds_codec`:

```bash
# Scramble → Convolutional encode → Reed–Solomon encode
cat payload.bin |
python -m ccsds_codec rand |
python -m ccsds_codec conv-enc |
python -m ccsds_codec rs-enc > encoded.bin

# Reverse direction (decode)
cat encoded.bin |
python -m ccsds_codec rs-dec |
python -m ccsds_codec conv-dec |
python -m ccsds_codec rand > recovered.bin
```

Punctured convolutional rates and Turbo rates are selected with `--rate`:

```bash
python -m ccsds_codec conv-enc --rate 7/8 < payload.bin > encoded.bin
python -m ccsds_codec conv-dec --rate 7/8 < encoded.bin > recovered.bin

python -m ccsds_codec turbo-enc --rate 1/6 < payload.bin > turbo.bin
python -m ccsds_codec turbo-dec --rate 1/6 < turbo.bin > recovered.bin
```

Reed–Solomon interleaving depth (1…5, default 1) is selected with `--depth`:

```bash
python -m ccsds_codec rs-enc --depth 5 < payload.bin > encoded.bin
python -m ccsds_codec rs-dec --depth 5 < encoded.bin > recovered.bin
```

## Python API

The high‑level API lives in `ccsds_codec.api` and is configured with value
objects from `ccsds_codec.config`:

```python
from ccsds_codec import ConvCodec, ConvConfig, RSCodec, RSConfig, TurboCodec, TurboConfig

conv = ConvCodec(ConvConfig(rate="3/4"))
encoded = conv.encode(data)          # list[int] bits in, bits out
decoded = conv.decode(encoded)

rs = RSCodec(RSConfig(depth=5))
encoded = rs.encode(data)            # bytes in, bytes out
decoded = rs.decode(encoded)         # round-trips when len(data) % (223 * depth) == 0

turbo = TurboCodec(TurboConfig(rate="1/3"))
encoded = turbo.encode(data, iterations=5)
decoded = turbo.decode(encoded, iterations=10)
```

The raw bit‑level primitives are available under `ccsds_codec.core`
(`convolutional`, `reed_solomon`, `turbo`, `randomizer`, `bits`).  The
`ccsds_codec.conv` / `rs` / `turbo` / `randomizer` / `utils` modules remain as
backwards‑compatible shims.

## Source layout

```
src/ccsds_codec/
├── core/            # pure algorithm modules (bits, galois, interleaver,
│                    #   convolutional, reed_solomon, turbo, randomizer)
├── api.py           # high-level codec classes (ConvCodec, RSCodec, ...)
├── config.py        # configuration value objects (ConvConfig, TurboConfig)
├── cli.py           # unified CLI dispatch
├── __main__.py      # python -m ccsds_codec entry point
├── conv.py          # backwards-compatible shims
├── rs.py
├── turbo.py
├── randomizer.py
└── utils.py
```

## Development

```bash
python -m pytest tests        # test suite (golden vectors, BER/FER sims)
ruff check src/ccsds_codec    # lint
python -m scripts.benchmark   # throughput benchmark
```
