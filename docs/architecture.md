# Software Architecture & Module Interface

## Module Overview
- `src/ccsds_codec/core/`: pure algorithm modules (`bits`, `galois`, `interleaver`, `convolutional`, `reed_solomon`, `turbo`, `randomizer`).
- `src/ccsds_codec/api.py`: high‑level codec classes (`RSCodec`, `ConvCodec`, `TurboCodec`, `Randomizer`).
- `src/ccsds_codec/config.py`: configuration dataclasses for each codec.
- `src/ccsds_codec/cli.py`: unified command‑line interface (`python -m ccsds_codec`).
- `src/ccsds_codec/__main__.py`: entry point for the module.
- Backwards‑compatible shim modules `conv.py`, `rs.py`, `turbo.py`, `randomizer.py`, `utils.py` exposing the core functions.

## Common Interface

High‑level API classes in `api.py` implement `encode` / `decode` methods for each codec. There is no abstract `BaseCodec` in this project; users interact via the concrete classes (`RSCodec`, `ConvCodec`, `TurboCodec`, `Randomizer`) or the functional core modules under `core/`.


