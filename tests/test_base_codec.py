"""Tests for the BaseEncoder / BaseDecoder abstract base classes."""

from __future__ import annotations

from abc import ABC

import pytest

from ccsds_codec.api import (
    BaseDecoder,
    BaseEncoder,
    ConvCodec,
    Randomizer,
    RSCodec,
    TurboCodec,
)
from ccsds_codec.config import ConvConfig, RSConfig, TurboConfig


def test_base_encoder_exists():
    """BaseEncoder must be exported from the API package."""
    assert BaseEncoder is not None


def test_base_decoder_exists():
    """BaseDecoder must be exported from the API package."""
    assert BaseDecoder is not None


def test_base_encoder_is_abstract():
    """BaseEncoder must be an abstract base class."""
    assert issubclass(BaseEncoder, ABC)
    with pytest.raises(TypeError):
        BaseEncoder()  # type: ignore[abstract]


def test_base_decoder_is_abstract():
    """BaseDecoder must be an abstract base class."""
    assert issubclass(BaseDecoder, ABC)
    with pytest.raises(TypeError):
        BaseDecoder()  # type: ignore[abstract]


def test_rs_codec_inherits_base_classes():
    """RSCodec must inherit from both BaseEncoder and BaseDecoder."""
    assert issubclass(RSCodec, BaseEncoder)
    assert issubclass(RSCodec, BaseDecoder)
    codec = RSCodec(RSConfig(depth=1))
    assert isinstance(codec, BaseEncoder)
    assert isinstance(codec, BaseDecoder)


def test_conv_codec_inherits_base_classes():
    """ConvCodec must inherit from both BaseEncoder and BaseDecoder."""
    assert issubclass(ConvCodec, BaseEncoder)
    assert issubclass(ConvCodec, BaseDecoder)
    codec = ConvCodec(ConvConfig(rate="1/2"))
    assert isinstance(codec, BaseEncoder)
    assert isinstance(codec, BaseDecoder)


def test_turbo_codec_inherits_base_classes():
    """TurboCodec must inherit from both BaseEncoder and BaseDecoder."""
    assert issubclass(TurboCodec, BaseEncoder)
    assert issubclass(TurboCodec, BaseDecoder)
    codec = TurboCodec(TurboConfig(rate="1/3"))
    assert isinstance(codec, BaseEncoder)
    assert isinstance(codec, BaseDecoder)


def test_randomizer_not_part_of_hierarchy():
    """Randomizer is a stateless utility wrapper, not a codec."""
    assert not issubclass(Randomizer, BaseEncoder)
    assert not issubclass(Randomizer, BaseDecoder)


def test_codec_roundtrip_through_base_interface():
    """Codecs must remain usable through the base Encoder/Decoder interface."""
    encoders: list[BaseEncoder] = [
        RSCodec(RSConfig(depth=1)),
        ConvCodec(ConvConfig(rate="1/2")),
        TurboCodec(TurboConfig(rate="1/3")),
    ]
    decoders: list[BaseDecoder] = [
        RSCodec(RSConfig(depth=1)),
        ConvCodec(ConvConfig(rate="1/2")),
        TurboCodec(TurboConfig(rate="1/3")),
    ]

    # RSCodec zero-pads inputs that are not a multiple of RS_K bytes.
    payload_bits = [0, 1, 0, 1, 1, 0, 1, 0] * 223
    rs_payload = bytes(payload_bits)

    encoder_inputs = [
        rs_payload,
        payload_bits,
        payload_bits,
    ]

    for enc, dec, data in zip(encoders, decoders, encoder_inputs):
        encoded = enc.encode(data)
        decoded = dec.decode(encoded)
        if isinstance(dec, RSCodec):
            assert decoded == data
        else:
            assert list(decoded) == list(data)
