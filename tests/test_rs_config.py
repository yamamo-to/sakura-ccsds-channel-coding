"""Tests for RSConfig interleaving depth configuration."""

import os

import pytest

from ccsds_codec import RSCodec, RSConfig
from ccsds_codec.rs import encode as rs_encode


def test_default_depth():
    cfg = RSConfig()
    assert cfg.depth == 1
    # equality with another default instance
    assert cfg == RSConfig()
    # hashability
    assert isinstance(hash(cfg), int)


def test_valid_depth_boundaries():
    assert RSConfig(depth=1).depth == 1
    assert RSConfig(depth=5).depth == 5


def test_invalid_depth_low():
    with pytest.raises(ValueError, match="depth must be in 1..5"):
        RSConfig(depth=0)


def test_invalid_depth_high():
    with pytest.raises(ValueError, match="depth must be in 1..5"):
        RSConfig(depth=6)


def test_rs_codec_roundtrip():
    c = RSCodec(RSConfig(depth=3))
    data = os.urandom(1000)
    encoded = c.encode(data)
    decoded = c.decode(encoded)
    assert decoded[: len(data)] == data


def test_rs_codec_default_encode_matches_rs_encode():
    c = RSCodec()
    data = os.urandom(1000)
    assert c.encode(data) == rs_encode(data)


def test_rs_codec_different_depths():
    data = os.urandom(800)  # length >= 600
    enc1 = RSCodec(RSConfig(depth=1)).encode(data)
    enc2 = RSCodec(RSConfig(depth=2)).encode(data)
    assert enc1 != enc2


def test_rs_codec_default_config():
    c = RSCodec()
    assert isinstance(c.config, RSConfig)
    assert c.config.depth == 1


def test_rs_codec_invalid_depth():
    with pytest.raises(ValueError):
        RSCodec(RSConfig(depth=0))
