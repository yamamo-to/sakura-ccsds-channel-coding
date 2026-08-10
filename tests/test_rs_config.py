"""Tests for RSConfig interleaving depth configuration."""

import pytest

from ccsds_codec import RSConfig


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
