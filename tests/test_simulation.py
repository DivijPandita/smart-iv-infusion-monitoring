"""Integration tests for the whole simulation layer (truth + load cell + IR sensor together)."""
import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.simulation.generate_data import generate_dataset


@pytest.fixture(scope="module")
def ds():
    return generate_dataset(load_config())


def test_timestamps_are_regular_and_ordered(ds):
    t = ds.truth["timestamp"].to_numpy()
    assert t[0] == 0.0
    assert np.allclose(np.diff(t), 1.0)
    assert (np.diff(ds.load_cell["timestamp"]) > 0).all()
    assert (np.diff(ds.drops["timestamp"]) >= 0).all()


def test_weight_decreases_by_the_bag_volume(ds):
    w = ds.load_cell["measured_weight_g"].to_numpy()
    start, end = np.median(w[:10]), np.median(w[-10:])
    assert 495.0 < start - end < 505.0                                # 500 mL left the bag
    assert (np.diff(ds.truth["true_weight_g"]) <= 1e-9).all()         # truth never increases


def test_drop_generation_is_reasonable(ds):
    expected = 500 * 14.5
    assert 0.9 * expected < len(ds.drops) < 1.05 * expected
    counts = ds.drops["event_type"].value_counts()
    assert counts["real"] > 0.9 * expected
    assert 0 < counts.get("false", 0) < 0.05 * expected


def test_both_sensors_describe_the_same_physics(ds):
    drops_per_ml = len(ds.drops) / 500.0
    assert 13.5 < drops_per_ml < 14.8                                 # about 14.3 as counted


def test_dataset_is_reproducible():
    a, b = generate_dataset(load_config()), generate_dataset(load_config())
    pd.testing.assert_frame_equal(a.load_cell, b.load_cell)
    pd.testing.assert_frame_equal(a.drops, b.drops)