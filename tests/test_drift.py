import numpy as np
import pandas as pd

from src.drift import (
    AMOUNT_MULTIPLIER,
    N_BATCHES,
    SHIFT_START,
    SHIFT_STD_MULTIPLIER,
    build_batches,
    inject_shift,
    psi,
)


def make_stream(rows=600):
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "Time": np.arange(rows),
            "V1": rng.normal(size=rows),
            "V2": rng.normal(size=rows),
            "Amount": rng.uniform(1, 100, size=rows),
            "Class": (rng.uniform(size=rows) < 0.05).astype(int),
        }
    )


def test_psi_same_distribution_is_small():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    b = rng.normal(size=5000)
    assert psi(a, b) < 0.05


def test_psi_shifted_distribution_is_large():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    b = rng.normal(loc=2.0, size=5000)
    assert psi(a, b) > 0.2


def test_psi_is_never_negative():
    rng = np.random.default_rng(2)
    a = rng.normal(size=1000)
    b = rng.normal(scale=1.5, size=1000)
    assert psi(a, b) >= 0


def test_inject_shift_changes_only_expected_columns():
    X = make_stream().drop(columns=["Time", "Class"])
    std = X.std()
    shifted = inject_shift(X, ["V1"], std)
    assert np.allclose(shifted["V1"], X["V1"] + SHIFT_STD_MULTIPLIER * std["V1"])
    assert np.allclose(shifted["V2"], X["V2"])
    assert np.allclose(shifted["Amount"], X["Amount"] * AMOUNT_MULTIPLIER)


def test_build_batches_natural_leaves_data_unchanged():
    stream = make_stream()
    std = stream[["V1", "V2", "Amount"]].std()
    batches = build_batches(stream, "natural", ["V1"], std)
    assert len(batches) == N_BATCHES
    for number, X, y in batches:
        assert np.allclose(X["Amount"].to_numpy(), stream.loc[X.index, "Amount"].to_numpy())
        assert len(X) == len(y)


def test_build_batches_shift_only_after_start():
    stream = make_stream()
    std = stream[["V1", "V2", "Amount"]].std()
    batches = build_batches(stream, "shift", ["V1"], std)
    for number, X, y in batches:
        original = stream.loc[X.index, "Amount"].to_numpy()
        if number > SHIFT_START:
            assert np.allclose(X["Amount"].to_numpy(), original * AMOUNT_MULTIPLIER)
        else:
            assert np.allclose(X["Amount"].to_numpy(), original)
