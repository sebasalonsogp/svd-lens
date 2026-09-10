import numpy as np

from svd_lab.samples import SampleImage, default_sample


def test_default_sample_is_ready_for_the_svd_pipeline() -> None:
    sample = default_sample()

    assert isinstance(sample, SampleImage)
    assert sample.name == "Geometric study"
    assert sample.description
    assert sample.matrix.shape == (180, 240)
    assert sample.matrix.dtype == np.float64
    assert np.isfinite(sample.matrix).all()
    assert sample.matrix.min() >= 0.0
    assert sample.matrix.max() <= 1.0
    assert np.unique(sample.matrix).size > 10
    assert not sample.matrix.flags.writeable


def test_default_sample_is_deterministic() -> None:
    first = default_sample()
    second = default_sample()

    np.testing.assert_array_equal(first.matrix, second.matrix)
