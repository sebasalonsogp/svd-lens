import numpy as np

from svd_lab.samples import SampleImage, available_samples, default_sample


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


def test_sample_catalog_offers_three_distinct_experiments() -> None:
    samples = available_samples()

    assert [sample.name for sample in samples] == [
        "Geometric study",
        "Soft bands",
        "Woven detail",
    ]
    assert len({sample.description for sample in samples}) == 3

    for sample in samples:
        assert sample.matrix.shape == (180, 240)
        assert sample.matrix.dtype == np.float64
        assert np.isfinite(sample.matrix).all()
        assert 0.0 <= sample.matrix.min() <= sample.matrix.max() <= 1.0
        assert not sample.matrix.flags.writeable

    assert not np.array_equal(samples[0].matrix, samples[1].matrix)
    assert not np.array_equal(samples[1].matrix, samples[2].matrix)


def test_default_sample_is_the_first_catalog_entry() -> None:
    assert default_sample() is available_samples()[0]
