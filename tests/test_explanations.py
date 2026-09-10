import pytest

from svd_lab.explanations import rank_presets, reconstruction_insight
from svd_lab.svd import ApproximationMetrics


def metrics(
    *,
    rank: int,
    retained_energy: float | None,
    relative_error: float = 0.2,
    representation_ratio: float = 2.0,
) -> ApproximationMetrics:
    return ApproximationMetrics(
        rank=rank,
        retained_energy=retained_energy,
        relative_error=relative_error,
        original_value_count=100,
        factorized_value_count=50,
        representation_ratio=representation_ratio,
    )


def test_rank_presets_cover_structure_balance_and_detail() -> None:
    presets = rank_presets(180)

    assert [(preset.label, preset.rank) for preset in presets] == [
        ("Structure", 1),
        ("Balanced", 12),
        ("Detail", 63),
    ]


def test_rank_presets_remove_duplicate_ranks_for_small_images() -> None:
    assert [preset.rank for preset in rank_presets(2)] == [1, 2]


def test_rank_presets_reject_an_invalid_maximum() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        rank_presets(0)


@pytest.mark.parametrize(
    ("current_metrics", "expected_title"),
    [
        (metrics(rank=1, retained_energy=0.45), "The dominant pattern"),
        (metrics(rank=8, retained_energy=0.85), "More detail is waiting"),
        (metrics(rank=12, retained_energy=0.97), "Structure does most of the work"),
        (metrics(rank=20, retained_energy=0.995), "Almost all energy is retained"),
        (
            metrics(rank=50, retained_energy=0.999, representation_ratio=0.8),
            "Detail now costs more values",
        ),
        (metrics(rank=1, retained_energy=None, relative_error=0.0), "No variation to recover"),
    ],
)
def test_reconstruction_insight_names_the_current_tradeoff(
    current_metrics: ApproximationMetrics,
    expected_title: str,
) -> None:
    insight = reconstruction_insight(current_metrics)

    assert insight.title == expected_title
    assert insight.body
