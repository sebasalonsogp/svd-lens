"""Plain-language interpretations of the current low-rank approximation."""

from dataclasses import dataclass

from svd_lab.svd import ApproximationMetrics


@dataclass(frozen=True, slots=True)
class RankPreset:
    """A named shortcut to a useful reconstruction rank."""

    label: str
    rank: int


@dataclass(frozen=True, slots=True)
class ReconstructionInsight:
    """A concise interpretation of the current quality-size tradeoff."""

    title: str
    body: str


def reconstruction_caption(rank: int) -> str:
    """Describe how many singular components produced a reconstruction."""
    if rank == 1:
        return "Rebuilt from the leading singular component."
    return f"Rebuilt from the first {rank} singular components."


def rank_presets(max_rank: int) -> tuple[RankPreset, ...]:
    """Return distinct structure, balanced, and detail shortcuts."""
    if isinstance(max_rank, bool) or not isinstance(max_rank, int) or max_rank <= 0:
        raise ValueError("max_rank must be a positive integer")

    candidates = (
        RankPreset("Structure", 1),
        RankPreset("Balanced", min(12, max_rank)),
        RankPreset("Detail", max(1, round(max_rank * 0.35))),
    )
    presets: list[RankPreset] = []
    seen_ranks: set[int] = set()
    for preset in candidates:
        if preset.rank not in seen_ranks:
            presets.append(preset)
            seen_ranks.add(preset.rank)
    return tuple(presets)


def reconstruction_insight(metrics: ApproximationMetrics) -> ReconstructionInsight:
    """Explain the most important property of the current reconstruction."""
    if metrics.retained_energy is None:
        return ReconstructionInsight(
            title="No variation to recover",
            body=(
                "Every singular value is zero, so the processed image is already "
                "reproduced exactly at any available rank."
            ),
        )

    if metrics.representation_ratio <= 1.0:
        return ReconstructionInsight(
            title="Detail now costs more values",
            body=(
                "At this rank, the matrix factors use at least as many scalar values as "
                "the original matrix. The reconstruction can still be accurate, but it "
                "is no longer a compact mathematical representation."
            ),
        )

    if metrics.rank == 1:
        return ReconstructionInsight(
            title="The dominant pattern",
            body=(
                "One singular component captures the strongest row-and-column pattern. "
                "Edges and texture need additional components."
            ),
        )

    retained_energy = metrics.retained_energy
    if retained_energy < 0.90:
        return ReconstructionInsight(
            title="More detail is waiting",
            body=(
                f"These components retain {retained_energy:.1%} of the matrix energy. "
                "Move right to recover more of the structure carried by smaller singular values."
            ),
        )
    if retained_energy < 0.99:
        return ReconstructionInsight(
            title="Structure does most of the work",
            body=(
                f"The selected components retain {retained_energy:.1%} of the energy. "
                "The remaining components mostly refine edges and texture."
            ),
        )
    return ReconstructionInsight(
        title="Almost all energy is retained",
        body=(
            f"The reconstruction retains {retained_energy:.1%} of the matrix energy "
            f"with {metrics.relative_error:.1%} relative error."
        ),
    )


__all__ = [
    "RankPreset",
    "ReconstructionInsight",
    "rank_presets",
    "reconstruction_caption",
    "reconstruction_insight",
]
