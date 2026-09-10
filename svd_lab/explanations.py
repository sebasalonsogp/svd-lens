"""Plain-language interpretations of the current low-rank approximation."""


def reconstruction_caption(rank: int) -> str:
    """Describe how many singular components produced a reconstruction."""
    if rank == 1:
        return "Rebuilt from the leading singular component."
    return f"Rebuilt from the first {rank} singular components."


__all__ = ["reconstruction_caption"]
