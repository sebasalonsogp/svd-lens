import numpy as np
import pytest
from plotly.graph_objects import Figure

from svd_lab.plots import singular_value_figure
from svd_lab.svd import decompose


def test_singular_value_figure_highlights_retained_components() -> None:
    result = decompose(np.diag([4.0, 2.0, 1.0]))

    figure = singular_value_figure(result, rank=2)

    assert isinstance(figure, Figure)
    assert figure.layout.title.text == "Pattern strength, strongest to weakest"
    assert figure.layout.xaxis.title.text == "Pattern number — strongest to weakest"
    assert figure.layout.yaxis.title.text == "Strength vs. strongest pattern"
    assert tuple(figure.layout.xaxis.range) == (0.5, 3.5)
    assert tuple(figure.layout.yaxis.range) == (0.0, 1.05)
    assert tuple(figure.data[0].x) == (1, 2, 3)
    assert tuple(figure.data[0].y) == pytest.approx((1.0, 0.5, 0.25))
    assert figure.data[0].name == "All available patterns"
    assert figure.data[0].line.color == "#8DA39B"
    assert tuple(figure.data[1].x) == (1, 2)
    assert tuple(figure.data[1].y) == pytest.approx((1.0, 0.5))
    assert figure.data[1].name == "Kept patterns — rank 2"
    assert figure.data[1].line.color == "#2EC4A6"
    assert len(figure.layout.shapes) == 2
    assert figure.layout.shapes[0].type == "rect"
    assert figure.layout.shapes[0].x0 == 0.5
    assert figure.layout.shapes[0].x1 == 2.5
    assert figure.layout.shapes[1].type == "line"
    assert figure.layout.shapes[1].x0 == 2.5
    assert figure.layout.annotations[0].text == "Rank 2 cutoff"


def test_singular_value_figure_handles_zero_matrix() -> None:
    result = decompose(np.zeros((3, 2)))

    figure = singular_value_figure(result, rank=1)

    np.testing.assert_array_equal(figure.data[0].y, [0.0, 0.0])


@pytest.mark.parametrize("rank", [0, 4])
def test_singular_value_figure_rejects_invalid_rank(rank: int) -> None:
    result = decompose(np.eye(3))

    with pytest.raises(ValueError, match="rank must be between"):
        singular_value_figure(result, rank)
