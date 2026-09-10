import numpy as np
import pytest
from plotly.graph_objects import Figure

from svd_lab.plots import singular_value_figure
from svd_lab.svd import decompose


def test_singular_value_figure_highlights_retained_components() -> None:
    result = decompose(np.diag([4.0, 2.0, 1.0]))

    figure = singular_value_figure(result, rank=2)

    assert isinstance(figure, Figure)
    assert figure.layout.title.text == "Singular-value spectrum"
    assert figure.layout.xaxis.title.text == "Component"
    assert figure.layout.yaxis.title.text == "Relative magnitude"
    assert tuple(figure.data[0].x) == (1, 2, 3)
    assert tuple(figure.data[0].y) == pytest.approx((1.0, 0.5, 0.25))
    assert figure.data[0].name == "All components"
    assert figure.data[0].line.color == "#8DA39B"
    assert tuple(figure.data[1].x) == (1, 2)
    assert tuple(figure.data[1].y) == pytest.approx((1.0, 0.5))
    assert figure.data[1].name == "Retained at rank 2"
    assert figure.data[1].line.color == "#2EC4A6"


def test_singular_value_figure_handles_zero_matrix() -> None:
    result = decompose(np.zeros((3, 2)))

    figure = singular_value_figure(result, rank=1)

    np.testing.assert_array_equal(figure.data[0].y, [0.0, 0.0])


@pytest.mark.parametrize("rank", [0, 4])
def test_singular_value_figure_rejects_invalid_rank(rank: int) -> None:
    result = decompose(np.eye(3))

    with pytest.raises(ValueError, match="rank must be between"):
        singular_value_figure(result, rank)
