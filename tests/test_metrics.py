import pytest

from cova.dnn import metrics


@pytest.mark.parametrize(
    ("bb1", "bb2", "overlap"),
    [
        [[0, 0, 100, 100], [0, 0, 100, 100], 1.0],
        [[0, 0, 50, 50], [0, 0, 100, 100], 1],
        [[0, 0, 100, 100], [0, 0, 50, 50], 0.25],
        [[0, 0, 100, 100], [0, 0, 100, 50], 0.5],
        [[0, 0, 50, 50], [50, 50, 100, 100], 0],
        [[100, 100, 200, 200], [150, 150, 160, 160], 0.01],
    ],
)
def test_get_overlap(bb1, bb2, overlap):
    assert metrics.get_overlap(bb1, bb2) == overlap


@pytest.mark.parametrize(
    ("bb1", "bb2", "iou"),
    [
        [[0, 0, 100, 100], [0, 0, 100, 100], 1.0],
        [[0, 0, 50, 50], [0, 0, 100, 100], 0.25],
        [[0, 0, 50, 50], [50, 50, 100, 100], 0],
    ],
)
def test_get_iou(bb1, bb2, iou):
    assert metrics.get_iou(bb1, bb2)[0] == iou


@pytest.mark.parametrize(("bb1", "bb2"), [["1", []], [[0, 0, 10, 10], [0, 10, 10, 0]]])
def test_get_iou_assert(bb1, bb2):
    with pytest.raises(AssertionError):
        metrics.get_iou(bb1, bb2)
