import numpy as np

from presto.sim.simulator import LinkStats
from presto.telemetry import RAW_FEATURES, Telemetry


def test_window_is_oldest_first_and_has_expected_shape():
    telemetry = Telemetry(2, 3, np.array([2.0, 4.0]), 20.0)
    for value in (1.0, 2.0, 3.0, 4.0):
        telemetry.push(
            LinkStats(
                delay=np.full(2, value * np.array([2.0, 4.0])),
                loss=np.zeros(2),
                queue=np.full(2, value),
                util=np.full(2, value),
                up=np.ones(2, dtype=bool),
            )
        )

    window = telemetry.window()

    assert window.shape == (2, 3, len(RAW_FEATURES))
    np.testing.assert_allclose(window[0, :, 0], [2.0, 3.0, 4.0])


def test_summary_slope_is_positive_for_rising_series():
    telemetry = Telemetry(1, 4, np.array([1.0]), 10.0)
    for value in (1.0, 2.0, 3.0, 4.0):
        telemetry.push(
            LinkStats(
                delay=np.array([value]),
                loss=np.array([0.0]),
                queue=np.array([0.0]),
                util=np.array([0.0]),
                up=np.array([True]),
            )
        )

    summary = telemetry.summary()

    assert summary.shape == (1, len(RAW_FEATURES) * 4 + 1)
    assert summary[0, 10] > 0.0


def test_partial_window_slope_uses_available_history():
    telemetry = Telemetry(1, 5, np.array([1.0]), 10.0)
    for value in (2.0, 4.0):
        telemetry.push(
            LinkStats(
                delay=np.array([value]),
                loss=np.array([0.0]),
                queue=np.array([0.0]),
                util=np.array([0.0]),
                up=np.array([True]),
            )
        )

    assert telemetry.summary()[0, 10] == 2.0


def test_up_flip_count_is_reported():
    telemetry = Telemetry(1, 3, np.array([1.0]), 10.0)
    for up in (True, False, True):
        telemetry.push(
            LinkStats(
                delay=np.array([1.0]),
                loss=np.array([0.0]),
                queue=np.array([0.0]),
                util=np.array([0.0]),
                up=np.array([up]),
            )
        )

    assert telemetry.summary()[0, -1] == 2
