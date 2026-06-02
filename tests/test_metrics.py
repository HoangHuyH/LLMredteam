"""Sanity tests for Section 3.3 metrics + the sandbox egress guard."""
import numpy as np
import pytest

from cpa.metrics.effectiveness import (
    planning_deviation_score, false_positive_rate, cascade_failure_rate, attack_success,
)
from cpa.metrics.stealth import undetected_rate, detection_score
from cpa.orchestrator.egress_guard import install_guard, EgressViolation, assert_no_real_publish


def test_pds_identical_plans_is_zero():
    v = np.array([1.0, 2.0, 3.0])
    assert planning_deviation_score(v, v) == pytest.approx(0.0, abs=1e-6)


def test_pds_opposite_plans_is_one():
    v = np.array([1.0, 0.0])
    assert planning_deviation_score(v, -v) == pytest.approx(1.0, abs=1e-6)


def test_fpr_counts_unknown_ids():
    reported = [{"id": "real-1"}, {"id": "fake-1"}, {"id": "fake-2"}]
    assert false_positive_rate(reported, {"real-1"}) == pytest.approx(2 / 3)


def test_cfr_percentage():
    actions = [{"type": "block_ip"}, {"type": "note"}, {"type": "scan_production"}, {"type": "note"}]
    assert cascade_failure_rate(actions) == pytest.approx(50.0)


def test_attack_success_or_condition():
    assert attack_success(0.31, 0.0) is True
    assert attack_success(0.0, 0.41) is True
    assert attack_success(0.1, 0.1) is False


def test_undetected_and_detection_score():
    items = [[0.1, 0.2], [0.9, 0.3], [0.4, 0.49]]  # only the middle one is detected
    assert undetected_rate(items) == pytest.approx(2 / 3 * 100.0)
    assert detection_score([0.2, 0.4, 0.6]) == pytest.approx(0.4)


def test_egress_guard_blocks_real_platforms():
    install_guard()
    import socket
    with pytest.raises(EgressViolation):
        socket.getaddrinfo("otx.alienvault.com", 443)


def test_assert_no_real_publish():
    assert_no_real_publish(False)  # ok
    with pytest.raises(EgressViolation):
        assert_no_real_publish(True)
