"""Tests for the shared reservoir-sampling helper (``utils.misc_utils``).

Covers the buffer-fill invariants (dense occupancy, slot bounds, running-count
bookkeeping) and the defining statistical property of Vitter's Algorithm R:
every item in a stream longer than the buffer is retained with equal
probability ``capacity / stream_length``.
"""

# ruff: noqa: E402

from __future__ import annotations

import os
import random
import sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.misc_utils import reservoir_slots


def test_fills_empty_slots_sequentially_until_full():
    slots, filled, seen = reservoir_slots(3, filled=0, seen=0, capacity=5)
    assert slots == [0, 1, 2]
    assert filled == 3
    assert seen == 3


def test_discards_or_replaces_once_full():
    # Buffer already full (filled == capacity); every further item either
    # replaces an existing slot (index < capacity) or is discarded (-1).
    random.seed(0)
    slots, filled, seen = reservoir_slots(50, filled=4, seen=4, capacity=4)
    assert filled == 4  # stays saturated
    assert seen == 54
    assert all(slot == -1 or 0 <= slot < 4 for slot in slots)
    assert any(slot >= 0 for slot in slots)  # some accepted
    assert any(slot == -1 for slot in slots)  # some rejected


def test_batched_call_matches_sequential_calls():
    # Placing a whole batch at once must equal placing items one-by-one.
    random.seed(123)
    batch_slots, batch_filled, batch_seen = reservoir_slots(
        20, filled=0, seen=0, capacity=4
    )

    random.seed(123)
    seq_slots = []
    filled, seen = 0, 0
    for _ in range(20):
        one, filled, seen = reservoir_slots(1, filled, seen, capacity=4)
        seq_slots.extend(one)

    assert batch_slots == seq_slots
    assert (batch_filled, batch_seen) == (filled, seen)


def test_uniform_retention_probability():
    # Over many independent streams, the final buffer contents should be a
    # near-uniform sample: each of the N stream items appears with probability
    # capacity / N. We track, per stream position, how often it survives.
    capacity = 4
    stream_length = 20
    trials = 20000
    random.seed(2024)

    survival_counts = Counter()
    for _ in range(trials):
        occupant = list(range(capacity))  # slot -> item id currently held
        filled, seen = capacity, capacity  # start already full of items 0..3
        for item_id in range(capacity, stream_length):
            slots, filled, seen = reservoir_slots(1, filled, seen, capacity)
            if slots[0] >= 0:
                occupant[slots[0]] = item_id
        for item_id in occupant:
            survival_counts[item_id] += 1

    expected = trials * capacity / stream_length
    for item_id in range(stream_length):
        # Allow 12% tolerance around the analytical expectation.
        assert (
            abs(survival_counts[item_id] - expected) < 0.12 * expected
        ), f"item {item_id}: {survival_counts[item_id]} vs expected ~{expected:.0f}"


def test_deterministic_under_seed():
    random.seed(7)
    first, _, _ = reservoir_slots(30, filled=4, seen=4, capacity=4)
    random.seed(7)
    second, _, _ = reservoir_slots(30, filled=4, seen=4, capacity=4)
    assert first == second
