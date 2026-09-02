#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from tracy import perf_metrics_common as mc
from tracy.perf_counter_analysis import COUNTER_TYPE_NAMES, PERF_COUNTER_CSV_HEADERS


class _View:
    """CounterView over a flat {counter_name: value} dict with one shared cycle count."""

    def __init__(self, values, cycles=2000.0):
        self._values = values
        self._cycles = cycles

    def count(self, bank, name):
        return float(self._values.get(name, 0.0))

    def cycles(self, bank):
        return self._cycles if self._values else 0.0

    def has(self, name):
        return name in self._values


def test_every_metric_key_has_a_label_and_a_family_suffix():
    out = mc.compute_metrics(_View({}))
    assert set(out) == set(mc.METRIC_LABELS)
    assert all(k.endswith("_pct") or k.endswith("_ratio") for k in out)
    assert mc.RATIO_KEYS == {k for k in out if k.endswith("_ratio")}


def test_empty_view_yields_none_not_zero():
    out = mc.compute_metrics(_View({}))
    assert all(v is None for v in out.values())


def test_full_view_percentages_stay_bounded():
    names = set(COUNTER_TYPE_NAMES.values()) - {"UNDEF"}
    out = mc.compute_metrics(_View({n: 1000.0 for n in names}))
    for key, value in out.items():
        assert value is not None, key
        if key.endswith("_pct"):
            assert 0.0 <= value <= 100.0, (key, value)


def test_cross_bank_stalls_gate_on_missing_pack_counters():
    # Unpack group captured without the pack group: absent counters read 0, and an ungated
    # complement would report a bogus 100% stall instead of N/A.
    out = mc.compute_metrics(_View({"MATH_INSTRN_AVAILABLE": 1500.0}))
    assert out["math_scoreboard_stall_pct"] is None
    assert out["math_dest_wr_port_stall_pct"] is None


def test_per_engine_packers_gate_on_wormhole_only_counters():
    out = mc.compute_metrics(_View({"PACKER_BUSY": 800.0}))
    assert out["packer0_util_pct"] is None
    assert out["packer1_util_pct"] is None
    assert out["packer2_util_pct"] is None
    assert out["packer3_util_pct"] == 40.0


def test_mean_port_util_averages_only_present_ports():
    view = _View({"L1_0_NOC_RING0_OUTGOING_0": 500.0, "L1_0_NOC_RING0_OUTGOING_1": 1500.0})
    assert mc.mean_port_util(view, "L1", mc.L1_RING0, 2000.0) == 0.5
    assert mc.mean_port_util(view, "L1", mc.L1_EXT_PACK, 2000.0) is None


def test_enum_parser_matches_the_compiled_ordinals():
    # UNDEF anchors ordinal 0 and the table is dense from there.
    assert COUNTER_TYPE_NAMES[0] == "UNDEF"
    assert sorted(COUNTER_TYPE_NAMES) == list(range(len(COUNTER_TYPE_NAMES)))
    enum_names = set(COUNTER_TYPE_NAMES.values())
    assert all(n in enum_names for n in mc.L1_ALL)


def test_csv_headers_cover_every_label_with_four_stats():
    assert len(PERF_COUNTER_CSV_HEADERS) == 4 * len(mc.METRIC_LABELS)
    assert "Avg FPU util on full grid (%)" in PERF_COUNTER_CSV_HEADERS
    assert "Stall Overlap T0 Min (ratio)" in PERF_COUNTER_CSV_HEADERS


def test_tech_report_catalogue_lists_every_metric():
    # The tech report is the single human-readable catalogue; fail when a metric is added to the
    # engine without documenting it (or renamed without updating the doc).
    report = (Path(__file__).resolve().parents[3] / "tech_reports" / "PerfCounters" / "perf-counters.md").read_text()
    for key, label in mc.METRIC_LABELS.items():
        assert f"`{key}`" in report, f"tech report is missing metric key {key}"
        assert label in report, f"tech report is missing metric label {label}"
