"""Chunked errors, seed statistics, threshold censoring, and amortization."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


def error_metrics(prediction: np.ndarray, reference: np.ndarray,
                  chunk_size: int = 262144) -> Dict[str, float]:
    """Relative L2, max absolute, and mean absolute traveltime errors in seconds."""
    prediction, reference = np.asarray(prediction), np.asarray(reference)
    if prediction.shape != reference.shape or prediction.size == 0:
        raise ValueError("Prediction and reference must have the same nonempty shape")
    predicted, exact = prediction.ravel(), reference.ravel()
    numerator, denominator, absolute_sum, maximum = 0.0, 0.0, 0.0, 0.0
    for start in range(0, prediction.size, chunk_size):
        p = np.asarray(predicted[start:start+chunk_size], dtype=np.float64)
        r = np.asarray(exact[start:start+chunk_size], dtype=np.float64)
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(r)):
            raise ValueError("Nonfinite predictions or references cannot be scored")
        difference = p-r
        numerator += float(np.dot(difference, difference))
        denominator += float(np.dot(r, r))
        absolute = np.abs(difference)
        absolute_sum += float(np.sum(absolute))
        maximum = max(maximum, float(np.max(absolute)))
    relative = math.sqrt(numerator/denominator) if denominator > 0 else (0.0 if numerator == 0 else math.inf)
    return {"relative_l2": relative, "max_abs_error": maximum,
            "mean_abs_error": absolute_sum/prediction.size}


def threshold_key(threshold: float) -> str:
    """Stable column name, e.g. time_to_rel_l2_1e-02_seconds."""
    return "time_to_rel_l2_%s_seconds" % format(threshold, ".0e")


def first_threshold_times(history: Sequence[Mapping[str, Any]],
                          thresholds: Sequence[float]) -> Dict[str, Optional[float]]:
    """First observed threshold crossing; unreached targets remain null.

    These times are right-censored at training completion and depend on the
    validation interval. No interpolation or extrapolation is performed.
    """
    result: Dict[str, Optional[float]] = {threshold_key(t): None for t in thresholds}
    for row in history:
        error = row.get("validation_relative_l2")
        if error is None or not np.isfinite(error):
            continue
        for threshold in thresholds:
            key = threshold_key(threshold)
            if result[key] is None and error <= threshold:
                result[key] = float(row["elapsed_seconds"])
    return result


def aggregate_records(records: Iterable[Mapping[str, Any]], group_keys: Sequence[str],
                      metric_keys: Sequence[str]) -> List[Dict[str, Any]]:
    """Sample mean/std (ddof=1), with seed and achieved-target counts.

    Missing/censored values are excluded from their own statistic and their
    count is reported. A singleton has std=null, rather than an invented zero.
    """
    groups: Dict[tuple, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(record.get(key) for key in group_keys)].append(record)
    aggregated = []
    for identity, rows in groups.items():
        output = dict(zip(group_keys, identity))
        output["run_count"] = len(rows)
        output["seed_count"] = len({r["seed"] for r in rows if "seed" in r})
        output["three_seed_requirement_met"] = output["seed_count"] >= 3
        for key in metric_keys:
            values = [float(r[key]) for r in rows
                      if r.get(key) is not None and np.isfinite(r[key])]
            output[key+"_count"] = len(values)
            output[key+"_mean"] = float(np.mean(values)) if values else None
            output[key+"_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        aggregated.append(output)
    return aggregated


def break_even_sources(training_seconds: float, inference_seconds: float,
                       fsm_seconds: float) -> Optional[int]:
    """Smallest integer K for which train + K*infer < K*FSM.

    Return None when inference itself costs as much as or more than FSM.
    Equality does not count as beating FSM, hence floor(ratio)+1.
    """
    values = [training_seconds, inference_seconds, fsm_seconds]
    if not all(np.isfinite(v) and v >= 0 for v in values) or fsm_seconds == 0:
        raise ValueError("Costs must be finite/nonnegative and FSM cost positive")
    saving = fsm_seconds-inference_seconds
    return None if saving <= 0 else math.floor(training_seconds/saving)+1
