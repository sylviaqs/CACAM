# -*- coding: utf-8 -*-
from ts_benchmark.evaluation.strategy.fixed_forecast import FixedForecast
from ts_benchmark.evaluation.strategy.anomaly_detect import (
    AllDetectBoth,
    AllDetectLabel,
    AllDetectScore,
    FixedDetectBoth,
    FixedDetectLabel,
    FixedDetectScore,
    UnFixedDetectBoth,
    UnFixedDetectLabel,
    UnFixedDetectScore,
)
from ts_benchmark.evaluation.strategy.rolling_forecast import RollingForecast

STRATEGY = {
    "fixed_forecast": FixedForecast,
    "rolling_forecast": RollingForecast,
    "fixed_detect_score": FixedDetectScore,
    "fixed_detect_label": FixedDetectLabel,
    "fixed_detect_both": FixedDetectBoth,
    "unfixed_detect_score": UnFixedDetectScore,
    "unfixed_detect_label": UnFixedDetectLabel,
    "unfixed_detect_both": UnFixedDetectBoth,
    "all_detect_score": AllDetectScore,
    "all_detect_label": AllDetectLabel,
    "all_detect_both": AllDetectBoth,
}
