__all__ = [
    "VAR_model",
    "LOF",
    "DCdetector",
    "AnomalyTransformer",
    "ModernTCN",
    "DualTF",
    "TFAD",
    "CACAM",
    "CACAM_FFT_mix",
]

import importlib


_MODEL_IMPORTS = {
    "LOF": ("ts_benchmark.baselines.self_impl.LOF.lof", "LOF"),
    "VAR_model": ("ts_benchmark.baselines.self_impl.VAR.VAR", "VAR_model"),
    "DCdetector": (
        "ts_benchmark.baselines.self_impl.DCdetector.DCdetector",
        "DCdetector",
    ),
    "AnomalyTransformer": (
        "ts_benchmark.baselines.self_impl.Anomaly_trans.AnomalyTransformer",
        "AnomalyTransformer",
    ),
    "ModernTCN": (
        "ts_benchmark.baselines.self_impl.ModernTCN.ModernTCN",
        "ModernTCN",
    ),
    "DualTF": ("ts_benchmark.baselines.self_impl.DualTF.DualTF", "DualTF"),
    "TFAD": ("ts_benchmark.baselines.self_impl.TFAD.TFAD", "TFAD"),
    "CACAM": ("ts_benchmark.baselines.self_impl.CACAM.CACAM", "CACAM"),
    "CACAM_FFT_mix": ("ts_benchmark.baselines.self_impl.CACAM_FFT_mix.CACAM", "CACAM"),
}


def __getattr__(name):
    if name not in _MODEL_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _MODEL_IMPORTS[name]
    module = importlib.import_module(module_name)
    attr = getattr(module, attr_name)
    globals()[name] = attr
    return attr
