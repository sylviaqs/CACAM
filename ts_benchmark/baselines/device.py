import torch


def get_torch_device(allow_mps: bool = True) -> torch.device:
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return torch.device("cuda")
    if allow_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
