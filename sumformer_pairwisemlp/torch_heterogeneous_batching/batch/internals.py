import torch
from torch import Tensor

class BatchError(ValueError):
    """Raised when there is a mismatch in batching"""

class SingletonMeta(type):
    """Metaclass for singletons
    stolen from https://stackoverflow.com/a/6798042/4948719 
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class OrderIndependent(metaclass=SingletonMeta):
    """Simple indicator that a property is order independent"""

def ptr_from_sizes(size: Tensor) -> Tensor:
    """returns a ptr vector from a sizes vector"""
    ptr = torch.zeros(size.size(0) + 1, dtype=torch.long, device=size.device)
    ptr[1:] = size.cumsum(0)
    return ptr




