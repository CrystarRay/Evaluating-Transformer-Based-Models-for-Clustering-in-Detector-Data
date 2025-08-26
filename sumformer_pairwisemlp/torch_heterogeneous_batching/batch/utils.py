import torch
from torch import Tensor
from ml_lib.misc import all_equal

from .batch import Batch
from .indicator import BatchIndicatorBase, BatchIndicatorProduct
from .internals import BatchError

def batch_product(*batches: Batch)-> tuple[Batch, ...]:
    """Takes a set of batches, and projects them onto the product space

    takes batch_1, ..., batch_n
    
    with each the same batch size N and feature_size F
    
    And n_k^i is the size of the kth set in batch i

    and returns

    batch_1*, ..., batch_n*

    batches over the tensor product of their respective space

    where batch_i*[k][u_1, ... , u_n] = batch_i[k][u_i]

    """


    product_indicator = BatchIndicatorProduct(*(batch.indicator for batch in batches))

    grids: tuple[Tensor, ...] = product_indicator.get_grid()
    datas: list[Tensor] = [batch.data[grid] for batch, grid in zip(batches, grids)]

    return tuple(Batch(data=data, order=None, indicator=product_indicator) 
            for data in datas)


def check_same_batch_size(*indicators, calling=None, none_ok=False): 
    """Check that all given indicators have the same number of batches"""
    batch_sizes = []
    for i in indicators:
        if isinstance(i, BatchIndicatorBase):
            batch_sizes.append(i.get_batch_size())
        elif isinstance(i, Batch):
            batch_sizes.append(i.batch_size)
        elif i is None:
            if not none_ok:
                raise ValueError(f"Got None in {__name__} but none_ok is False")
        else:
            assert isinstance(i, torch.Tensor), f"{__name__} was only expecting Indicators, Batches, and eventually None if none_ok, got {i}"
            batch_sizes.append(i.size(0))
    if all_equal(*batch_sizes): return
    callername: str
    match calling:
        case None:
            callername = ""
        case str(s):
            callername = s
        case _ if hasattr(calling, "__name__"):
            callername = calling.__name
        case _ if hasattr(calling, "__class__"):
            callername = calling.__class__.__name__
        case _:
            callername = repr(calling)
    if callername: callername = callername + " "
    error = callername + f"cannot be called on indicators of different batch sizes, got {batch_sizes}"   
    raise BatchError(error)
