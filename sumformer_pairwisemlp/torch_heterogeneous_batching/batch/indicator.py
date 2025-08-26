from typing import (Self, Any, get_type_hints, get_origin, Sequence, 
                    overload, NoReturn, Iterator, Annotated, TypeAlias, TypeVar,)
import torch
from torch import Tensor
from ml_lib.misc import all_equal

from .internals import OrderIndependent, ptr_from_sizes 

CachedVar = TypeVar("CachedVar", Tensor|None, dict[int|None, Tensor])
Cached: TypeAlias = Annotated[CachedVar, "BatchIndicatorCached"]
CachedOrderIndepTensor: TypeAlias = Annotated[Tensor|None, "BatchIndicatorCached", "OrderIndependent"]

CachedDict: TypeAlias = Cached[dict[int|None, Tensor]]
CachedTensor: TypeAlias = Cached[Tensor|None]


class BatchIndicatorBase():
    """
    Base class for batch indicators

    An indicator contains the size of every element in the batch, 
    and allows for easy access to useful values such as 
    - the ptr vector (offsets of each batch in the data vector)
    - the batch vector (which batch each element belongs to)
    - the diagonal of the matrix (only valid for order 2)
    - the grid (which coordinates in the data vector correspond to which coordinates in the matrix)
    """
    n_nodes: Tensor

    def __init__(self, n_nodes):
        self.n_nodes = n_nodes.to(torch.int32)

    def get_n_nodes(self) -> Tensor:
        return self.n_nodes
    def get_n_edges(self) -> Tensor:
        raise NotImplementedError
    def get_ptr1(self) -> Tensor:
        raise NotImplementedError
    def get_ptr2(self) -> Tensor:
        raise NotImplementedError
    def get_batch1(self) -> Tensor:
        raise NotImplementedError
    def get_batch2(self) -> Tensor:
        raise NotImplementedError
    def get_diagonal(self) -> Tensor:
        raise NotImplementedError
    def get_grid(self) -> tuple[Tensor, ...]:
        raise NotImplementedError
    def get_batch_size(self) -> int:
        return len(self.get_n_nodes())

    def shapes(self, order: int|None) -> Iterator[tuple[int, ...]]:
        raise NotImplementedError

    def grid(self):
        return self.get_grid() #legacy alias

class BatchIndicatorAutoCacheMixin():
    """
    Automatically cache computed values. 
    If a class member "_attr" is annotated with "Cached", 
    when "get_attr" is called, it will check if the value is already computed (in _attr)
    otherwise it will call "_compute_attr" and cache the result.
    """
    def _init_caches(self):
        for x, t in get_type_hints(self.__class__, include_extras=True).items():
            if not x.startswith("_"): continue
            if "BatchIndicatorCached" not in t.__metadata__: continue
            t = t.__origin__
            if get_origin(t) is dict:
                setattr(self, x, dict())
            elif t == (Tensor|None):
                setattr(self, x, None)

    def is_cached(self, attr: str):
        t = get_type_hints(self.__class__, include_extras=True).get(f"_{attr}")
        if t is None: return False
        return "BatchIndicatorCached"  in t.__metadata__

    def is_order_independent(self, attr: str):
        t = get_type_hints(self.__class__, include_extras=True).get(f"_{attr}")
        if t is None: return False
        return "OrderIndependent" in t.__metadata__

    def __getattribute__(self, attr: str):
        get_function = object.__getattribute__(self, "get")
        if attr == "get":
            return get_function
        if not attr.startswith("get_"):
            return object.__getattribute__(self, attr)

        attr_stem = attr[len("get_"):]
        if self.is_cached(attr_stem):
            if self.is_order_independent(attr_stem):
                return lambda: get_function(attr_stem, OrderIndependent())
            return lambda order: get_function(attr_stem, order)

        # shortcuts for order 1 and 2
        if attr.endswith("1") and self.is_cached(attr_stem[:-1]):
            return lambda: get_function(attr_stem[:-1], 1)
        if attr.endswith("_nodes"):
            return lambda: get_function(attr_stem[:-len("_nodes")], 1)
        if attr.endswith("2") and self.is_cached(attr_stem[:-1]):
            return lambda: get_function(attr_stem[:-1], 2)
        if attr.endswith("_edges"):
            return lambda: get_function(attr_stem[:-len("_edges")], 2)

        # if it's not a cached value, just default
        return object.__getattribute__(self, attr)


    def get(self, attr_name: str, order: int|None|OrderIndependent):
        assert isinstance(attr_name, str)
        assert isinstance(order, int) \
                or order is None or order is OrderIndependent()
        if order is OrderIndependent():
            attr = getattr(self, f"_{attr_name}")
            if attr is not None: return attr
            compute_function = getattr(self, f"_compute_{attr_name}")
            ret_value = compute_function()
            setattr(self, f"_{attr_name}", ret_value)
            if isinstance(ret_value, Tensor):
                assert ret_value.device == self.device
            return ret_value
            
        attr_dict: dict[int, Any] = getattr(self, f"_{attr_name}")
        assert isinstance(attr_dict, dict)
        if order in attr_dict:
            return attr_dict[order]
        else:
            compute_function = getattr(self, f"_compute_{attr_name}")
            ret_val = compute_function(order)
            assert ret_val.device == self.device
            attr_dict[order] = ret_val
            return ret_val



class BatchIndicator(BatchIndicatorBase, BatchIndicatorAutoCacheMixin):
    """Represents a batch of arbitrary-ordered tensor data"""

    n_nodes: Tensor
    """n_nodes: (batch_size) long indicating the number of nodes in each batch"""

    _n: CachedDict
    """n[0] = batch_size, n[1] = n_nodes, n[2] = n_edges etc."""

    _ptr: CachedDict
    """ptr[order]: (batch_size + 1) long indicating the offset of each batch in the batch[order] vector (optional, will be recomputed otherwise)
    since it is +1, it includes 0 at the beginning, and the last element at the end
    (ie all the bounds)
    """

    _batch: CachedDict
    """batch[order]: (sum_i n_nodes_i^order) long indicating the batch of each node (optional, will be recomputed otherwise)"""

    _ntotal: CachedDict
    _max_n: CachedDict

    _diagonal: CachedOrderIndepTensor = None
    """diagonal: (sum_i n_nodes_i) long indicating the diagonal of each matrix (optional, will be recomputed otherwise)
        only valid for order 2 (otherwise doesn't make sense )
    """
    _transpose_indices: CachedOrderIndepTensor = None
    """
        Only valid for order 2 (sum_i n_nodes_i^2), indices for the transpose of each matrix
    """

    _grid: CachedOrderIndepTensor = None

    def __init__(self, n_nodes):
        super().__init__(n_nodes)
        self._init_caches()

    # dummy code for type hinting
    def get_n(self, order: int|None) -> Tensor:
        del order
        assert False, "This code should be unreachable (because of the __getattribute__)"
    def get_ntotal(self, order: int|None) -> Tensor:
        del order
        assert False, "This code should be unreachable (because of the __getattribute__)"
    def get_max_n(self, order: int|None) -> Tensor:
        del order
        assert False, "This code should be unreachable (because of the __getattribute__)"
    def get_ptr(self, order: int|None) -> Tensor:
        del order
        assert False, "This code should be unreachable (because of the __getattribute__)"
    def get_batch(self, order: int|None) -> Tensor:
        del order
        assert False, "This code should be unreachable (because of the __getattribute__)"

    def _compute_n(self, order):
        if order == 1:
            return self.n_nodes
        return torch.pow(self.n_nodes, order)
    def _compute_batch(self, order):
        return torch.cat([i * torch.ones(n_i.item(), dtype=torch.long) for i, n_i in enumerate(self.get_n(order))]).to(self.device)#type: ignore
    def _compute_ptr(self, order):
        return ptr_from_sizes(self.get_n(order))
    def _compute_diagonal(self):
        diagonals = []
        for start, end, n_nodes in zip(self.get_ptr2(), self.get_ptr2()[1:], self.n_nodes):
            start = start.item()
            end= end.item()
            n_nodes = n_nodes.item()
            diagonal = torch.arange(start, end + 1, n_nodes + 1 )
            diagonals.append(diagonal)
        return torch.cat(diagonals, dim=0)
    def _compute_transpose_indices(self):
        indices = []
        for start, end, n_nodes in zip(self.get_ptr2(), self.get_ptr2()[1:], self.n_nodes):
            start = start.item()
            end= end.item()
            n_nodes = n_nodes.item()
            indices.append(torch.arange(start, end).reshape(n_nodes, n_nodes).T.reshape(-1))
        return torch.cat(indices, dim=0)

    def _compute_grid(self):
        ptr1 = self.get_ptr1()
        subgrids_x = []
        subgrids_y = []
        for n, ptr in zip(self.n_nodes, ptr1):
            n = n.item()
            ptr = ptr.item()
            arange = torch.arange(n, device = ptr1.device)
            arange = arange + ptr
            arange_x = arange[:, None].expand(n, n).reshape(-1)
            arange_y = arange[None, :].expand(n, n).reshape(-1)
            subgrids_x.append(arange_x)
            subgrids_y.append(arange_y)
        return torch.cat(subgrids_x, dim=0), torch.cat(subgrids_y, dim=0)

    def _compute_ntotal(self, order):
        return self.get_n(order).sum()
    def _compute_max_n(self, order):
        return self.get_n(order).max()

    def shapes(self, order: int|None) -> Iterator[tuple[int]]:
        for n in self.get_n(1):
            n = n.item()
            assert isinstance(n, int)
            yield (n,) * order

    @classmethod
    def from_other(cls, other: Self, device=None, to_kwargs={}):
        """move / shallow copy constructor"""

        def to(x, device):
            if device is None: return x
            if x is None: return x
            if isinstance(x, Tensor): return x.to(device, **to_kwargs)
            if isinstance(x, dict):
                return {order: y.to(device, **to_kwargs) for order, y in x.items()}
            if isinstance(x, tuple):
                return tuple(to(y, device) for y in x)
            raise ValueError

        batch = cls(to(other.n_nodes, device))

        for x, t in get_type_hints(cls).items():
            if not x.startswith("_"): continue
            if get_origin(t) is dict or t == (Tensor|None):
                other_value = getattr(other, x)
                other_value = to(other_value, device)
                setattr(batch, x, other_value)
        return batch

    def pin_memory(self):
        def pin_memory(x):
            if x is None: return
            if isinstance(x, Tensor): return x.pin_memory()
            if isinstance(x, dict):
                return {order: pin_memory(y) for order, y in x.items()}
            if isinstance(x, tuple):
                return tuple(pin_memory(y) for y in x)
            return
        for x, t in get_type_hints(self).items():
            if get_origin(t) is dict or t == (Tensor|None) or issubclass(t, Tensor):
                value  = getattr(self, x)
                setattr(self, x, pin_memory(value))

        return self


    @property
    def device(self):
        return self.n_nodes.device

    def to(self, device, **kwargs):
        return self.from_other(self, device, to_kwargs=kwargs)

    def __eq__(self, other: BatchIndicatorBase):
        if other is self: return True # might actually bring a performance gain, since I rarely copy batch indicators
        if not isinstance(other, BatchIndicatorBase): return NotImplemented
        return (self.n_nodes == other.n_nodes).all().item()

    def __repr__(self):
        return f"{self.__class__.__name__}(batch_size={self.get_batch_size()})"

class BatchIndicatorProduct(BatchIndicator):
    """Tensor product of batch indicators
    Order is undefined on this object, so it should always be set to None

    All sub-indicators have an associated order. 
    I would say it makes it easier to reason about to keep those at the default value of 1 (just use them several times if needed) — but do what thou wilt an it harm none
    """
    orders: tuple[int]
    indicators: tuple[BatchIndicator, ...]

    def __init__(self, *indicators: BatchIndicator, orders: Sequence[int]|None=None):
        from .utils import check_same_batch_size # here to avoid circular imports
        check_same_batch_size(*indicators, calling=self)
        self._init_caches()
        self.indicators = indicators
        if orders is not None:
            self.orders = tuple(orders)
        else:
            self.orders = tuple(1 for _ in indicators)


    @overload
    def _check_order_is_none(self, order:None) -> None:
        ...

    @overload
    def _check_order_is_none(self, order:int) -> NoReturn:
        ...

    def _check_order_is_none(self, order:Any):
        if order is not None:
            raise ValueError(f"order is undefined for {self.__class__}, "
                             f"it should always be set to None, got {order}")

    def _compute_n(self, order: int|None=None):
        self._check_order_is_none(order)
        all_n = [indicator.get_n(order) for indicator, order 
                 in zip(self.indicators, self.orders)]
        all_n = torch.stack(all_n, dim=-1)
        return torch.prod(all_n, dim=-1)
        
    #other _compute_*** functions call get_n() and will thus still be valid
    # inheritance is great sometimes

    def _compute_diagonal(self):
        raise ValueError(f"Diagonal doenst make sense with {self.__class__}")

    def _compute_grid(self, order=None):
        """ Returns
        G1, G2, ... G_K
        where K is the number of indicators in the product

        such as G_k[i] corresponds to which coordinates in the kth element of the 
        product the ith element in the data points to.

        This function is pure black magic, I sincerely hope there's no bugs in it
        """
        len_prod: int = len(self.indicators)
        all_ptrs: list[Tensor] = [indicator.get_ptr(order) 
                for indicator, order in zip(self.indicators, self.orders)]
        all_ns: list[Tensor] = [indicator.get_n(order) 
                for indicator, order in zip(self.indicators, self.orders)]
        device: torch.device = all_ptrs[0].device

        all_subgrids: list[list[Tensor]] = [[] for _ in range(len_prod)]

        for ns, ptrs in zip(zip(*all_ns), zip(*all_ptrs)):
            ns = [n.item() for n in ns]
            # ptrs = [ptr. for ptr in ptrs]
            aranges: list[Tensor] = [torch.arange(n, device = device) + ptr 
                       for n, ptr in zip(ns, ptrs)] 
            aranges = [
                    arange.reshape(*([1] * i), -1, *([1] * (len_prod - 1-i)))
                    for i, arange in enumerate(aranges)
                    ] # add empty dimensions  to prepare for the expand
            aranges = [arange.expand(*ns) for arange in aranges]
            for subgrid, arange in zip(all_subgrids, aranges):
                subgrid.append(arange.reshape(-1))
        return [torch.cat(subgrid, dim=0) for subgrid in all_subgrids]

    def shapes(self, order: int|None) -> Iterator[tuple[int]]:
        self._check_order_is_none(order)
        sub_shapes = [i.shapes(o) for (i, o) in zip(self.indicators, self.orders)]
        for sub_shapes_i in zip(*sub_shapes):
            flattened_shape = tuple(size for shape in sub_shapes_i 
                                    for size in shape)
            yield flattened_shape

    def get_batch_size(self):
        return len(self.get_n(None))


    @classmethod
    def from_other(cls, other: Self, device=None, to_kwargs= {}):
        """move / shallow copy constructor"""
        batch = cls(*[i.to(device, **to_kwargs) for i in other.indicators], orders=other.orders)
        return batch

    def __eq__(self, other: BatchIndicator):
        if not isinstance(other, BatchIndicatorProduct): return False
        return self.indicators == other.indicators and self.orders == other.orders

    def __repr__(self):
        return f"{self.__class__.__name__}{self.indicators}"

    @property
    def device(self):
        devices = [i.device for  i in self.indicators ]
        assert all_equal(*devices)
        return devices[0]
