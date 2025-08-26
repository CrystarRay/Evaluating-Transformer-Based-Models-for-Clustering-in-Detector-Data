"""
Batch support

Batched input should be given in the following format:
    input2 = (sum_i n_nodes_i^2 , n_features) float
        contains the features of all matrices flattened and concatenated
        must be in order (ie the batch2 vector must be in ascending order)
    batch2  = (sum_i n_nodes_i^2) long indicating which batch each nodes belongs to. 
        must be in ascending order.
        Optional, will be recomputed otherwise.
    n_nodes  = (batch_size) long indicating the number of nodes in each batch (optional, will be recomputed otherwise)

either batch2 or n_nodes must be given if the output is batched 
(otherwise it will be assumed not batched). 
If both are given, they must be consistent.
"""
from typing import (Literal, Self, Final, Callable, 
                    ClassVar, Any, get_type_hints, get_origin, Sequence, 
                    overload, NoReturn, Iterator)
import torch
from torch import Tensor, LongTensor
import torch.nn.functional as F
from dataclasses import dataclass
from ml_lib.misc import all_equal

from torch_scatter import segment_csr as segment

from .internals import BatchError
from .indicator import BatchIndicator, BatchIndicatorBase, BatchIndicatorProduct

from logging import getLogger; log = getLogger(__name__)

@dataclass
class Batch:
    """
    Class representing a batch of sets/ square matrices / non-square matrices /higher order tensors
    of elements.

    As an example if a batch contains two sets of resp 5 and 7 vectors (all of these vectors are of dim 13)
    
    then 

    ``batch.data``: is of dim ``(5 + 7, 13)`` (contains all of the batch's data)
    ``batch.order``: is 1 because we have a batch of sets (it would be 2 for square matrices/2-tensors, and None for a more complicated shape (non-square))

    ``batch.n_features`` is 13

    ``batch.n`` is [5, 7] (the sizes of each element of the batch)
    ``batch.ptr`` is [0, 5, 5 + 7] (the limits of each element of the batch in the data representation)
    ``batch.batch`` is [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1] (such as batch.batch[i] tells us which elemnet of the batch batch.data[i] is in)
    ``batch.batch_size`` is 2 (the number of sets in the batch)

    Some torch functions can be applied directly to ``Batch`` object. They are listed in ``HANDLED_FUNCTIONS``. In addition simple arithmetic operations should work too (they get applied directly to the inner data).

    For reduction you can use ``batch.mean()``, ``batch.sum()`` or the more general ``batch.segment()`` which will reduce each element in the batch independently (so on our example, batch.sum() would return a (2, 13) tensor.
    For other functions you may use the ``map`` method or work directly on ``batch.data`` and then use ``batch.batch_like`` to recreate a new batch with the same structure and the new data.


    """
    data: Tensor
    """data: (sum_i n_nodes_i^order, n_features) float tensor"""
    order: int|None
    """1 or 2, depending on whether the data is a 1-tensor or a 2-tensor"""

    indicator: BatchIndicator

    def __post_init__(self):
        if not __debug__:
            return
        sum_nodes_i, _ = self.data.shape          
        sum_nodes_i_v2 = self.n_total
        assert sum_nodes_i == sum_nodes_i_v2

    @property
    def n_features(self):
        """Feature size of this batch"""
        return self.data.shape[-1]

    @property
    def n(self):
        """number of nodes / edges / elements in each element of the batch"""
        return self.indicator.get_n(self.order)

    @property
    def n_total(self):
        """this is n.sum()"""
        return self.indicator.get_ntotal(self.order)

    @property
    def max_n(self):
        """this is n.max()"""
        return self.indicator.get_max_n(self.order)

    @property
    def batch(self):
        """batch[i] is the set/graph/elemnt of the batch to which data[i] belongs"""
        return self.indicator.get_batch(self.order)

    @property
    def ptr(self):
        """ptr[i] indictates where the data for ith element in the batch starts in the data tensor"""
        return self.indicator.get_ptr(self.order)

    @property
    def n_nodes(self) -> Tensor:
        return self.indicator.n_nodes
    @property
    def n_edges(self)-> Tensor:
        return self.indicator.get_n_edges()
        
    @property
    def batch1(self)-> Tensor:
        return self.indicator.get_batch1()

    @property
    def batch2(self)-> Tensor:
        return self.indicator.get_batch2()

    @property
    def ptr1(self)-> Tensor:
        return self.indicator.get_ptr1()

    @property
    def ptr2(self)-> Tensor:
        return self.indicator.get_ptr2()

    @property
    def batch_size(self) -> int:
        """Number of elements in the batch (eg number of sets)"""
        return self.indicator.get_batch_size()

    @property
    def diagonal(self) -> Tensor:
        return self.indicator.get_diagonal()

    @classmethod
    def from_unbatched(cls, data: Tensor):
        order = data.ndim - 1
        *n_nodes_order_times, n_features = data.shape
        assert all_equal(*n_nodes_order_times)
        n_nodes = torch.as_tensor([n_nodes_order_times[0]], dtype=torch.long)
        batched_data = data.reshape(-1, n_features)
        return cls(data=batched_data, 
                   order=order, 
                   indicator=BatchIndicator(n_nodes=n_nodes))

    @classmethod
    def from_batched(cls, data: Tensor, n_nodes: Tensor, order: Literal[1, 2]):
        return cls(data=data,
                   order=order,
                   indicator=BatchIndicator(n_nodes=n_nodes))

    @classmethod
    def from_other(cls, data: Tensor, other: Self, order=None):
        return cls(data=data,
                   order=order or other.order,
                   indicator=other.indicator)

    def batch_like(self, other_data, order=None):
        """Creates a new batch object, with other data, but same indicator/order
        """
        return self.from_other(data=other_data, other=self, order=order)

    @classmethod
    def from_list(cls, data: list[Tensor], order: int|None|Literal["auto"]):
        all_n_features = []
        all_n_nodes = []
        if order == "auto":
            order = data[0].ndim - 1
        assert isinstance(order, int) or order is None
        if isinstance(order, int):
            for d in data:
                assert d.ndim == order + 1
                *n_nodes_order_times, n_features_ = d.shape
                assert all_equal(*n_nodes_order_times)
                n_nodes = n_nodes_order_times[0] 
                all_n_nodes.append(n_nodes)
                all_n_features.append(n_features_)
            data = [d.reshape(-1, d.shape[-1]) for d in data]
        # if order == 1:
        #     for d in data:
        #         assert d.ndim == 2
        #         n_nodes,  n_features_ = d.shape
        #         all_n_nodes.append(n_nodes)
        #         all_n_features.append(n_features_)
        # elif order == 2:
        #     for d in data:
        #         assert d.ndim == 3
        #         n_nodes, n_nodes_, n_features_ = d.shape
        #         assert n_nodes == n_nodes_
        #         all_n_nodes.append(n_nodes)
        #         all_n_features.append(n_features_)
        #     data = [d.reshape(-1, d.shape[-1]) for d in data]
        elif order is None:
            all_ndims = []
            for d in data:
                all_ndims.append(d.ndim)
                *n_nodes, n_features_ = d.shape
                all_n_nodes.append(n_nodes)
                all_n_features.append(n_features_)
            assert all_equal(*all_ndims)
            assert all_equal(*all_n_features)
            product_size = all_ndims[0] - 1
            n_features = all_n_features[0]
            n_nodes_tensor = torch.as_tensor(all_n_nodes) # batch_size, product_size
            indicators = [BatchIndicator(n_nodes_tensor[:, i]) 
                          for i in range(product_size)]
            orders = [1] * product_size
            product_indicator = BatchIndicatorProduct(*indicators, orders=orders)
            data = [d.reshape(-1, n_features) for d in data]
            concat_data = torch.cat(data, dim=0)
            return cls(concat_data, order=None, indicator=product_indicator.to(concat_data.device))

        assert all_equal(*all_n_features)
        concat_data = torch.cat(data, dim=0)
        all_n_nodes = torch.as_tensor(all_n_nodes, dtype=torch.long, 
                                      device=concat_data.device)
        return cls.from_batched(concat_data, all_n_nodes, order)

    def batch_split(self) -> Iterator[Tensor]:
        ptr = self.ptr
        order = self.order
        shapes = self.indicator.shapes(order)
        
        for begin, end, shape in zip(ptr, ptr[1:], shapes):
            begin = begin.item()
            end = end.item()
            d = self.data[begin:end, :]
            yield d.reshape(*shape, self.n_features)#type: ignore
            
    def same_batch(self, other: Self):
        return self.indicator == other.indicator

    def to_padded_array(self, pad_to=None):
        assert self.order == 1, "Only implemented for order 1 for now"
        if pad_to is None:
            pad_to = self.n.max().item()
            pad_to = int(pad_to)
        assert isinstance(pad_to, int)
        x_padded = torch.zeros((self.batch_size, pad_to, self.n_features))
        mask = torch.zeros((self.batch_size, pad_to), dtype=torch.bool)
        for i, v in enumerate(self):
            n, _ = v.shape
            x_padded[i, :n] = v
            mask[i, :n] = 1
        return x_padded, mask

    @classmethod
    def from_padded_array(cls, x_padded: Tensor, mask: Tensor):
        return cls.from_list([x_padded[i][mask[i]] for i in range(x_padded.size(0))], order=1)


    HANDLED_FUNCTIONS: ClassVar[dict[Callable, None|Callable[..., bool]]] = {
        torch.add: None, 
        torch.div: None,
        torch.sub: None,
        torch.mul: None, 
        torch.exp: None,
        F.layer_norm: None, 
        torch.square: None, 
        F.linear: None, 
        F.leaky_relu: None,
        F.relu: None,
        F.elu: None,
        F.sigmoid: None,
        torch.sigmoid: None,
        torch.gt: None, 
        torch.clamp: None, 
        torch.log: None, 
        }

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Magic code that makes torch functions work directly on Batch objects"""
        __tracebackhide__ = True
        if kwargs is None:
            kwargs = {}
        new_args = []
        new_kwargs = {}
        indicators = []
        orders = []

        ########################### check if the function is supported
        if func not in cls.HANDLED_FUNCTIONS:
            # log.warning(f"Function {func} called on a Batch object directly, but not in the supported list")
            return NotImplemented
        is_handled = cls.HANDLED_FUNCTIONS[func]
        if callable(is_handled) and not is_handled(*args, **kwargs):
            # log.warning(f"Function {func} called on a Batch object directly, but not supported with the given arguments")
            return NotImplemented

        ############################# extract the data from the batches
        for arg in args:
            if not isinstance(arg, Batch):
                new_args.append(arg)
                continue
            indicators.append(arg.indicator)
            orders.append(arg.order)
            new_args.append(arg.data)
        for k, v in kwargs.items():
            if not isinstance(v, Batch):
                new_kwargs[k] = v
                continue
            indicators.append(v.indicator)
            orders.append(v.order)
            new_kwargs[k] = v.data

        
        #################################check that all batches have the same structure
        if not all_equal(indicators):
            raise ValueError(f"Calling function {func} on Batch elements with different structure\n maybe you want to use the data attribute?")
        if not all_equal(orders):
            raise ValueError(f"Calling function {func} on Batch elements with different order\n maybe you want to use the data attribute?")
        indicator = indicators[0]
        order = orders[0]
        
        ##############################actual call
        result_data = func(*new_args, **new_kwargs)

        ##############################return
        assert result_data.ndim == 2
        return cls(data=result_data, order=order, indicator=indicator)

    def __add__(self, b) -> "Batch":
        return torch.add(self, b)#type: ignore
    def __radd__(self, b) -> "Batch":
        return torch.add(b, self)#type: ignore
    def __sub__(self, b) -> "Batch":
        return torch.sub(self, b)#type: ignore
    def __rsub__(self, b) -> "Batch":
        return torch.sub(b, self)#type: ignore
    def __mul__(self, b) -> "Batch":
        return torch.mul(self, b)#type: ignore
    def __rmul__(self, b) -> "Batch":
        return torch.mul( b, self)#type: ignore
    def __truediv__(self, b) -> "Batch":
        return torch.div(self, b)#type: ignore
    def __rtruediv__(self, b) -> "Batch":
        return torch.div(b, self)#type: ignore
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(data=[...], n_features={self.n_features}, "\
                                  f"indicator={self.indicator})"

    def to(self, *args, **kwargs):
        return type(self)(data=self.data.to(*args, **kwargs), 
                          order=self.order, 
                          indicator=self.indicator.to(*args, **kwargs))

    @property 
    def dtype(self):
        return self.data.dtype

    def __getitem__(self, i):
        assert 0 <= i < self.batch_size
        ptr = self.ptr
        order = self.order
        shapes = list(self.indicator.shapes(order))
        shape = shapes[i]
        begin = ptr[i].item()
        end = ptr[i+1].item()
        d = self.data[begin:end, :]
        return d.reshape(*shape, self.n_features)

    def __iter__(self):
        return self.batch_split()

    def __len__(self):
        return self.batch_size

    @property
    def T(self) -> "Batch":
        """Transpose"""
        assert self.order==2
        transpose_indices = self.indicator.get_transpose_indices()
        return Batch(self.data[transpose_indices], self.order, self.indicator)

    def segment(self, reduce: str="sum") -> Tensor:
        data = self.data
        if not torch.is_complex(self.data) or torch.is_floating_point(data):
            data = data.to(torch.float)
        return segment(data, indptr=self.ptr, reduce=reduce)

    # aliases
    reduce = segment

    def sum(self) -> Tensor:
        return self.segment("sum")

    def mean(self) -> Tensor:
        return self.segment("mean")

    def map(self, f: Callable[[Tensor], Tensor]):
        """x.map(f) will return a new batch, 
        with the same shape as x where f was applied to each element 
        """
        return self.batch_like(f(self.data))

    def pre_compute_indicator(self) -> None:
        """Pre-computes some useful values

        Useful to compute everything on CPU before moving the batch to GPU"""
        # yes this is kind of ugly, I am just calling the properties for side effects
        self.n_total;
        self.max_n;
        self.batch;
        self.ptr;


    def pin_memory(self):
        self.data = self.data.pin_memory()
        self.indicator = self.indicator.pin_memory()
        return self

