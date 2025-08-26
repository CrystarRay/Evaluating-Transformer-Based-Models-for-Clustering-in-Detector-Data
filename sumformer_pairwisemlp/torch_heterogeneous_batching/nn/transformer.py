import warnings

from typing import TYPE_CHECKING, Literal
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch_geometric.utils import softmax as pyg_softmax, scatter

from ml_lib.misc import all_equal
from ml_lib.misc.torch_functions import ShouldBe as should_be
from ml_lib.models.layers import MLP, MultiInputLinear

if TYPE_CHECKING:
    from .sumformer import GlobalEmbedding

from torch_heterogeneous_batching.batch import Batch

class SelfAttention(nn.Module):
    """
    Self attention layer for batched sets
    """
    n_heads: int
    input_dim: int
    key_dim: int
    value_dim: int
    head_value_dim: int
    use_flash: bool

    query_key_value: nn.Linear
    out_proj: nn.Linear|None

    def __init__(self, input_dim, value_dim=None, key_dim=None, n_heads=1, 
                 use_out_proj=True, 
                 use_flash = True):
        """
        Computes self-attention on batched 

        Args:
            input_dim: dimension of the input
            value_dim: dimension of the output, must be divisible by n_heads. default: input_dim
            key_dim: dimension of the keys (default: value_dim / n_heads)

        """
        super().__init__()
        dtype_float = {"dtype": torch.float16} if use_flash else {}
        self.n_heads = n_heads
        self.input_dim = input_dim
        value_dim = value_dim or input_dim
        key_dim = key_dim or value_dim // n_heads
        assert value_dim % n_heads == 0, "value_dim must be divisible by n_heads"
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.head_value_dim = value_dim // n_heads
        self.query_key_value = nn.Linear(input_dim, 
            n_heads * (2 * key_dim  + self.head_value_dim), **dtype_float)
        if use_out_proj: self.out_proj = nn.Linear(value_dim, value_dim, **dtype_float)
        else: self.out_proj = None
        self.use_flash = use_flash


    def forward(self, x: Batch):
        # import pdb; pdb.set_trace()
        assert x.order == 1
        use_flash = getattr(self, "use_flash", 
                            self.key_dim == self.head_value_dim)
        if use_flash: 
            x_device = x.map(lambda y: y.to(torch.float16))
        else: x_device = x
        qkv = self.query_key_value(x_device).data
        if use_flash: 
            attention_results = self.flash_self_attention(qkv, x)
        else: 
            attention_results = self.naive_self_attention(qkv, x)
        if self.out_proj is not None:
            attention_results = attention_results.map(self.out_proj)
        attention_results = attention_results.map(lambda y: y.to(x.data.device))
        return attention_results

    def flash_self_attention(self, qkv, x):
            assert self.key_dim == self.head_value_dim
            from flash_attn import flash_attn_varlen_qkvpacked_func
            n, _ = qkv.shape
            assert n == x.data.shape[0]
            qkv = qkv.reshape(n, 3, self.n_heads, self.key_dim,) 
            attention_results = flash_attn_varlen_qkvpacked_func(
                qkv, 
                cu_seqlens=x.ptr.to(torch.int32), 
                max_seqlen=x.max_n, 
                causal=False, 
                    )   <~~~~~ should_be(n, self.n_heads, self.head_value_dim)
            attention_results = attention_results.contiguous().reshape(n, self.value_dim)
            return x.batch_like(attention_results)

    def naive_self_attention(self, qkv, x):
        """x only given for its indicator"""
        total_n_nodes = x.data.shape[0]
        indicator = x.indicator
        grid_x, grid_y = indicator.grid()
        n_edges, = grid_x.shape

        query, key, value = self.split_qkv(qkv)
        query_grid_x = query[grid_x] # n_edges, n_heads, query_dim
        key_grid_y = key[grid_y]  # n_edges, n_heads, query_dim
        value_grid_y = value[grid_y] # #n_edges, n_heads, head_value_dim

        attention_logits = (query_grid_x * key_grid_y).sum(dim=-1, keepdim=True) #n_edges, n_heads, 1
        assert attention_logits.shape == (n_edges, self.n_heads, 1)
        attention_logits = attention_logits / indicator.n_nodes[indicator.get_batch2()].sqrt()[..., None, None] #n_edges, n_heads, 1
        assert attention_logits.shape == (n_edges, self.n_heads, 1)
        #TODO Tristan check if index shouldn't be grid_y instead????
        # they should stay the same for varying y but same x
        # so no they’re grid_x
        attention_weights = pyg_softmax(attention_logits, index=grid_x, dim=0, num_nodes=total_n_nodes) #n_edges, n_heads, 1
        weighted_values = attention_weights * value_grid_y #n_edges, n_heads, head_value_dim
        assert weighted_values.shape == (n_edges, self.n_heads, self.head_value_dim)
        weighted_values_concat = weighted_values.reshape(-1, self.value_dim) #n_edges, value_dim

        attention_results = scatter(weighted_values_concat, index=grid_x, reduce="sum", dim_size=total_n_nodes) #n_nodes, value_dim
        attention_results = Batch.from_other(attention_results, x)
        return attention_results

    def split_qkv(self, qkv: Tensor):
        n, _ = qkv.shape
        if self.key_dim == self.head_value_dim:
            #qkv (n, 3*query_dim)
            # use 3, key_dim and not key_dim, 3, cf https://github.com/pytorch/pytorch/blob/758dbac308019395890f19ddcfdafa200feda04a/torch/nn/functional.py#L5512
            qkv = qkv.reshape(n, 3, self.n_heads, self.key_dim,) 
            query = qkv[:, 0, ...]
            key = qkv[:, 1, ...]
            value = qkv[:, 2, ...]
        else:
            warnings.warn("key_dim != head_value_dim, this case is not tested")
            qkv = qkv.reshape(n, self.n_heads, 2 * self.key_dim + self.head_value_dim)
            query = qkv[..., :self.key_dim]
            key = qkv[..., self.key_dim: 2 * self.key_dim]
            value = qkv[..., 2 * self.key_dim:]
        # key: n, n_heads, query_dim
        # query: n, n_heads, query_dim
        # value: n, n_heads, head_value_dim
        return query, key, value


class TransformerBlockBase(nn.Module):
    """implement the transformer block residual connections"""
    def __init__(self,  input_dim, *,  layer_norm_position: Literal["pre", "post", "none"]="pre"):
        super().__init__()
        assert layer_norm_position in ["pre", "post", "none"]
        self.layer_norm_position = layer_norm_position
        self.layer_norm = nn.LayerNorm(input_dim)
        
    def forward(self, x: Batch):
        y = x
        if self.layer_norm_position == "pre": y = y.map(self.layer_norm)
        y = self.inner(y)
        y = x + y
        if self.layer_norm_position == "post": y = self.layer_norm(y)
        return y

    def inner(self, x: Batch):
        del x
        raise NotImplementedError("This is an abstract class")

class TransformerBlock(TransformerBlockBase):
    """
    Inner TransformerBlock, with a residual connection and a layer norm.
    """
    def __init__(self, input_dim, key_dim=None, n_heads=1, layer_norm_position: Literal["pre", "post", "none"]="post"):
        super().__init__(input_dim, layer_norm_position=layer_norm_position)
        self.self_attention = SelfAttention(input_dim, value_dim=input_dim, key_dim=key_dim, n_heads=n_heads)
        
    def inner(self, x: Batch):
        y = self.self_attention(x)
        return y

class TransformerFeedForwardBlock(TransformerBlockBase):
    """
    Inner TransformerBlock, with a residual connection and a layer norm.
    """
    mlp: MLP
    def __init__(self, input_dim, hidden_dim=512, layer_norm_position: Literal["pre", "post", "none"]="post"):
        super().__init__(input_dim, layer_norm_position=layer_norm_position)
        self.mlp = MLP(input_dim, hidden_dim, hidden_dim, input_dim, batchnorm=False)
        
    def inner(self, x: Batch):
        y = x.map(self.mlp) 
        return y

class TransformerFeedForwardBlockWithGlobalEmbedding(TransformerFeedForwardBlock):
    """
    Inner TransformerBlock, with a residual connection and a layer norm.
    """
    global_embedding: "GlobalEmbedding"
    concat: MultiInputLinear

    def __init__(self, input_dim, hidden_dim=512, layer_norm_position: Literal["pre", "post", "none"]="pre"):
        from set2graph.layers.new.sumformer import GlobalEmbedding
        super().__init__(input_dim, hidden_dim, layer_norm_position=layer_norm_position)
        self.global_embedding = GlobalEmbedding(
                input_dim=input_dim, embed_dim=input_dim, 
                hidden_dim=hidden_dim, n_layers=3, 
                aggregation="MultiAggregation", 
                aggregation_args=dict(aggrs=["MeanAggregation", "SumAggregation", "MaxAggregation"],)
            )
        self.concat = MultiInputLinear([input_dim, input_dim], input_dim)

    def inner(self, x: Batch):
        embed = self.global_embedding(x)
        embed_back = embed[x.batch]
        x_embed = self.concat(x.data, embed_back)
        x_embed = F.leaky_relu(x_embed)
        y = self.mlp(x_embed)
        y = x.batch_like(y)
        return y

class TransformerFullBlock(nn.Module):
    def __init__(self, input_dim, key_dim=None, n_heads=1, n_layers=3, hidden_dim=512):
        super().__init__()
        self.input_dim = input_dim
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.attention = TransformerBlock(input_dim, key_dim, n_heads)
        self.feed_forward = TransformerFeedForwardBlock(input_dim, hidden_dim)

    def forward(self, x):
        with torch.autograd.profiler.record_function("attention"):
            y = self.attention(x)
        with torch.autograd.profiler.record_function("feed_forward"):
            y = self.feed_forward(x)
        return y

class Transformer(nn.Sequential):
    """decoder-only Transformer

    This is very barebones, no positional encoding (just add it to the input), no input linear (idem)
    intendend to be used as a component in a model
    """

    def __init__(self, input_dim, key_dim=None, n_heads=1, n_layers=3, hidden_dim=512, use_global_embedding=False, layer_norm_position: Literal["pre", "post", "none"]="post"):
        super().__init__()
        self.input_dim = input_dim
        self.n_heads = n_heads
        self.n_layers = n_layers

        ff_block_type = TransformerFeedForwardBlockWithGlobalEmbedding if use_global_embedding else TransformerFeedForwardBlock

        for i in range(n_layers):
            self.add_module(f"attention_{i}", TransformerBlock(input_dim, key_dim, n_heads, layer_norm_position=layer_norm_position))
            self.add_module(f"feed_forward_{i}", ff_block_type(input_dim, hidden_dim, layer_norm_position=layer_norm_position))


