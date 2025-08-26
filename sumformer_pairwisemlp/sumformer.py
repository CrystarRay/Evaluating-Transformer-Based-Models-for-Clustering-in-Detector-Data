"""
Batched version of the sumformer
"""
from typing import Literal, Any
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ml_lib.misc.basic import eventually_tuple
from ml_lib.misc.typing import take_annotation_from
from ml_lib.misc.data_structures import Maybe
from ml_lib.models.layers import MLP, Repeat, ResidualShortcut
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.resolver import aggregation_resolver

from torch_heterogeneous_batching.batch import Batch


class GlobalEmbedding(nn.Module):

    input_dim: int
    embed_dim: int

    mlp: MLP
    r"""The MLP that changes the input features to be summed (\phi in the paper)"""
    activation: nn.Module
    r"""The last activation after that MLP"""
    aggregation: nn.Module
    r"""The aggregation function (sum or mean. resolved using torch_geometric.nn.resolver.aggregation_resolver, so the choices are the same as in torch_geometric.nn.aggr.Multi)"""

    def __init__(self, input_dim, embed_dim, hidden_dim=256, n_layers = 3, aggregation:str = "mean", aggregation_args={}):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.mlp = MLP(input_dim, *[hidden_dim]*n_layers, embed_dim, batchnorm=False, activation=nn.LeakyReLU)
        self.activation = nn.LeakyReLU()
        # if embed_dim == 3: 
        #     import pdb; pdb.set_trace()
        if "multi" in aggregation.lower(): 
            aggregation_args["mode"]= "proj"
            aggregation_args["mode_kwargs"] = {
                    "in_channels": embed_dim, 
                    "out_channels": embed_dim, 
                    **aggregation_args.get("mode_kwargs", {})}
        self.aggregation = aggregation_resolver(aggregation, **aggregation_args)

    def forward(self, x: Batch):
        node_embeddings = self.activation(self.mlp(x.data)) #n_nodes_total, key_dim
        return self.aggregation(node_embeddings, ptr=x.ptr) #batch_size, key_dim


class SumformerInnerBlock(nn.Module):
    """
    Here we implement the sumformer "attention" block (in quotes, because it is not really attention)
    It is permutation-equivariant
    and almost equivalent to a 2-step MPNN on a disconnected graph with a single witness node.

    We implement the MLP-sumformer (not the polynomial sumformer). Why?
        1. Simpler.
        2. They do say that polynomial tends to train better at the beginning, but the MLP catches up, 
            and it’s on synthetic functions which may perform very differently from real data 
            (and gives an advantage to the polynomial sumformer, which has fewer parameters).

    """

    input_dim: int
    """dimension of the input features"""

    key_dim: int
    """Dimesion of the aggregate sigma"""

    hidden_dim: int
    """Dimension of the hidden layers of the MLPs"""

    witness_coef: float
    """New: this network can take an input global embedding ("witness") 
        it will be added to the computed global embedding like so 
        new_embedding = (1 - witness_coef) * global_embedding * witness_coef*witness
    """

    aggreg_linear: nn.Linear
    psi: MLP
    
    def __init__(self: Any, input_dim: int, hidden_dim: int=512, key_dim :int = 256 , 
                 aggregation:str = "mean", aggregation_args: dict[str, Any]={}, 
                 node_embed_n_layers: int=3, output_n_layers: int =3,
                 witness_coef=.5): #self is typed as Any bc of @take annotations from
        super().__init__()
        self.input_dim = input_dim
        self.key_dim = key_dim
        self.hidden_dim = hidden_dim
        self.witness_coef = witness_coef
        self.global_embedding = GlobalEmbedding(
                input_dim=input_dim, embed_dim=key_dim, 
                hidden_dim=hidden_dim, n_layers=node_embed_n_layers, 
                aggregation=aggregation, aggregation_args=aggregation_args
        ) 

        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.aggreg_linear = nn.Linear(key_dim, hidden_dim)
        self.psi = MLP(hidden_dim, *[hidden_dim]*output_n_layers, input_dim, 
                          batchnorm=False, activation=nn.LeakyReLU)

    def forward(self, x: Tensor|Batch, witness:Tensor|None=None, 
                return_witness: bool|None=None, witness_as_maybe: bool = False):
        """This is a faster, equivalent formulation of the sumformer attention block.
        See my notes for the derivation (that i’ll transcribe to here at some point)

        Caution! This approximation may not be exact (but should still be universal)
        if the aggregation is not linear (ie sum or average).
        """
        if isinstance(x, Tensor): x = Batch.from_unbatched(x)
        assert isinstance(x, Batch)
        assert x.n_features == self.input_dim
        if return_witness is None: return_witness = witness is not None
        sigma = self.global_embedding(x)

        if witness is not None:
            sigma = (1-self.witness_coef) * sigma + self.witness_coef * witness

        sigma_hiddendim = self.aggreg_linear(sigma) #batch_size, hidden_dim
        x_hiddendim = self.input_linear(x.data) #n_nodes_total, hidden_dim
        
        psi_input = x_hiddendim + sigma_hiddendim[x.batch, :] #n_nodes_total, hidden_dim
        psi_input = F.leaky_relu(psi_input) #n_nodes_total, hidden_dim

        out = Batch.from_other(self.psi(psi_input), x) #n_nodes_total, input_dim
        if witness_as_maybe:
            if return_witness:
                out_witness = Maybe(sigma)
            else: out_witness = Maybe()
            return out, out_witness
        if return_witness:
            return out, sigma
        return out

class SumformerBlock(nn.Module):
    """
    Inner SumformerBlock, with a residual connection and a layer norm.
    """
    
    @take_annotation_from(SumformerInnerBlock.__init__)
    def __init__(self, *block_args, **block_kwargs):
        super().__init__()
        self.block = SumformerInnerBlock(*block_args, **block_kwargs)
        self.norm  = nn.LayerNorm(self.block.input_dim)

    def forward(self, x: Tensor|Batch, *args, **kwargs):
        residual, eventually_witness = self.block(x, *args, **kwargs, witness_as_maybe=True)
        result = self.norm(x + residual)
        return eventually_tuple(result, *eventually_witness)

class Sumformer(Repeat):
    def __init__(self, num_blocks: int, *block_args, **block_kwargs):
        make_block = lambda: SumformerBlock(*block_args, **block_kwargs)
        super().__init__(num_blocks, make_block)
