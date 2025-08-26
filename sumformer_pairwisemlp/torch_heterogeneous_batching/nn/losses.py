import torch
from torch import nn, Tensor
from torch_heterogeneous_batching.batch import Batch, check_same_batch_size

from ml_lib.datasets.feature_specification import FeatureSpecification

class BatchMSELoss(nn.Module):
    """MSE Loss for batched sets
    instead of being the mean ovver the union of the sets, it is the mean of the mean in each set (and thus avoids overponderating bigger sets)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Batch, target: Batch, weights: Batch|None=None):
        check_same_batch_size(x, target, weights, none_ok=True)
        unreduced_loss = (x - target).data.square().mean(-1)
        if weights is not None:
            unreduced_loss = unreduced_loss * weights.data.squeeze(-1)
        error = unreduced_loss.mean()
        error = error.mean()
        return error

class BatchLoss(nn.Module):
    """More general loss for batched sets.
    Uses a ml_lib.FeatureSpecification, and the mean is not overponderated on bigger sets as in MSELoss
    """

    feature_spec: FeatureSpecification

    def __init__(self, feature_spec: FeatureSpecification):
        super().__init__()
        self.feature_spec = feature_spec

    def forward(self, x: Batch, target: Batch):
        check_same_batch_size(x, target)
        error: Tensor = self.feature_spec.compute_loss(x.data, target.data, reduce=False)
        error = error.mean()
        error = error.mean()
        return error


class BatchGeomLoss(nn.Module):
    """For now we use a terrible non-batched implementation.
    Padding may help, but ultimately, thisis kinda dependent on what is discussed in 
    https://github.com/jeanfeydy/geomloss/pull/35#issuecomment-806162285
    """

    samples_loss: "geomloss.SamplesLoss"

    def __init__(self, *samples_loss_args,  **samples_loss_kwargs):
        super().__init__()
        from geomloss import SamplesLoss
        self.samples_loss = SamplesLoss(*samples_loss_args, **samples_loss_kwargs)

    def forward(self, x: Batch, target: Batch):
        individual_losses = []
        for set_x, set_y in zip(x, target):
            individual_losses.append(self.samples_loss(set_x, set_y))
        error = torch.stack(individual_losses).mean()
        return error

