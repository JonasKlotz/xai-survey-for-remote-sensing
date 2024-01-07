import torch


# todo a problem here is how to get th explanation while training?
# the problem is that generating the explanation might mess up the regula loss calulation,e.g. forwards pass in the explanation
# messing up gradients


class RightForRightReasonsLoss(torch.nn.Module):
    def __init__(self, lambda_=1, explanation_method=None):
        """

        The right for the right reasons loss is defined as:

        ::math::
        \mathcal{L}_{rrr} (f_{\theta^t}(x_i), y_i) = \mathcal{L}_{pred}(f_{\theta^t}(x_i), y_i) + \lambda \mathcal{L}_{reason} (r_i^{l,t}, a_i^l)

        Parameters
        ----------
        loss: torch.nn.Module
            The other loss function to use
        model: torch.nn.Module
            The model to explain
        lambda_: float
            The weight of the explanation loss
        """
        super().__init__()
        self.explanation_method = explanation_method
        self.lambda_ = lambda_

    def forward(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        s_batch: torch.Tensor,
        regular_loss_value: float = 0,
    ):
        """

        Parameters
        ----------
        regular_loss_value
        x_batch : torch.Tensor
            The input batch
        y_batch: torch.Tensor
            The target batch
        s_batch: torch.Tensor
            The relevancy map, this map is 1 where the explanation should be 0

        Returns
        -------

        """
        attrs = self.explanation_method.explain_batch(x_batch, y_batch)
        print(f"attrs: {attrs.shape}")
        print(f"s_batch: {s_batch.shape}")

        # convert sbatch into relevancy maps (0 where the explanation should be 0)

        rrr_loss = torch.linalg.norm(attrs * s_batch)
        rrr_loss /= len(x_batch)
        return regular_loss_value + self.lambda_ * rrr_loss


def segmentations_to_relevancy_map(y_batch, s_batch):
    """
    Convert a batch of segmentations to a relevancy map

    Parameters
    ----------
    s_batch: torch.Tensor
        The batch of segmentations

    Returns
    -------
    relevancy_map: torch.Tensor
        The relevancy map
    """
    pass
