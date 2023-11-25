import torch
import torch.nn.functional as F


# todo a problem here is how to get th explanation while training?
# the problem is that generating the explanation might mess up the regula loss calulation,e.g. forwards pass in the explanation
# messing up gradients


class RightForRightReasonsLoss(torch.nn.Module):
    def __init__(self, loss=F.nll_loss, lambda_=1):
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
        self.explanation_method = None
        self.loss = loss
        self.lambda_ = lambda_

    def forward(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        s_batch: torch.Tensor,
        explanation_method,
    ):
        """

        Parameters
        ----------
        x_batch : torch.Tensor
            The input batch
        y_batch: torch.Tensor
            The target batch
        s_batch: torch.Tensor
            The relevancy map, this map is 1 where the explanation should be 0

        explanation_method: Explanation
            The explanation method to use

        Returns
        -------

        """
        rrr_loss = 0
        # todo vectorize
        for x, y, s in zip(x_batch, y_batch, s_batch):
            explanation = explanation_method.explain(x, y)
            rrr_loss += torch.linalg.norm(explanation * s)

        rrr_loss /= len(x_batch)
        regular_loss = self.loss(x_batch, y_batch)
        return regular_loss + self.lambda_ * rrr_loss
