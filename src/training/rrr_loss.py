import torch

# todo a problem here is how to get th explanation while training?
# the problem is that generating the explanation might mess up the regula loss calulation,e.g. forwards pass in the explanation
# messing up gradients


class RightForRightReasonsLoss(torch.nn.Module):
    def __init__(self, explanation_method, loss):
        super().__init__()
        self.explanation_method = explanation_method

    def forward(self, x_batch, y_batch, s_batch):
        pass

