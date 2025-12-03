import torch
import torch.nn as nn

class OdeSolverWrapper(nn.Module):
    def __init__(self, estimator, mask, mu, spks, cond):
        super().__init__()
        self.estimator = estimator
        # Store context variables
        self.mask = mask
        self.mu = mu
        self.spks = spks
        self.cond = cond

    def forward(self, t, x):
        """
        torchdiffeq calls this as forward(t, x).
        We map it to your original signature: estimator(x, mask, mu, t, spks, cond)
        """
        return self.estimator(x, self.mask, self.mu, t, self.spks, self.cond)