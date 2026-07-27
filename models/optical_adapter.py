import torch.nn as nn

class OpticalAdapter(nn.Module):
    def __init__(self, in_dim=896, out_dim=2048):
        super().__init__()
        self.norm = nn.RMSNorm(in_dim)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim, bias=False),
        )

    def forward(self, x):
        return self.proj(self.norm(x))
