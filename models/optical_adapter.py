import torch.nn as nn


class OpticalAdapter(nn.Module):
    """
    OpticalAdapter : projection des tokens visuels (dim 896)
    vers l'espace d'embedding de Qwen (dim 2048).

    Soit x ∈ ℝ^d avec d = 896 (in_dim).
    Adapter(x) = W₂ · φ( W₁ · RMSNorm(x) ) ∈ ℝ^D avec D = 2048 (out_dim).

    avec φ = SiLU (Sigmoid Linear Unit).
    """

    def __init__(self, in_dim=896, out_dim=2048):
        super().__init__()
        self.norm = nn.RMSNorm(in_dim)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),  # W₁ ∈ ℝ^{D×d}
            nn.SiLU(),  # φ(t) = t ⊙ σ(t)
            nn.Linear(out_dim, out_dim, bias=False),  # W₂ ∈ ℝ^{D×D}
        )

    def forward(self, x):
        return self.proj(self.norm(x))
