import torch
import torch.nn as nn
import torch.nn.functional as F

class YoungNet(nn.Module):
    # 2 → 6 → 2 → 2 → 8 → 2 → 2 → 6 → 2
    def __init__(self):
        super().__init__()
        self.fcs = nn.ModuleList([
            nn.Linear(2, 6),
            nn.Linear(6, 2),
            nn.Linear(2, 2),
            nn.Linear(2, 8),
            nn.Linear(8, 2),
            nn.Linear(2, 2),
            nn.Linear(2, 6),
            nn.Linear(6, 2),
        ])

    def forward(self, x):
        for layer in self.fcs[:-1]:
            x = F.relu(layer(x))
        return self.fcs[-1](x)  # logits (or raw regression outputs)
