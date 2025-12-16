#!/usr/bin/env python3
"""
PythonPyTorchCombined.py

Architecture:
2 → 6 → 2 → 2 → 8 → 2 → 2 → 6 → 2

Output modes:
A) regression   : 2 continuous outputs
B) softmax      : 2-class classification (mutually exclusive)
C) multilabel   : 2 independent probabilities
"""

from __future__ import annotations
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------
# Model definition
# -----------------------

class YoungNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
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
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)  # raw logits / regression outputs


# -----------------------
# Dummy data generators
# -----------------------

def make_dummy_data(mode: str, batch_size: int):
    x = torch.randn(batch_size, 2)

    if mode == "regression":
        y = torch.randn(batch_size, 2)

    elif mode == "softmax":
        y = torch.randint(0, 2, (batch_size,), dtype=torch.long)

    elif mode == "multilabel":
        y = torch.randint(0, 2, (batch_size, 2)).float()

    else:
        raise ValueError("Invalid mode")

    return x, y


# -----------------------
# Training loop
# -----------------------

def train(mode: str, epochs: int = 5, lr: float = 1e-3, batch_size: int = 64):
    model = YoungNet()

    if mode == "regression":
        criterion = nn.MSELoss()
    elif mode == "softmax":
        criterion = nn.CrossEntropyLoss()
    elif mode == "multilabel":
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError("Invalid mode")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        x, y = make_dummy_data(mode, batch_size)

        logits = model(x)

        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

    return model


# -----------------------
# Inference examples
# -----------------------

def infer(model: YoungNet, mode: str):
    model.eval()
    with torch.no_grad():
        x_test = torch.tensor([[0.1, -0.2],
                               [1.0,  0.5]])

        logits = model(x_test)

        if mode == "regression":
            preds = logits

        elif mode == "softmax":
            preds = torch.softmax(logits, dim=1)

        else:  # multilabel
            preds = torch.sigmoid(logits)

        print("\nPredictions:")
        print(preds)


# -----------------------
# Main
# -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["regression", "softmax", "multilabel"],
        default="regression",
        help="Output interpretation",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(0)

    model = train(
        mode=args.mode,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
    )

    infer(model, args.mode)

    print("\nTarget formatting reminder:")
    if args.mode == "regression":
        print("  y shape: (N,2) float32, real values")
    elif args.mode == "softmax":
        print("  y shape: (N,) int64 with values {0,1}")
    else:
        print("  y shape: (N,2) float32 with values {0,1}")


if __name__ == "__main__":
    main()
