"""
Test file for EEGFeatureMLP
Run using:
    python -m testing.test_mlp
"""

import torch
import sys
import os

# Make src importable when running from project root
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from models.mlp import EEGFeatureMLP


def test_mlp_forward():
    print("=== Running MLP forward test ===")

    # Suppose your feature vector has 14 dims (CSP + PSD + asymmetry)
    input_dim = 14
    batch_size = 8

    model = EEGFeatureMLP(input_dim=input_dim, hidden_dim=64, n_classes=2)

    dummy = torch.randn(batch_size, input_dim)

    output = model(dummy)

    print("Input shape: ", dummy.shape)
    print("Output shape:", output.shape)
    print("Output tensor:")
    print(output)

    assert output.shape == (batch_size, 2), "Output shape mismatch!"

    print("✅ Test Passed!")


if __name__ == "__main__":
    test_mlp_forward()
