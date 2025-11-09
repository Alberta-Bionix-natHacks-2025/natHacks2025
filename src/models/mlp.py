import torch
import torch.nn as nn


class EEGFeatureMLP(nn.Module):
    """
    MLP classifier for EEG feature vectors.
    Input: feature vector (batch_size, feature_dim)
    Output: logits (batch_size, n_classes)
    """

    def __init__(self, input_dim, hidden_dim=64, n_classes=2, dropout=0.2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, input_dim)
        """
        return self.net(x)


# run a quick internal test when executed directly
if __name__ == "__main__":
    model = EEGFeatureMLP(input_dim=14, hidden_dim=64, n_classes=2)

    dummy_input = torch.randn(5, 14)  # 5 samples, 14-dim feature vector
    output = model(dummy_input)

    print("Input shape: ", dummy_input.shape)
    print("Output shape:", output.shape)
    print("Output:", output)