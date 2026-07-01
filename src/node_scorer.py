import torch
import torch.nn as nn


class NodeScorer(nn.Module):

    def __init__(self, input_dim=7, hidden_dim=32):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, gpu_features, job_features):
        scores = []

        for gpu in gpu_features:
            x = torch.cat([gpu, job_features])
            score = self.network(x)
            scores.append(score)

        return torch.stack(scores).squeeze(-1)