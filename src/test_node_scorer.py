import torch

from node_scorer import NodeScorer


model = NodeScorer(input_dim=7)

# GPU features:
# queue_length, busy, speed, memory, temperature
gpus_3 = torch.tensor([
    [0, 0, 1.0, 24, 40],
    [1, 1, 0.7, 16, 55],
    [2, 1, 0.5, 8, 70],
], dtype=torch.float32)

gpus_5 = torch.tensor([
    [0, 0, 1.0, 24, 40],
    [1, 1, 0.7, 16, 55],
    [2, 1, 0.5, 8, 70],
    [0, 0, 0.9, 12, 45],
    [3, 1, 0.4, 6, 60],
], dtype=torch.float32)

# Job features:
# runtime, memory_required
job = torch.tensor([30, 10], dtype=torch.float32)

scores_3 = model(gpus_3, job)
scores_5 = model(gpus_5, job)

print("Scores for 3 GPUs:")
print(scores_3)
print("Selected GPU:", torch.argmax(scores_3).item())

print("\nScores for 5 GPUs:")
print(scores_5)
print("Selected GPU:", torch.argmax(scores_5).item())