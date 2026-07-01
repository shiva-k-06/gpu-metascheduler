print("Starting Round Robin...")
from cluster import create_cluster
from workload import generate_jobs
from simulator import Simulator
from metrics import calculate_metrics


gpus = create_cluster()
jobs = generate_jobs()
sim = Simulator(gpus, jobs)

gpu_index = 0

for job in jobs:
    sim.assign_job_to_gpu(job, gpu_index)
    gpu_index = (gpu_index + 1) % len(gpus)

while len(sim.completed_jobs) < len(jobs):
    sim.step()

metrics = calculate_metrics(sim, len(jobs))

print("\n=== Round Robin Evaluation Metrics ===")
for key, value in metrics.items():
    print(key, ":", value)