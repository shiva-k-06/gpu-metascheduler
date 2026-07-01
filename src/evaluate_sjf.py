from cluster import create_cluster
from workload import generate_jobs
from simulator import Simulator
from metrics import calculate_metrics


gpus = create_cluster()
jobs = generate_jobs()

jobs = sorted(jobs, key=lambda job: job.runtime)

sim = Simulator(gpus, jobs)

for job in jobs:

    best_gpu = min(
        gpus,
        key=lambda gpu: len(gpu.queue)
    )

    sim.assign_job_to_gpu(job, best_gpu.gpu_id)

while len(sim.completed_jobs) < len(jobs):
    sim.step()

metrics = calculate_metrics(sim, len(jobs))

print("\n=== SJF Evaluation Metrics ===")
for key, value in metrics.items():
    print(key, ":", value)